# fixed_pyspark_triplet_training.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ---- Dataset ----
class TripletNameDataset(Dataset):
    """
    Expects a pandas DataFrame with columns: 'anchor', 'positive', 'negative'.
    On a Spark cluster you can pass a Spark DataFrame; train_triplet_model will convert it to pandas.
    """
    def __init__(self, triplets_df, max_len=None):
        # triplets_df must be pandas.DataFrame
        self.triplets = triplets_df.reset_index(drop=True)

        # build character set deterministically
        all_text = ''.join(self.triplets['anchor'].astype(str).tolist() +
                           self.triplets['positive'].astype(str).tolist() +
                           self.triplets['negative'].astype(str).tolist())
        all_chars = set(all_text) | {'_'}
        self.char_to_idx = {char: idx for idx, char in enumerate(sorted(all_chars))}

        if max_len is None:
            max_len = int(max(
                self.triplets['anchor'].astype(str).str.len().max(),
                self.triplets['positive'].astype(str).str.len().max(),
                self.triplets['negative'].astype(str).str.len().max()
            ))
        self.max_len = max_len

    def _pad_and_convert(self, name):
        # ensure name is str
        name = str(name)
        padded = name.ljust(self.max_len, '_')
        return torch.tensor([self.char_to_idx.get(c, self.char_to_idx['_']) for c in padded], dtype=torch.long)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        row = self.triplets.iloc[idx]
        return {
            'anchor': self._pad_and_convert(row['anchor']),
            'positive': self._pad_and_convert(row['positive']),
            'negative': self._pad_and_convert(row['negative'])
        }

# ---- Model ----
class ImprovedCharacterEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_len, dropout_rate=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        pos_encoding = self._create_positional_encoding(max_len, embedding_dim)
        # register buffer so it's moved with model.to(device)
        self.register_buffer('pos_encoding', pos_encoding)

        self.conv_layers = nn.Sequential(
            nn.Conv1d(embedding_dim, embedding_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(embedding_dim * 2),
            nn.ReLU(),
            nn.Conv1d(embedding_dim * 2, embedding_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(embedding_dim * 2),
            nn.ReLU(),
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim * 2,
            num_heads=4,
            dropout=dropout_rate,
            batch_first=False  # we'll transpose to (seq, batch, embed)
        )

        self.attention_bn = nn.BatchNorm1d(embedding_dim * 2)

        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2 * max_len, embedding_dim * 4),
            nn.LayerNorm(embedding_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim * 4, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )

        self.max_len = max_len
        self.embedding_dim = embedding_dim

    def _create_positional_encoding(self, max_len, d_model):
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        return pe  # shape (max_len, d_model)

    def forward(self, x):
        # x: (batch, seq_len) long tensor
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)
        # make sure pos_encoding broadcasts correctly: pos_encoding is (max_len, d_model)
        embedded = embedded + self.pos_encoding.unsqueeze(0).to(embedded.device)

        conv_input = embedded.transpose(1, 2)  # (batch, channels=embed_dim, seq_len)
        conv_output = self.conv_layers(conv_input)  # (batch, channels2, seq_len)

        attention_input = conv_output.transpose(1, 2)  # (batch, seq_len, channels2)
        # MultiheadAttention expects (seq_len, batch, embed) when batch_first=False
        att_in = attention_input.transpose(0, 1)  # (seq_len, batch, embed)
        att_out, _ = self.attention(att_in, att_in, att_in)
        att_out = att_out.transpose(0, 1)  # (batch, seq_len, embed)
        # batchnorm expects (batch, channels, seq_len)
        att_bn_in = att_out.transpose(1, 2)  # (batch, channels, seq_len)
        att_bn = self.attention_bn(att_bn_in).transpose(1, 2)  # back to (batch, seq_len, embed)

        flattened = att_bn.reshape(att_bn.size(0), -1)
        output = self.fc_layers(flattened)

        return nn.functional.normalize(output, p=2, dim=1)

# ---- helper (unused but kept) ----
def get_triplet_mask(labels):
    indices_equal = torch.eye(labels.size(0), dtype=torch.bool)
    indices_not_equal = ~indices_equal
    i_not_equal_j = indices_not_equal.unsqueeze(2)
    i_not_equal_k = indices_not_equal.unsqueeze(1)
    j_not_equal_k = indices_not_equal.unsqueeze(0)
    distinct_indices = (i_not_equal_j & i_not_equal_k & j_not_equal_k)
    return distinct_indices

# ---- training ----
def train_triplet_model(triplets_df, embedding_dim=64, batch_size=32, epochs=20, device=None):
    """
    triplets_df: either a pandas DataFrame or a pyspark.sql.DataFrame.
    Returns: model (on CPU), char_to_idx, max_len
    """
    # Detect Spark DataFrame and convert if necessary (safe to call even if no Spark present)
    try:
        from pyspark.sql import DataFrame as SparkDF
    except Exception:
        SparkDF = None

    if SparkDF is not None and isinstance(triplets_df, SparkDF):
        # collect to driver; be cautious with very large datasets
        triplets_df = triplets_df.toPandas()

    if not isinstance(triplets_df, pd.DataFrame):
        raise ValueError("triplets_df must be a pandas DataFrame or a pyspark.sql.DataFrame")

    # device handling
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    dataset = TripletNameDataset(triplets_df)
    # num_workers=0 for Spark/cluster safety; set pin_memory if CUDA available
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                            pin_memory=(device.type == "cuda"))

    model = ImprovedCharacterEmbeddingModel(
        vocab_size=len(dataset.char_to_idx),
        embedding_dim=embedding_dim,
        max_len=dataset.max_len
    ).to(device)

    # define distance function for TripletMarginWithDistanceLoss
    def cosine_distance(x, y):
        # x,y are (batch, dim). cosine_similarity returns (batch,)
        return 1.0 - nn.functional.cosine_similarity(x, y, dim=1)

    criterion = nn.TripletMarginWithDistanceLoss(
        margin=0.5,
        distance_function=cosine_distance,
        reduction='mean'
    )

    learning_rate = 0.005
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.001)

    # scheduler requires steps_per_epoch > 0
    steps_per_epoch = max(1, len(dataloader))
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in dataloader:
            optimizer.zero_grad()

            # move batch to device
            anchor = batch['anchor'].to(device)
            positive = batch['positive'].to(device)
            negative = batch['negative'].to(device)

            anchor_emb = model(anchor)
            positive_emb = model(positive)
            negative_emb = model(negative)

            loss = criterion(anchor_emb, positive_emb, negative_emb)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            # guard scheduler.step so it doesn't error on very small datasets
            try:
                scheduler.step()
            except Exception:
                pass

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(dataloader))
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

    # move model to CPU before returning so it can be serialized/saved easily
    model.to(torch.device("cpu"))
    return model, dataset.char_to_idx, dataset.max_len

# ---- helper to get single name embedding ----
def get_name_embedding(model, name, char_to_idx, max_len):
    """
    model expected on CPU or device; we'll move inputs to the model device briefly.
    Returns numpy array of shape (1, embedding_dim)
    """
    model_device = next(model.parameters()).device
    device = model_device if model_device is not None else torch.device("cpu")

    padded = str(name).ljust(max_len, '_')
    indices = torch.tensor([char_to_idx.get(c, char_to_idx['_']) for c in padded], dtype=torch.long).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        emb = model(indices)  # (1, dim)
    emb_cpu = emb.detach().cpu().numpy()
    return emb_cpu

# ---- example usage ----
if __name__ == "__main__":
    # small example dataset; in practice this could be a pyspark DataFrame collected to driver
    triplets_df = pd.DataFrame({
        'anchor': ['John', 'Maria', 'Peter', 'Michael', 'Elizabeth'],
        'positive': ['Jon', 'Marie', 'Pete', 'Mike', 'Elisabeth'],
        'negative': ['Sarah', 'David', 'Emma', 'Thomas', 'Catherine']
    })

    model, char_to_idx, max_len = train_triplet_model(triplets_df, embedding_dim=64, batch_size=2, epochs=6)

    name1, name2 = "John", "Jon"
    emb1 = get_name_embedding(model, name1, char_to_idx, max_len)  # shape (1, D)
    emb2 = get_name_embedding(model, name2, char_to_idx, max_len)

    # compute cosine similarity correctly on flattened vectors
    v1 = emb1[0]
    v2 = emb2[0]
    similarity = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12))
    print(f"Similarity between {name1} and {name2}: {similarity:.4f}")
