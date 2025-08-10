# triplet_distributed_training.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import pandas as pd
import numpy as np
import os
import torch.distributed as dist
from pyspark.ml.torch.distributor import TorchDistributor

# ---------------- Dataset ----------------
class TripletNameDataset(Dataset):
    def __init__(self, triplets_df, max_len=None):
        self.triplets = triplets_df.reset_index(drop=True)
        all_text = ''.join(self.triplets['anchor'].tolist() +
                           self.triplets['positive'].tolist() +
                           self.triplets['negative'].tolist())
        all_chars = set(all_text) | {'_'}
        self.char_to_idx = {char: idx for idx, char in enumerate(sorted(all_chars))}
        if max_len is None:
            max_len = int(max(
                self.triplets['anchor'].str.len().max(),
                self.triplets['positive'].str.len().max(),
                self.triplets['negative'].str.len().max()
            ))
        self.max_len = max_len

    def _pad_and_convert(self, name):
        padded = str(name).ljust(self.max_len, '_')
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

# ---------------- Model ----------------
class ImprovedCharacterEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_len, dropout_rate=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        pos_encoding = self._create_positional_encoding(max_len, embedding_dim)
        self.register_buffer('pos_encoding', pos_encoding)

        self.conv_layers = nn.Sequential(
            nn.Conv1d(embedding_dim, embedding_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(embedding_dim * 2),
            nn.ReLU(),
            nn.Conv1d(embedding_dim * 2, embedding_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(embedding_dim * 2),
            nn.ReLU(),
        )

        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim * 2, num_heads=4, dropout=dropout_rate)
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
        pos = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        return pe

    def forward(self, x):
        embedded = self.embedding(x) + self.pos_encoding.unsqueeze(0).to(x.device)
        conv_input = embedded.transpose(1, 2)
        conv_output = self.conv_layers(conv_input)

        attention_input = conv_output.transpose(1, 2)
        attn_out, _ = self.attention(attention_input.transpose(0, 1),
                                     attention_input.transpose(0, 1),
                                     attention_input.transpose(0, 1))
        attn_out = attn_out.transpose(0, 1)
        attn_out = self.attention_bn(attn_out.transpose(1, 2)).transpose(1, 2)

        flattened = attn_out.reshape(attn_out.size(0), -1)
        return nn.functional.normalize(self.fc_layers(flattened), p=2, dim=1)

# ---------------- Distributed Training ----------------
def distributed_train_fn(checkpoint_dir=None):
    # 1. Init process group
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Load or generate training data (must be done on each worker)
    data = pd.DataFrame({
        'anchor': ['John', 'Maria', 'Peter', 'Michael', 'Elizabeth'] * 100,
        'positive': ['Jon', 'Marie', 'Pete', 'Mike', 'Elisabeth'] * 100,
        'negative': ['Sarah', 'David', 'Emma', 'Thomas', 'Catherine'] * 100
    })

    dataset = TripletNameDataset(data)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

    # 3. Build model & wrap in DDP
    model = ImprovedCharacterEmbeddingModel(
        vocab_size=len(dataset.char_to_idx),
        embedding_dim=64,
        max_len=dataset.max_len
    ).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank] if device.type == "cuda" else None)

    # 4. Loss, optimizer
    def cosine_distance(x, y):
        return 1.0 - nn.functional.cosine_similarity(x, y, dim=1)

    criterion = nn.TripletMarginWithDistanceLoss(margin=0.5, distance_function=cosine_distance)
    optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.001)

    # 5. Training loop
    epochs = 5
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        model.train()
        total_loss = 0
        for batch in dataloader:
            anchor = batch['anchor'].to(device)
            positive = batch['positive'].to(device)
            negative = batch['negative'].to(device)

            optimizer.zero_grad()
            loss = criterion(model(anchor), model(positive), model(negative))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        if rank == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    dist.destroy_process_group()

# ---------------- Spark launcher ----------------
if __name__ == "__main__":
    TorchDistributor(num_processes=4, local_mode=False, use_gpu=True).run(distributed_train_fn)
