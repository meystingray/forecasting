import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, collect_list, size, max as spark_max
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, DoubleType
from pyspark.ml.feature import StringIndexer
import pyspark.sql.functions as F
from typing import Iterator, Tuple, Dict, Any
import pickle
import os

class TripletNameDataset(Dataset):
    def __init__(self, triplets_data, char_to_idx, max_len):
        """
        triplets_data: list of dictionaries with 'anchor', 'positive', 'negative' keys
        """
        self.triplets = triplets_data
        self.char_to_idx = char_to_idx
        self.max_len = max_len
    
    def _pad_and_convert(self, name):
        padded = name.ljust(self.max_len, '_')
        return torch.tensor([self.char_to_idx.get(c, self.char_to_idx['_']) for c in padded])
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        row = self.triplets[idx]
        return {
            'anchor': self._pad_and_convert(row['anchor']),
            'positive': self._pad_and_convert(row['positive']),
            'negative': self._pad_and_convert(row['negative'])
        }

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
        
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim * 2,
            num_heads=4,
            dropout=dropout_rate
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
        pos = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        return pe

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded + self.pos_encoding
        
        conv_input = embedded.transpose(1, 2)
        conv_output = self.conv_layers(conv_input)
        
        attention_input = conv_output.transpose(1, 2)
        attention_output, _ = self.attention(
            attention_input.transpose(0, 1),
            attention_input.transpose(0, 1),
            attention_input.transpose(0, 1)
        )
        attention_output = attention_output.transpose(0, 1)
        attention_output = self.attention_bn(attention_output.transpose(1, 2)).transpose(1, 2)
        
        flattened = attention_output.reshape(attention_output.size(0), -1)
        output = self.fc_layers(flattened)
        
        return nn.functional.normalize(output, p=2, dim=1)

def create_character_vocabulary(spark_df):
    """Create character vocabulary from all names in the dataset"""
    # Collect all unique characters from anchor, positive, and negative columns
    chars_df = spark_df.select(
        F.explode(F.split(F.concat_ws("", col("anchor"), col("positive"), col("negative")), "")).alias("char")
    ).distinct()
    
    # Add padding character
    padding_df = spark.createDataFrame([("_",)], ["char"])
    all_chars_df = chars_df.union(padding_df).distinct()
    
    # Create character to index mapping
    chars_list = [row.char for row in all_chars_df.collect()]
    char_to_idx = {char: idx for idx, char in enumerate(sorted(chars_list))}
    
    return char_to_idx

def get_max_length(spark_df):
    """Get maximum name length from the dataset"""
    max_len_df = spark_df.select(
        spark_max(F.length(col("anchor"))).alias("max_anchor"),
        spark_max(F.length(col("positive"))).alias("max_positive"),
        spark_max(F.length(col("negative"))).alias("max_negative")
    ).collect()[0]
    
    return max(max_len_df.max_anchor, max_len_df.max_positive, max_len_df.max_negative)

def train_partition(partition_data: Iterator[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
    """Train model on a partition of data"""
    # Convert iterator to list to work with multiple times
    partition_list = list(partition_data)
    if not partition_list:
        return iter([])
    
    # Extract training parameters from broadcast variables
    # These would be set in the main training function
    char_to_idx = partition_list[0]['char_to_idx']
    max_len = partition_list[0]['max_len']
    embedding_dim = partition_list[0]['embedding_dim']
    batch_size = partition_list[0]['batch_size']
    epochs = partition_list[0]['epochs']
    
    # Extract actual triplet data
    triplets_data = []
    for row in partition_list:
        if 'anchor' in row:  # Skip metadata rows
            triplets_data.append({
                'anchor': row['anchor'],
                'positive': row['positive'],
                'negative': row['negative']
            })
    
    if not triplets_data:
        return iter([])
    
    # Create dataset and dataloader
    dataset = TripletNameDataset(triplets_data, char_to_idx, max_len)
    dataloader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)
    
    # Initialize model
    model = ImprovedCharacterEmbeddingModel(
        vocab_size=len(char_to_idx),
        embedding_dim=embedding_dim,
        max_len=max_len
    )
    
    criterion = nn.TripletMarginWithDistanceLoss(
        margin=0.5,
        distance_function=lambda x, y: 1.0 - nn.functional.cosine_similarity(x, y),
        reduction='mean'
    )
    
    learning_rate = 0.005
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.001)
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=epochs,
        steps_per_epoch=len(dataloader),
        pct_start=0.3
    )
    
    # Training loop
    total_loss = 0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for batch in dataloader:
            optimizer.zero_grad()
            
            anchor_emb = model(batch['anchor'])
            positive_emb = model(batch['positive'])
            negative_emb = model(batch['negative'])
            
            loss = criterion(anchor_emb, positive_emb, negative_emb)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
        
        total_loss += epoch_loss / len(dataloader)
    
    avg_loss = total_loss / epochs
    
    # Serialize model state dict
    model_state = {
        'state_dict': model.state_dict(),
        'vocab_size': len(char_to_idx),
        'embedding_dim': embedding_dim,
        'max_len': max_len,
        'loss': avg_loss
    }
    
    # Return model state as pickled bytes
    yield {
        'model_state': pickle.dumps(model_state),
        'partition_loss': avg_loss,
        'num_samples': len(triplets_data)
    }

def aggregate_models(model_states, char_to_idx, max_len, embedding_dim):
    """Aggregate models from different partitions using averaging"""
    if not model_states:
        raise ValueError("No model states to aggregate")
    
    # Initialize the aggregated model
    aggregated_model = ImprovedCharacterEmbeddingModel(
        vocab_size=len(char_to_idx),
        embedding_dim=embedding_dim,
        max_len=max_len
    )
    
    # Calculate weighted average of model parameters
    total_samples = sum(state['num_samples'] for state in model_states)
    aggregated_state_dict = {}
    
    for i, model_state_info in enumerate(model_states):
        model_state = pickle.loads(model_state_info['model_state'])
        weight = model_state_info['num_samples'] / total_samples
        
        for param_name, param_value in model_state['state_dict'].items():
            if i == 0:
                aggregated_state_dict[param_name] = param_value * weight
            else:
                aggregated_state_dict[param_name] += param_value * weight
    
    aggregated_model.load_state_dict(aggregated_state_dict)
    return aggregated_model

def train_triplet_model_distributed(spark, triplets_df, embedding_dim=64, batch_size=32, epochs=20, num_partitions=4):
    """
    Train triplet model using distributed processing with PySpark
    
    Args:
        spark: SparkSession
        triplets_df: Spark DataFrame with columns ['anchor', 'positive', 'negative']
        embedding_dim: Embedding dimension
        batch_size: Batch size for training
        epochs: Number of training epochs
        num_partitions: Number of partitions for distributed training
    """
    
    # Create character vocabulary and get max length
    char_to_idx = create_character_vocabulary(triplets_df)
    max_len = get_max_length(triplets_df)
    
    print(f"Vocabulary size: {len(char_to_idx)}")
    print(f"Max name length: {max_len}")
    
    # Repartition the data for distributed training
    triplets_df = triplets_df.repartition(num_partitions)
    
    # Add metadata to each partition
    def add_metadata(partition):
        for row in partition:
            # Yield metadata first
            yield {
                'char_to_idx': char_to_idx,
                'max_len': max_len,
                'embedding_dim': embedding_dim,
                'batch_size': batch_size,
                'epochs': epochs
            }
            # Then yield the actual data
            yield {
                'anchor': row.anchor,
                'positive': row.positive,
                'negative': row.negative
            }
    
    # Convert to RDD and apply training function to each partition
    triplets_rdd = triplets_df.rdd
    metadata_rdd = triplets_rdd.mapPartitions(add_metadata)
    
    # Train models on each partition
    trained_models_rdd = metadata_rdd.mapPartitions(train_partition)
    
    # Collect results from all partitions
    model_states = trained_models_rdd.collect()
    
    print(f"Trained {len(model_states)} partition models")
    
    # Aggregate models from all partitions
    aggregated_model = aggregate_models(model_states, char_to_idx, max_len, embedding_dim)
    
    # Calculate average loss
    avg_loss = sum(state['partition_loss'] for state in model_states) / len(model_states)
    print(f"Average training loss: {avg_loss:.4f}")
    
    return aggregated_model, char_to_idx, max_len

def get_name_embedding(model, name, char_to_idx, max_len):
    """Get embedding for a single name"""
    model.eval()
    padded = name.ljust(max_len, '_')
    indices = torch.tensor([char_to_idx.get(c, char_to_idx['_']) for c in padded])
    
    with torch.no_grad():
        indices_tensor = indices.unsqueeze(0)
        embedding = model(indices_tensor)
    
    return embedding.numpy()

def get_name_embeddings_distributed(spark, model, names_df, char_to_idx, max_len):
    """Get embeddings for multiple names using distributed processing"""
    
    # Broadcast model parameters
    model_state = model.state_dict()
    vocab_size = len(char_to_idx)
    embedding_dim = model.embedding_dim
    
    def compute_embeddings(partition):
        """Compute embeddings for a partition of names"""
        # Recreate model on each worker
        local_model = ImprovedCharacterEmbeddingModel(vocab_size, embedding_dim, max_len)
        local_model.load_state_dict(model_state)
        local_model.eval()
        
        results = []
        for row in partition:
            name = row.name
            padded = name.ljust(max_len, '_')
            indices = torch.tensor([char_to_idx.get(c, char_to_idx['_']) for c in padded])
            
            with torch.no_grad():
                indices_tensor = indices.unsqueeze(0)
                embedding = local_model(indices_tensor)
                embedding_list = embedding.squeeze().tolist()
            
            results.append((name, embedding_list))
        
        return results
    
    # Apply embedding computation to each partition
    embeddings_rdd = names_df.rdd.mapPartitions(compute_embeddings)
    embeddings_df = spark.createDataFrame(embeddings_rdd, ["name", "embedding"])
    
    return embeddings_df

if __name__ == "__main__":
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("DistributedTripletModel") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()
    
    # Create sample data as Spark DataFrame
    triplets_data = [
        ('John', 'Jon', 'Sarah'),
        ('Maria', 'Marie', 'David'),
        ('Peter', 'Pete', 'Emma'),
        ('Michael', 'Mike', 'Thomas'),
        ('Elizabeth', 'Elisabeth', 'Catherine'),
        ('Robert', 'Bob', 'Jennifer'),
        ('William', 'Bill', 'Jessica'),
        ('James', 'Jim', 'Ashley'),
        ('Christopher', 'Chris', 'Amanda'),
        ('Daniel', 'Dan', 'Melissa')
    ]
    
    triplets_df = spark.createDataFrame(
        triplets_data, 
        ['anchor', 'positive', 'negative']
    )
    
    print("Training distributed triplet model...")
    model, char_to_idx, max_len = train_triplet_model_distributed(
        spark, triplets_df, embedding_dim=64, batch_size=16, epochs=10, num_partitions=2
    )
    
    # Test similarity computation
    name1, name2 = "John", "Jon"
    emb1 = get_name_embedding(model, name1, char_to_idx, max_len)
    emb2 = get_name_embedding(model, name2, char_to_idx, max_len)
    
    similarity = np.dot(emb1[0], emb2[0]) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    print(f"Similarity between {name1} and {name2}: {similarity:.4f}")
    
    # Example of distributed embedding computation
    test_names_data = [('John',), ('Jon',), ('Maria',), ('Marie',), ('Peter',), ('Pete',)]
    test_names_df = spark.createDataFrame(test_names_data, ['name'])
    
    print("\nComputing embeddings for multiple names...")
    embeddings_df = get_name_embeddings_distributed(spark, model, test_names_df, char_to_idx, max_len)
    embeddings_df.show(truncate=False)
    
    spark.stop()