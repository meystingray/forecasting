import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesDataset(Dataset):
    """Custom dataset for time series data"""
    
    def __init__(self, data: np.ndarray, seq_length: int, pred_length: int = 1):
        self.data = data
        self.seq_length = seq_length
        self.pred_length = pred_length
        
    def __len__(self):
        return len(self.data) - self.seq_length - self.pred_length + 1
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + self.seq_length:idx + self.seq_length + self.pred_length]
        return torch.FloatTensor(x), torch.FloatTensor(y)

class LSTMForecaster(nn.Module):
    """LSTM-based time series forecasting model"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                 output_size: int, dropout: float = 0.2):
        super(LSTMForecaster, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Take the last output
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = self.relu(self.fc1(last_output))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

class GRUForecaster(nn.Module):
    """GRU-based time series forecasting model"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                 output_size: int, dropout: float = 0.2):
        super(GRUForecaster, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # GRU forward pass
        gru_out, _ = self.gru(x, h0)
        
        # Take the last output
        last_output = gru_out[:, -1, :]
        
        # Fully connected layers
        out = self.relu(self.fc1(last_output))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

class TransformerForecaster(nn.Module):
    """Transformer-based time series forecasting model"""
    
    def __init__(self, input_size: int, d_model: int, nhead: int, num_layers: int, 
                 output_size: int, dropout: float = 0.1):
        super(TransformerForecaster, self).__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoding = self._generate_pos_encoding(1000, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.fc = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def _generate_pos_encoding(self, max_len: int, d_model: int):
        pos_encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(np.log(10000.0) / d_model))
        
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        
        return pos_encoding.unsqueeze(0)
    
    def forward(self, x):
        seq_len = x.size(1)
        
        # Project input to d_model dimensions
        x = self.input_projection(x)
        
        # Add positional encoding
        pos_enc = self.pos_encoding[:, :seq_len, :].to(x.device)
        x = x + pos_enc
        
        # Transformer forward pass
        transformer_out = self.transformer(x)
        
        # Take the last output and project to output size
        last_output = transformer_out[:, -1, :]
        out = self.fc(self.dropout(last_output))
        
        return out

class TimeSeriesForecaster:
    """Main class for time series forecasting"""
    
    def __init__(self, model_type: str = 'lstm', seq_length: int = 60, 
                 pred_length: int = 1, hidden_size: int = 64, num_layers: int = 2,
                 dropout: float = 0.2, learning_rate: float = 0.001):
        
        self.model_type = model_type.lower()
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        
        self.scaler = MinMaxScaler()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def prepare_data(self, data: np.ndarray, train_ratio: float = 0.8) -> Tuple:
        """Prepare and split data for training"""
        
        # Normalize data
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        scaled_data = self.scaler.fit_transform(data)
        
        # Split data
        train_size = int(len(scaled_data) * train_ratio)
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size:]
        
        # Create datasets
        train_dataset = TimeSeriesDataset(train_data, self.seq_length, self.pred_length)
        test_dataset = TimeSeriesDataset(test_data, self.seq_length, self.pred_length)
        
        return train_dataset, test_dataset
    
    def create_model(self, input_size: int, output_size: int):
        """Create model based on specified type"""
        
        if self.model_type == 'lstm':
            model = LSTMForecaster(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                output_size=output_size,
                dropout=self.dropout
            )
        elif self.model_type == 'gru':
            model = GRUForecaster(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                output_size=output_size,
                dropout=self.dropout
            )
        elif self.model_type == 'transformer':
            model = TransformerForecaster(
                input_size=input_size,
                d_model=self.hidden_size,
                nhead=8,
                num_layers=self.num_layers,
                output_size=output_size,
                dropout=self.dropout
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return model.to(self.device)
    
    def train(self, train_dataset: Dataset, val_dataset: Optional[Dataset] = None,
              epochs: int = 100, batch_size: int = 32, patience: int = 10):
        """Train the model"""
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None
        
        # Get input/output dimensions from first batch
        sample_x, sample_y = train_dataset[0]
        input_size = sample_x.shape[-1]
        output_size = sample_y.shape[-1]
        
        # Create model
        self.model = self.create_model(input_size, output_size)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training history
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Training {self.model_type.upper()} model on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation phase
            if val_loader:
                self.model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                        outputs = self.model(batch_x)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                
                avg_val_loss = val_loss / len(val_loader)
                val_losses.append(avg_val_loss)
                
                # Learning rate scheduling
                scheduler.step(avg_val_loss)
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(self.model.state_dict(), 'best_model.pth')
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}")
        
        # Load best model
        if val_loader:
            self.model.load_state_dict(torch.load('best_model.pth'))
        
        return train_losses, val_losses
    
    def predict(self, dataset: Dataset, batch_size: int = 32) -> np.ndarray:
        """Make predictions on dataset"""
        
        self.model.eval()
        predictions = []
        actuals = []
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(batch_y.numpy())
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Inverse transform to original scale
        predictions_scaled = self.scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        actuals_scaled = self.scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()
        
        return predictions_scaled, actuals_scaled
    
    def evaluate(self, dataset: Dataset, batch_size: int = 32) -> dict:
        """Evaluate model performance"""
        
        predictions, actuals = self.predict(dataset, batch_size)
        
        mse = mean_squared_error(actuals, predictions)
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mse)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
        
        return {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }
    
    def plot_predictions(self, dataset: Dataset, num_samples: int = 200):
        """Plot predictions vs actual values"""
        
        predictions, actuals = self.predict(dataset)
        
        # Limit samples for visualization
        if len(predictions) > num_samples:
            idx = np.random.choice(len(predictions), num_samples, replace=False)
            predictions = predictions[idx]
            actuals = actuals[idx]
        
        plt.figure(figsize=(12, 6))
        plt.plot(actuals, label='Actual', alpha=0.7)
        plt.plot(predictions, label='Predicted', alpha=0.7)
        plt.legend()
        plt.title(f'{self.model_type.upper()} Model Predictions vs Actual')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.grid(True, alpha=0.3)
        plt.show()

# Example usage and demonstration
def generate_sample_data(n_samples: int = 1000) -> np.ndarray:
    """Generate sample time series data for demonstration"""
    
    # Create a synthetic time series with trend, seasonality, and noise
    t = np.arange(n_samples)
    trend = 0.02 * t
    seasonal = 10 * np.sin(2 * np.pi * t / 50) + 5 * np.cos(2 * np.pi * t / 30)
    noise = np.random.normal(0, 2, n_samples)
    
    data = trend + seasonal + noise + 50  # Add baseline
    return data

if __name__ == "__main__":
    # Generate sample data
    print("Generating sample time series data...")
    data = generate_sample_data(1000)
    
    # Create forecaster
    forecaster = TimeSeriesForecaster(
        model_type='lstm',  # Options: 'lstm', 'gru', 'transformer'
        seq_length=60,
        pred_length=1,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        learning_rate=0.001
    )
    
    # Prepare data
    train_dataset, test_dataset = forecaster.prepare_data(data, train_ratio=0.8)
    
    # Split training data for validation
    train_size = int(len(train_dataset) * 0.8)
    val_dataset = TimeSeriesDataset(
        train_dataset.data[train_size:], 
        forecaster.seq_length, 
        forecaster.pred_length
    )
    train_dataset.data = train_dataset.data[:train_size]
    
    # Train model
    print("\nTraining model...")
    train_losses, val_losses = forecaster.train(
        train_dataset, 
        val_dataset, 
        epochs=50, 
        batch_size=32, 
        patience=10
    )
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    metrics = forecaster.evaluate(test_dataset)
    
    print("Test Set Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot results
    print("\nPlotting predictions...")
    forecaster.plot_predictions(test_dataset)
