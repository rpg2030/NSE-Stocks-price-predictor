import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.amp import autocast
import logging
from torchsummary import summary


logging.basicConfig(
    filename='logs/training_logs.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class CNNLSTMModel(nn.Module):
    def __init__(self, sequence_length=150):
        super(CNNLSTMModel, self).__init__()
        # CNN layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        # LSTM layers
        self.lstm1 = nn.LSTM(input_size=32, hidden_size=50, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(input_size=50, hidden_size=50, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        # Dense layers
        self.dense1 = nn.Linear(50, 25)
        self.dense2 = nn.Linear(25, 1)
        # Activation
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, seq_len, 1) -> (batch, 1, seq_len)
        x = x.permute(0, 2, 1)
        # CNN
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        # Reshape for LSTM: (batch, channels, seq_len) -> (batch, seq_len, channels)
        x = x.permute(0, 2, 1)
        # LSTM
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        # Take last timestep
        x = x[:, -1, :]
        # Dense
        x = self.relu(self.dense1(x))
        x = self.dense2(x)
        return x

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        self.best_state = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def train_cnn_lstm_model():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")
        logging.info(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
        
        preprocessed_dir = 'E:/AI ML/MY PROJECT/Stock Price Prediction/data/preprocessed/npy_file'
        stock_files = [f for f in os.listdir(preprocessed_dir) if f.endswith('_X_train.npy')]
        
        for stock_file in stock_files:
            try:
                stock_ticker = stock_file.replace('_X_train.npy', '')
                logging.info(f"Training CNN-LSTM model for {stock_ticker}")
                # Load data
                X_train = np.load(f'{preprocessed_dir}/{stock_ticker}_X_train.npy')
                y_train = np.load(f'{preprocessed_dir}/{stock_ticker}_y_train.npy')
                X_test = np.load(f'{preprocessed_dir}/{stock_ticker}_X_test.npy')
                y_test = np.load(f'{preprocessed_dir}/{stock_ticker}_y_test.npy')
                
                # Verify shape
                sequence_length = X_train.shape[1]
                if sequence_length != 150:
                    logging.warning(f"Unexpected sequence length for {stock_ticker}: {sequence_length}. Expected 60.")
                
                # Convert to tensors
                X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
                y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
                X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
                y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
                
                # DataLoader
                train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
                test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
                test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
                
                # Model
                model = CNNLSTMModel().to(device)
                
                # Debug: Check parameters
                param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
                logging.info(f"Parameters for {stock_ticker}: {param_count}")
                if param_count == 0:
                    raise ValueError("Model has no trainable parameters")
                
                # Optimizer and loss
                optimizer = optim.Adam(model.parameters(), lr=0.0005)
                criterion = nn.MSELoss()
                
                # Early stopping
                early_stopping = EarlyStopping(patience=5)
                
                # Mixed precision
                scaler = GradScaler()
                
                # Training loop
                for epoch in range(50):
                    model.train()
                    train_loss = 0
                    for batch_x, batch_y in train_loader:
                        optimizer.zero_grad()
                        with autocast('cuda'):
                            output = model(batch_x)
                            loss = criterion(output, batch_y)
                            # L2 regularization
                            l2_lambda = 0.001
                            l2_norm = sum(p.pow(2).sum() for p in model.parameters() if p.requires_grad)
                            loss = loss + l2_lambda * l2_norm
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        train_loss += loss.item() * batch_x.size(0)
                    train_loss /= len(train_loader.dataset)
                    
                    # Validation
                    model.eval()
                    val_loss = 0
                    with torch.no_grad():
                        for batch_x, batch_y in test_loader:
                            with autocast('cuda'):
                                output = model(batch_x)
                                loss = criterion(output, batch_y)
                            val_loss += loss.item() * batch_x.size(0)
                        val_loss /= len(test_loader.dataset)
                    
                    logging.info(f"Epoch {epoch+1}/{50}, {stock_ticker}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")
                    
                    early_stopping(val_loss, model)
                    if early_stopping.early_stop:
                        logging.info(f"Early stopping triggered for {stock_ticker} at epoch {epoch+1}")
                        break
                
                # Restore best weights
                if early_stopping.best_state:
                    model.load_state_dict(early_stopping.best_state)
               
                # Save model
                os.makedirs('E:/AI ML/MY PROJECT/Stock Price Prediction/models', exist_ok=True)
                torch.save(model.state_dict(), f'E:/AI ML/MY PROJECT/Stock Price Prediction/models/{stock_ticker}_cnn_lstm_model.pt')
                logging.info(f"CNN-LSTM model saved for {stock_ticker}")
                
            except Exception as e:
                logging.error(f"Error training model for {stock_ticker}: {str(e)}")
                continue
                
    except Exception as e:
        logging.error(f"Error in training: {str(e)}")
        raise

if __name__ == "__main__":
    train_cnn_lstm_model()