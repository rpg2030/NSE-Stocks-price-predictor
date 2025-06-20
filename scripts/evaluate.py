import numpy as np
import pandas as pd
import torch
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import logging

logging.basicConfig(
    filename='logs/training_logs.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class CNNLSTMModel(torch.nn.Module):
    def __init__(self, sequence_length=150):
        super(CNNLSTMModel, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.pool1 = torch.nn.MaxPool1d(kernel_size=2)
        self.conv2 = torch.nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm1d(32)
        self.pool2 = torch.nn.MaxPool1d(kernel_size=2)
        self.lstm1 = torch.nn.LSTM(input_size=32, hidden_size=50, batch_first=True)
        self.dropout1 = torch.nn.Dropout(0.2)
        self.lstm2 = torch.nn.LSTM(input_size=50, hidden_size=50, batch_first=True)
        self.dropout2 = torch.nn.Dropout(0.2)
        self.dense1 = torch.nn.Linear(50, 25)
        self.dense2 = torch.nn.Linear(25, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x = x[:, -1, :]
        x = self.relu(self.dense1(x))
        x = self.dense2(x)
        return x

def evaluate_model(stock_ticker):
    try:
        logging.info(f"Evaluating model for {stock_ticker}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device for evaluation: {device}")
        
        # Load scaler
        scaler_path = f'E:/AI ML/MY PROJECT/Stock Price Prediction/data/preprocessed/pickel/{stock_ticker}_scaler.pkl'
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Load model
        model_path = f'E:/AI ML/MY PROJECT/Stock Price Prediction/models/{stock_ticker}_cnn_lstm_model.pt'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model = CNNLSTMModel().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        # Load test data
        X_test_path = f'E:/AI ML/MY PROJECT/Stock Price Prediction/data/preprocessed/npy_file/{stock_ticker}_X_test.npy'
        y_test_path = f'E:/AI ML/MY PROJECT/Stock Price Prediction/data/preprocessed/npy_file/{stock_ticker}_y_test.npy'
        if not os.path.exists(X_test_path) or not os.path.exists(y_test_path):
            raise FileNotFoundError(f"Test data not found for {stock_ticker}")
        
        X_test = np.load(X_test_path)
        y_test = np.load(y_test_path)
        
        # Predict
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        with torch.no_grad():
            y_pred = model(X_test_tensor).cpu().numpy()
        
        # Unscale
        y_pred_unscaled = scaler.inverse_transform(y_pred).flatten()
        y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        # Metrics
        mae = mean_absolute_error(y_test_unscaled, y_pred_unscaled)
        rmse = np.sqrt(mean_squared_error(y_test_unscaled, y_pred_unscaled))
        mask = y_test_unscaled != 0
        mape = np.mean(np.abs((y_test_unscaled[mask] - y_pred_unscaled[mask]) / y_test_unscaled[mask])) * 100
        
        logging.info(f"Evaluation for {stock_ticker}: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%")
        
        return {
            'stock': stock_ticker,
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape)
        }
        
    except Exception as e:
        logging.error(f"Error evaluating for {stock_ticker}: {str(e)}")
        raise Exception(f"Error evaluating for {stock_ticker}: {str(e)}")