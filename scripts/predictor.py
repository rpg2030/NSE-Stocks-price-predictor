import numpy as np
import pandas as pd
import torch
import pickle
from datetime import datetime, timedelta
import logging
import os

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

def calculate_historical_volatility(close_prices, window=None):
    returns = np.diff(close_prices.flatten()) / close_prices[:-1].flatten()
    if window is None or window >= len(returns):
        volatility = np.std(returns)
    else:
        volatility = np.std(returns[-window:])
    return volatility

def predict_stock_price(stock_ticker, future_days, sequence_length=150):
    try:
        import pdb;pdb.set_trace()
        logging.info(f"Starting prediction for {stock_ticker} for {future_days} days")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device for prediction: {device}")
        
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
        
        # Load data
        data_path = f'E:/AI ML/MY PROJECT/Stock Price Prediction/data/processed/{stock_ticker}.csv'
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        df = pd.read_csv(data_path)
        close_prices = df['Close'].values.reshape(-1, 1)
        
        if len(close_prices) < sequence_length:
            raise ValueError(f"Not enough data for {stock_ticker}: {len(close_prices)} days available, need {sequence_length}")
        
        scaled_data = scaler.transform(close_prices)
        volatility = calculate_historical_volatility(close_prices)
        noise_factor = volatility * 0.5
        logging.info(f"Volatility for {stock_ticker}: {volatility:.6f}, Noise factor: {noise_factor:.6f}")
        
        last_sequence = scaled_data[-sequence_length:]
        future_predictions = []
        current_sequence = last_sequence.copy()
        
        last_date = pd.to_datetime(df['Date'].iloc[-1])
        future_dates = [(last_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, future_days + 1)]
        
        for _ in range(future_days):
            current_sequence_tensor = torch.tensor(current_sequence, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                next_pred = model(current_sequence_tensor).cpu().numpy()[0, 0]
            noise = np.random.normal(0, noise_factor)
            next_pred_with_noise = np.clip(next_pred + noise, 0, 1)
            future_predictions.append(next_pred_with_noise)
            current_sequence = np.append(current_sequence[1:], [[next_pred_with_noise]], axis=0)
        
        future_predictions = np.array(future_predictions).reshape(-1, 1)
        future_predictions_transformed = scaler.inverse_transform(future_predictions).flatten()
        
        logging.info(f"Prediction completed for {stock_ticker}: {len(future_predictions)} days predicted")
        
        return {
            'dates': future_dates,
            'predictions': future_predictions_transformed.tolist()
        }
        
    except Exception as e:
        logging.error(f"Error predicting for {stock_ticker}: {str(e)}")
        raise Exception(f"Error predicting for {stock_ticker}: {str(e)}")