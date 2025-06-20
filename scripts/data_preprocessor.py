import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import pickle
import logging
from multiprocessing import Pool, cpu_count
try:
    import cupy as cp
    USE_CUPY = True
except ImportError:
    USE_CUPY = False

logging.basicConfig(
    filename='logs/training_logs.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def preprocess_stock(stock_file):
    try:
        stock_ticker = stock_file.replace('.csv', '')
        logging.info(f"Preprocessing data for {stock_ticker}")
        
        df = pd.read_csv(f'E:/AI ML/MY PROJECT/Stock Price Prediction/data/processed/{stock_file}')
        df['Close'] = df['Close'].ffill().bfill()
        data = df[['Close']].values
        
        print(f"Price range for {stock_ticker}: Min={data.min():.2f}, Max={data.max():.2f}")
        
        sequence_length = 150
        if len(data) < sequence_length:
            logging.warning(f"Not enough data for {stock_ticker}. Need at least {sequence_length} days, but only {len(data)} available.")
            return stock_ticker, False
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        print(f"Scaled data range for {stock_ticker}: Min={scaled_data.min():.2f}, Max={scaled_data.max():.2f}")
        
        os.makedirs('E:/AI ML/MY PROJECT/Stock Price Prediction/data/preprocessed/pickel', exist_ok=True)
        with open(f'E:/AI ML/MY PROJECT/Stock Price Prediction/data/preprocessed/pickel/{stock_ticker}_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        
        if USE_CUPY:
            scaled_data = cp.asarray(scaled_data)
            X, y = [], []
            for i in range(sequence_length, len(scaled_data)):
                X.append(scaled_data[i-sequence_length:i])
                y.append(scaled_data[i])
            X, y = cp.stack(X), cp.asarray(y)
            X, y = cp.asnumpy(X), cp.asnumpy(y)
        else:
            X, y = [], []
            for i in range(sequence_length, len(scaled_data)):
                X.append(scaled_data[i-sequence_length:i])
                y.append(scaled_data[i])
            X, y = np.array(X), np.array(y)
        
        train_size = int(0.8 * len(X))
        if train_size == 0:
            logging.warning(f"Not enough sequences for {stock_ticker} to split into train/test.")
            return stock_ticker, False
        
        X_train, y_train = X[:train_size], y[:train_size]
        X_test, y_test = X[train_size:], y[train_size:]
        
        os.makedirs('E:/AI ML/MY PROJECT/Stock Price Prediction/data/preprocessed/npy_file', exist_ok=True)
        np.save(f'E:/AI ML/MY PROJECT/Stock Price Prediction/data/preprocessed/npy_file/{stock_ticker}_X_train.npy', X_train)
        np.save(f'E:/AI ML/MY PROJECT/Stock Price Prediction/data/preprocessed/npy_file/{stock_ticker}_y_train.npy', y_train)
        np.save(f'E:/AI ML/MY PROJECT/Stock Price Prediction/data/preprocessed/npy_file/{stock_ticker}_X_test.npy', X_test)
        np.save(f'E:/AI ML/MY PROJECT/Stock Price Prediction/data/preprocessed/npy_file/{stock_ticker}_y_test.npy', y_test)
        
        logging.info(f"Preprocessed data saved for {stock_ticker}")
        return stock_ticker, True
        
    except Exception as e:
        logging.error(f"Error preprocessing {stock_ticker}: {str(e)}")
        return stock_ticker, False

def preprocess_data(sequence_length=60):
    try:
        stock_files = [f for f in os.listdir('E:/AI ML/MY PROJECT/Stock Price Prediction/data/processed') if f.endswith('.csv')]
        logging.info(f"Preprocessing {len(stock_files)} stocks")
        
        import pdb;pdb.set_trace()
        
        with Pool(processes=cpu_count()) as pool:
            results = pool.map(preprocess_stock, stock_files)
        
        successes = sum(1 for _, status in results if status)
        logging.info(f"Successfully preprocessed {successes}/{len(stock_files)} stocks")
        
    except Exception as e:
        logging.error(f"Error in preprocessing: {str(e)}")
        raise

if __name__ == "__main__":
    preprocess_data()