import yfinance as yf
import pandas as pd
import os
import logging
from datetime import datetime, timedelta
import sys
from multiprocessing import Pool, cpu_count

logging.basicConfig(
    filename='logs/training_logs.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def fetch_stock(stock):
    try:
        ticker = f"{stock}.NS"
        logging.info(f"Fetching data for {ticker}")
        end_date = datetime.today().strftime('%Y-%m-%d')
        start_date = (datetime.today() - timedelta(days=6*365)).strftime('%Y-%m-%d')
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            logging.warning(f"No data fetched for {ticker}")
            return ticker, False
        
        data = data.reset_index()[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        data['Date'] = pd.to_datetime(data['Date']).dt.strftime('%Y-%m-%d')
        
        output_path = os.path.join('E:/AI ML/MY PROJECT/Stock Price Prediction/data/processed', f"{ticker}.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        data.to_csv(output_path, index=False, header=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        
        if os.path.exists(output_path):
            logging.info(f"Saved data for {ticker} to {output_path}")
            return ticker, True
        else:
            logging.error(f"Failed to save data for {ticker} to {output_path}")
            return ticker, False
            
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {str(e)}")
        return ticker, False

def fetch_stock_data():
    try:
        equity_df = pd.read_csv('E:/AI ML/MY PROJECT/Stock Price Prediction/data/raw/EQUITY_L.csv')
        if 'SYMBOL' not in equity_df.columns:
            logging.error("SYMBOL column not found in EQUITY_L.csv")
            raise KeyError("SYMBOL column not found in EQUITY_L.csv")
        
        stocks = equity_df['SYMBOL'].tolist()
        logging.info(f"Selected {len(stocks)} stocks")
        
        with Pool(processes=cpu_count()) as pool:
            results = pool.map(fetch_stock, stocks)
        
        successes = sum(1 for _, status in results if status)
        logging.info(f"Successfully fetched data for {successes}/{len(stocks)} stocks")
        
    except Exception as e:
        logging.error(f"Error in fetch_stock_data: {str(e)}")
        raise

if __name__== "__main__":
    try:
        fetch_stock_data()
        logging.info("Data fetching completed successfully")
    except Exception as e:
        logging.error(f"Script failed: {str(e)}")
        sys.exit(1)