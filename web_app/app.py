import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from flask import Flask, render_template, jsonify, request
import pandas as pd
from scripts.predictor import predict_stock_price
from scripts.evaluate import evaluate_model

app = Flask(__name__)

@app.route('/')
def index():
    try:
        stock_files = [f for f in os.listdir('E:/AI ML/MY PROJECT/Stock Price Prediction/data/processed') if f.endswith('.csv')]
        stocks = [f.replace('.csv', '') for f in stock_files]
        return render_template('index.html', stocks=stocks)
    except Exception as e:
        print(f"Error loading index: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict/<stock>/<int:days>')
def predict(stock, days):
    try:
        print(f"Processing /predict/{stock}/{days}")
        if not os.path.exists(f'E:/AI ML/MY PROJECT/Stock Price Prediction/models/{stock}_cnn_lstm_model.pt'):
            print(f"Model not found: {stock}_cnn_lstm_model.pt")
            return jsonify({'error': 'Model not found for this stock'}), 404
        pred_data = predict_stock_price(stock, days)
        return jsonify(pred_data)
    except Exception as e:
        print(f"Error in /predict/{stock}/{days}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/original/<stock>')
def original(stock):
    try:
        print(f"Processing /original/{stock}")
        data_file = f'E:/AI ML/MY PROJECT/Stock Price Prediction/data/processed/{stock}.csv'
        if not os.path.exists(data_file):
            print(f"Data file not found: {data_file}")
            return jsonify({'error': 'Original data not found for this stock'}), 404
        df = pd.read_csv(data_file)
        days = request.args.get('days', type=int)
        if days is not None and days > 0:
            df = df.tail(days)
        return jsonify({
            'dates': df['Date'].tolist(),
            'close_prices': df['Close'].tolist()
        })
    except Exception as e:
        print(f"Error in /original/{stock}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/evaluate/<stock>')
def evaluate(stock):
    try:
        print(f"Processing /evaluate/{stock}")
        if not os.path.exists(f'E:/AI ML/MY PROJECT/Stock Price Prediction/models/{stock}_cnn_lstm_model.pt'):
            print(f"Model not found: {stock}_cnn_lstm_model.pt")
            return jsonify({'error': 'Model not found for this stock'}), 404
        metrics = evaluate_model(stock)
        return jsonify(metrics)
    except Exception as e:
        print(f"Error in /evaluate/{stock}: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    try:
        print("Starting Flask server on http://127.0.0.1:5000")
        app.run(debug=True)
    except Exception as e:
        print(f"Failed to start Flask server: {str(e)}")


