from flask import Flask, render_template, jsonify
import pandas as pd
from datetime import datetime, timedelta
import json
from kafka import KafkaConsumer
from threading import Thread
import queue
import joblib
import shap
from kafka_consumer import FraudDetector
from model_utils import IsolationForestWrapper

app = Flask(__name__)

# Global queue for real-time transactions
transaction_queue = queue.Queue(maxsize=100)

# Load the trained model for SHAP values
model = joblib.load('../models/xgboost_model.joblib')

def kafka_consumer_thread():
    """Background thread to consume Kafka messages"""
    consumer = KafkaConsumer(
        'transactions',
        bootstrap_servers=['localhost:9092'],
        value_deserializer=lambda x: json.loads(x.decode('utf-8')),
        auto_offset_reset='latest'
    )
    
    # Initialize fraud detector
    detector = FraudDetector()
    
    for message in consumer:
        try:
            transaction = message.value
            
            # Make predictions
            result = detector.predict_fraud(transaction)
            
            # Update transaction queue
            if transaction_queue.full():
                transaction_queue.get()  # Remove oldest transaction if queue is full
            transaction_queue.put(transaction)
            
        except Exception as e:
            print(f"Error processing transaction: {str(e)}")
            # Add error transaction to queue
            error_tx = transaction.copy()
            error_tx['ensemble_prediction'] = 0  # Mark as clean for error cases
            error_tx['timestamp'] = datetime.now().isoformat()
            if transaction_queue.full():
                transaction_queue.get()
            transaction_queue.put(error_tx)

@app.route('/')
def index():
    """Render the main dashboard"""
    return render_template('index.html')

@app.route('/api/transactions/live')
def get_live_transactions():
    """Get current transactions in the queue"""
    transactions = []
    while not transaction_queue.empty():
        transactions.append(transaction_queue.get())
    return jsonify(transactions)

@app.route('/api/fraud/stats')
def get_fraud_stats():
    """Get fraud statistics from the last 24 hours"""
    try:
        # Read and parse the fraud log
        try:
            fraud_log = pd.read_csv('../data/fraud_log.csv')
            print(f"Read {len(fraud_log)} entries from fraud log")  # Debug line
        except FileNotFoundError:
            print("Fraud log file not found")  # Debug line
            # Return empty stats if log file doesn't exist yet
            return jsonify({
                'total_transactions': 0,
                'total_fraud': 0,
                'total_amount_fraud': 0.0,
                'recent_frauds': []
            })
        
        # Convert timestamps and ensure UTC
        fraud_log['timestamp'] = pd.to_datetime(fraud_log['timestamp'])
        
        # Filter last 24 hours
        last_24h = pd.Timestamp.now() - pd.Timedelta(days=1)
        recent_transactions = fraud_log[fraud_log['timestamp'] > last_24h].copy()
        print(f"Found {len(recent_transactions)} transactions in last 24h")  # Debug line
        
        if recent_transactions.empty:
            return jsonify({
                'total_transactions': 0,
                'total_fraud': 0,
                'total_amount_fraud': 0.0,
                'recent_frauds': []
            })
        
        # Get fraudulent transactions
        fraudulent = recent_transactions[recent_transactions['ensemble_prediction'] == 1]
        print(f"Found {len(fraudulent)} fraudulent transactions")  # Debug line
        
        # Calculate basic stats
        total_amount = float(fraudulent['amount'].sum()) if not fraudulent.empty else 0.0
        print(f"Total fraud amount: ${total_amount:.2f}")  # Debug line
        
        stats = {
            'total_transactions': len(recent_transactions),
            'total_fraud': len(fraudulent),
            'total_amount_fraud': total_amount,
            'recent_frauds': []
        }
        
        # Get recent frauds with enhanced details
        if not fraudulent.empty:
            recent_frauds = fraudulent.sort_values('timestamp', ascending=False)\
                          .head(5)\
                          .assign(
                              timestamp=lambda x: x['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S'),
                              amount=lambda x: x['amount'].round(2)
                          )
            stats['recent_frauds'] = recent_frauds.to_dict('records')
            
        print(f"Returning stats: {stats}")  # Debug line
        return jsonify(stats)
        
    except Exception as e:
        print(f"Error getting fraud stats: {str(e)}")  # Debug line
        return jsonify({'error': str(e)}), 500

@app.route('/api/model/shap/<int:transaction_id>')
def get_shap_values(transaction_id):
    """Get SHAP values for a specific transaction"""
    try:
        # Load the transaction data
        fraud_log = pd.read_csv('../data/fraud_log.csv')
        transaction = fraud_log[fraud_log['transaction_id'] == transaction_id].iloc[0]
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(transaction.drop(['timestamp', 'transaction_id', 'amount']))
        
        # Format SHAP values for visualization
        feature_importance = dict(zip(
            transaction.drop(['timestamp', 'transaction_id', 'amount']).index,
            shap_values[0]
        ))
        
        return jsonify(feature_importance)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Start Kafka consumer thread
    consumer_thread = Thread(target=kafka_consumer_thread, daemon=True)
    consumer_thread.start()
    
    # Run Flask app
    app.run(debug=True, port=5000)
