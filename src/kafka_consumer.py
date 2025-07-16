import os
import json
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from kafka import KafkaConsumer
import joblib
from utils import SMSNotifier
from blockchain_logger import BlockchainLogger

class FraudDetector:
    def __init__(self, bootstrap_servers=['localhost:9092']):
        """Initialize Kafka consumer and load models"""
        self.consumer = KafkaConsumer(
            'transactions',
            bootstrap_servers=bootstrap_servers,
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            auto_offset_reset='latest'
        )
        
        # Load ensemble model
        print("Loading ensemble model...")
        model_path = '../models/ensemble_model.joblib'
        if not os.path.exists(model_path):
            raise ValueError("Ensemble model not found! Please train models first.")
        self.model = joblib.load(model_path)
        print("Model loaded successfully!")
        
        # Get feature names from model
        self.feature_names = self.model.feature_names_in_
        print(f"Model expects {len(self.feature_names)} features")
        
        # Initialize notifiers
        self.sms_notifier = SMSNotifier()
        
        # Try to initialize blockchain logger
        try:
            self.blockchain_logger = BlockchainLogger()
            self.blockchain_enabled = True
            print("Blockchain logging enabled")
        except Exception as e:
            print(f"Blockchain logging disabled: {str(e)}")
            self.blockchain_enabled = False
            
        # Ensure fraud log directory exists
        os.makedirs('../data', exist_ok=True)
        self.fraud_log_path = '../data/fraud_log.csv'
        
        # Create fraud log if it doesn't exist
        if not os.path.exists(self.fraud_log_path):
            with open(self.fraud_log_path, 'w') as f:
                f.write('timestamp,transaction_id,amount,ensemble_prediction\n')
    
    def log_to_csv(self, transaction, prediction):
        """Log transaction to CSV file"""
        try:
            # Create DataFrame for consistent formatting
            log_df = pd.DataFrame([{
                'timestamp': transaction['timestamp'],
                'transaction_id': transaction['TransactionID'],
                'amount': transaction['TransactionAmt'],
                'ensemble_prediction': 1 if prediction['is_fraud'] else 0
            }])
            
            # Write to CSV
            log_df.to_csv(self.fraud_log_path, mode='a', header=False, index=False)
            print(f"Logged transaction {transaction['TransactionID']} to CSV")
                
        except Exception as e:
            print(f"Failed to log to CSV: {str(e)}")
    
    def predict_fraud(self, transaction_data):
        """Predict if a transaction is fraudulent using the ensemble model"""
        try:
            # Convert transaction data to DataFrame
            df = pd.DataFrame([transaction_data])
            
            # Create feature array in correct order
            feature_array = np.zeros(len(self.feature_names))
            
            # Fill in available features
            for i, feature in enumerate(self.feature_names):
                if feature in df.columns:
                    feature_array[i] = df[feature].iloc[0]
            
            # Create DataFrame with correct feature names
            X = pd.DataFrame([feature_array], columns=self.feature_names)
            
            # Make prediction using ensemble model
            prediction = self.model.predict(X)[0]
            
            # Add prediction to transaction data
            transaction_data['ensemble_prediction'] = int(prediction)  # 1 for fraud, 0 for clean
            transaction_data['timestamp'] = datetime.now().isoformat()
            
            # Return prediction result
            result = {
                'transaction_id': transaction_data.get('TransactionID', 'unknown'),
                'is_fraud': bool(prediction),
                'timestamp': transaction_data['timestamp']
            }
            
            return result
            
        except Exception as e:
            logging.error(f"Error predicting fraud: {str(e)}")
            raise
    
    def process_transactions(self):
        """Process incoming transactions in real-time"""
        print("Starting fraud detection service...")
        
        try:
            for message in self.consumer:
                transaction = message.value
                
                # Make prediction
                result = self.predict_fraud(transaction)
                predicted_fraud = "FRAUD" if result['is_fraud'] else "CLEAN"
                
                # Print prediction
                print(f"\nTransaction {transaction['TransactionID']}:")
                print(f"Amount: ${transaction['TransactionAmt']:.2f}")
                print(f"Predicted: {predicted_fraud}")
                
                # Log all transactions to CSV
                self.log_to_csv(transaction, result)
                
                # If fraud detected, send alerts
                if result['is_fraud']:
                    print("Fraud detected!")
                    # Send SMS alert
                    # self.sms_notifier.send_fraud_alert(
                    #     transaction_id=transaction['TransactionID'],
                    #     amount=transaction['TransactionAmt'],
                    #     confidence=0.95  # Default confidence for now
                    # )
                    
                    # # Log to blockchain if enabled
                    # if self.blockchain_enabled:
                    #     try:
                    #         self.blockchain_logger.log_fraud(
                    #             transaction_id=str(transaction['TransactionID']),
                    #             timestamp=result['timestamp'],
                    #             fraud_score=0.95  # Using same confidence as SMS
                    #         )
                    #     except Exception as e:
                    #         print(f"Failed to log to blockchain: {str(e)}")
                
        except KeyboardInterrupt:
            print("\nStopping fraud detection service...")
            self.consumer.close()
        except Exception as e:
            print(f"Error processing transactions: {str(e)}")
            self.consumer.close()
            raise

if __name__ == "__main__":
    # Initialize and run fraud detector
    detector = FraudDetector()
    detector.process_transactions()
