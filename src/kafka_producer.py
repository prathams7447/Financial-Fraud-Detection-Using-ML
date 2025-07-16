import pandas as pd
import json
import time
from kafka import KafkaProducer
from datetime import datetime

class TransactionProducer:
    def __init__(self, bootstrap_servers=['localhost:9092']):
        """Initialize Kafka producer"""
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )
        
    def load_transactions(self, file_path):
        """Load preprocessed test transaction data"""
        print("Loading test transaction data...")
        self.data = pd.read_csv(file_path)
        
        # Convert all column names to string type
        self.data.columns = self.data.columns.astype(str)
        
        # Ensure we have all required columns
        required_cols = ['TransactionID', 'TransactionDT', 'TransactionAmt']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Print feature information
        print(f"\nLoaded {len(self.data):,} test transactions")
        print(f"Number of features: {len(self.data.columns)}")
        print("\nFeatures:")
        print(sorted(self.data.columns.tolist()))
        
    def validate_transaction(self, transaction):
        """Validate transaction data before sending"""
        # Check for negative or zero amounts
        if transaction['TransactionAmt'] <= 0:
            print(f"Warning: Invalid transaction amount ${transaction['TransactionAmt']} for ID {transaction['TransactionID']}")
            return False
        return True

    def simulate_transactions(self, delay_seconds=1):
        """Simulate transactions by sending them to Kafka"""
        print("\nStarting transaction simulation...")
        print("Press Ctrl+C to stop the simulation")
        
        valid_count = 0
        invalid_count = 0
        
        try:
            for _, transaction in self.data.iterrows():
                try:
                    # Convert transaction to dictionary
                    transaction_dict = transaction.to_dict()
                    
                    # Validate transaction
                    if self.validate_transaction(transaction_dict):
                        # Send transaction to Kafka
                        self.producer.send(
                            'transactions',
                            value=transaction_dict
                        )
                        valid_count += 1
                        
                        # Print transaction info
                        print(f"Sent transaction {transaction['TransactionID']} - Amount: ${transaction['TransactionAmt']:.2f}")
                        
                    else:
                        invalid_count += 1
                        
                    time.sleep(delay_seconds)
                    
                except Exception as e:
                    print(f"Error sending transaction: {str(e)}")
                    invalid_count += 1
                    
        except KeyboardInterrupt:
            print("\nSimulation stopped by user")
            
        print(f"\nSimulation complete:")
        print(f"Valid transactions: {valid_count}")
        print(f"Invalid transactions: {invalid_count}")
            
        # Flush and close producer
        self.producer.flush()
        self.producer.close()

if __name__ == "__main__":
    # Initialize producer
    producer = TransactionProducer()
    
    # Load and simulate transactions from preprocessed test data
    producer.load_transactions('../models/preprocessed_test.csv')
    producer.simulate_transactions()
    print("Test transaction simulation completed!")
