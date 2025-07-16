import pandas as pd
import numpy as np
from datetime import datetime

def create_balanced_sample(total_size=2000, fraud_ratio=3/13):
    """
    Create a balanced sample dataset with specified ratio of fraud to non-fraud transactions.
    fraud_ratio of 3/13 means for every 13 transactions, 3 will be fraud (approximately 23%)
    """
    print("Loading training data...")
    
    # Load training data
    train_transaction = pd.read_csv('../data/train_transaction.csv')
    train_identity = pd.read_csv('../data/train_identity.csv')
    
    # First merge transaction and identity data
    print("Merging transaction and identity data...")
    merged_data = train_transaction.merge(train_identity, on='TransactionID', how='left')
    
    # Separate fraud and non-fraud transactions
    fraud_transactions = merged_data[merged_data['isFraud'] == 1]
    non_fraud_transactions = merged_data[merged_data['isFraud'] == 0]
    
    print(f"\nOriginal dataset statistics:")
    print(f"Total transactions: {len(merged_data)}")
    print(f"Fraud transactions: {len(fraud_transactions)}")
    print(f"Non-fraud transactions: {len(non_fraud_transactions)}")
    print(f"Transactions with identity info: {len(train_identity)}")
    
    # Calculate required number of each type
    n_fraud = int(total_size * fraud_ratio)
    n_non_fraud = total_size - n_fraud
    
    print(f"\nCreating sample with:")
    print(f"Total size: {total_size}")
    print(f"Fraud transactions: {n_fraud}")
    print(f"Non-fraud transactions: {n_non_fraud}")
    
    # Sample from each group
    sampled_fraud = fraud_transactions.sample(n=n_fraud, random_state=42)
    sampled_non_fraud = non_fraud_transactions.sample(n=n_non_fraud, random_state=42)
    
    # Combine samples and shuffle
    combined_sample = pd.concat([sampled_fraud, sampled_non_fraud])
    combined_sample = combined_sample.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Generate new sequential TransactionIDs
    combined_sample = combined_sample.reset_index()
    combined_sample['NewTransactionID'] = combined_sample.index + 1
    
    # Separate back into transaction and identity dataframes
    transaction_cols = train_transaction.columns.tolist()
    identity_cols = train_identity.columns.tolist()
    identity_cols.remove('TransactionID')  # Remove TransactionID as we'll add it back
    
    # Create new transaction dataframe
    new_transaction = combined_sample[transaction_cols].copy()
    new_transaction['TransactionID'] = combined_sample['NewTransactionID']
    
    # Create new identity dataframe (only for rows that had identity info)
    new_identity = combined_sample[combined_sample[identity_cols[0]].notna()][identity_cols].copy()
    new_identity['TransactionID'] = combined_sample[combined_sample[identity_cols[0]].notna()]['NewTransactionID']
    
    # Save to CSV
    transaction_output = '../data/test_transaction.csv'
    identity_output = '../data/test_identity.csv'
    
    new_transaction.to_csv(transaction_output, index=False)
    new_identity.to_csv(identity_output, index=False)
    
    print(f"\nGenerated dataset statistics:")
    print(f"Total transactions: {len(new_transaction)}")
    print(f"Fraud transactions: {len(new_transaction[new_transaction['isFraud'] == 1])}")
    print(f"Non-fraud transactions: {len(new_transaction[new_transaction['isFraud'] == 0])}")
    print(f"Transactions with identity info: {len(new_identity)}")
    
    print(f"\nFiles saved:")
    print(f"Transaction data: {transaction_output}")
    print(f"Identity data: {identity_output}")
    
    # Print sample of transactions with their identity info
    print("\nFirst 5 transactions with identity info:")
    sample_with_identity = new_transaction.merge(new_identity, on='TransactionID', how='inner').head()
    print("\nTransaction details:")
    print(sample_with_identity[['TransactionID', 'TransactionAmt', 'isFraud']].to_string())
    
    # Verify ID matching
    common_ids = set(new_transaction['TransactionID']).intersection(set(new_identity['TransactionID']))
    print(f"\nID Verification:")
    print(f"Total unique TransactionIDs in transaction data: {new_transaction['TransactionID'].nunique()}")
    print(f"Total unique TransactionIDs in identity data: {new_identity['TransactionID'].nunique()}")
    print(f"Number of transactions with matching identity data: {len(common_ids)}")
    
    return transaction_output, identity_output

if __name__ == "__main__":
    # Create sample with 2000 entries, ratio of 3:10 (fraud:non-fraud)
    create_balanced_sample(total_size=2000, fraud_ratio=3/13)
