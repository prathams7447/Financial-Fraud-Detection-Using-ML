import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
import os

class DataPreprocessor:
    def __init__(self, train_transaction_path, train_identity_path, 
                 test_transaction_path=None, test_identity_path=None):
        self.train_transaction_path = train_transaction_path
        self.train_identity_path = train_identity_path
        self.test_transaction_path = test_transaction_path
        self.test_identity_path = test_identity_path
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self):
        """Load and merge transaction and identity datasets for both train and test"""
        print("Loading training data...")
        try:
            # Load transaction data
            train_transaction = pd.read_csv(self.train_transaction_path, encoding='utf-8')
            print("\nShape of transaction data:", train_transaction.shape)
            
            # Basic data validation
            if 'TransactionAmt' in train_transaction.columns:
                print("\nTransaction amount statistics:")
                print(train_transaction['TransactionAmt'].describe())
                
                # Handle negative amounts - flag them but keep them
                neg_amounts = train_transaction['TransactionAmt'] < 0
                if neg_amounts.any():
                    print(f"\nFound {np.sum(neg_amounts):,} negative transaction amounts")
                    print("These will be kept as potential fraud indicators")
            
            print("\nTarget value distribution in transaction data:")
            print(train_transaction['isFraud'].value_counts(dropna=False))
            
            # Load identity data if available
            try:
                train_identity = pd.read_csv(self.train_identity_path, encoding='utf-8')
                print("\nShape of identity data:", train_identity.shape)
                
                # Merge training datasets
                self.train_data = train_transaction.merge(train_identity, on='TransactionID', how='left')
                print("\nShape after merge:", self.train_data.shape)
            except FileNotFoundError:
                print("\nNo identity data found, using transaction data only")
                self.train_data = train_transaction
            
            # Basic data cleaning
            # Keep TransactionID and actual transaction amount for reference
            self.reference_cols = {
                'TransactionID': self.train_data['TransactionID'].copy(),
                'TransactionAmt': self.train_data['TransactionAmt'].copy() if 'TransactionAmt' in self.train_data.columns else None
            }
            
            print("\nTarget value distribution in final dataset:")
            print(self.train_data['isFraud'].value_counts(dropna=False))
            
            if self.test_transaction_path and self.test_identity_path:
                print("\nLoading test data...")
                test_transaction = pd.read_csv(self.test_transaction_path, encoding='utf-8')
                try:
                    test_identity = pd.read_csv(self.test_identity_path, encoding='utf-8')
                    self.test_data = test_transaction.merge(test_identity, on='TransactionID', how='left')
                except FileNotFoundError:
                    print("\nNo test identity data found, using test transaction data only")
                    self.test_data = test_transaction
                
                # Keep reference columns for test data too
                self.test_reference_cols = {
                    'TransactionID': self.test_data['TransactionID'].copy(),
                    'TransactionAmt': self.test_data['TransactionAmt'].copy() if 'TransactionAmt' in self.test_data.columns else None
                }
        except Exception as e:
            print(f"\nError loading data: {str(e)}")
            raise
        
        return self.train_data
    
    def handle_missing_values(self):
        """Handle missing values in both training and test datasets"""
        print("Handling missing values...")
        
        # Identify numeric and categorical columns
        numeric_cols = self.train_data.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = self.train_data.select_dtypes(include=['object']).columns
        
        # Remove target column from numeric columns if present
        numeric_cols = [col for col in numeric_cols if col != 'isFraud']
        
        # Handle missing values in numeric columns
        if len(numeric_cols) > 0:
            self.num_imputer = SimpleImputer(strategy='median')  # Use median for robustness
            self.train_data[numeric_cols] = self.num_imputer.fit_transform(self.train_data[numeric_cols])
            if hasattr(self, 'test_data'):
                self.test_data[numeric_cols] = self.num_imputer.transform(self.test_data[numeric_cols])
        
        # Handle missing values in categorical columns
        if len(categorical_cols) > 0:
            self.cat_imputer = SimpleImputer(strategy='constant', fill_value='unknown')
            self.train_data[categorical_cols] = self.cat_imputer.fit_transform(self.train_data[categorical_cols])
            if hasattr(self, 'test_data'):
                self.test_data[categorical_cols] = self.cat_imputer.transform(self.test_data[categorical_cols])
    
    def encode_categorical(self):
        """Encode categorical variables"""
        print("Encoding categorical variables...")
        
        # Get categorical columns
        categorical_cols = self.train_data.select_dtypes(include=['object']).columns
        
        # Initialize label encoders dictionary
        self.label_encoders = {}
        
        # Encode each categorical column
        for col in categorical_cols:
            print(f"Encoding column: {col}")
            le = LabelEncoder()
            
            # Get all unique values including test data if available
            unique_values = set(self.train_data[col].unique())
            if hasattr(self, 'test_data'):
                unique_values.update(self.test_data[col].unique())
            
            # Ensure 'unknown' is in the encoder
            unique_values.add('unknown')
            
            # Fit encoder on all possible values
            le.fit(list(unique_values))
            
            # Transform the data
            self.train_data[col] = le.transform(self.train_data[col])
            if hasattr(self, 'test_data'):
                self.test_data[col] = le.transform(self.test_data[col])
            
            # Store the encoder
            self.label_encoders[col] = le
    
    def scale_features(self):
        """Scale numeric features"""
        print("Scaling features...")
        
        # Get numeric columns excluding target and ID
        numeric_cols = self.train_data.select_dtypes(include=['int64', 'float64']).columns
        numeric_cols = [col for col in numeric_cols if col not in ['isFraud', 'TransactionID']]
        
        if len(numeric_cols) > 0:
            # Initialize scaler
            self.scaler = StandardScaler()
            
            # Fit and transform training data
            self.train_data[numeric_cols] = self.scaler.fit_transform(self.train_data[numeric_cols])
            
            # Transform test data if available
            if hasattr(self, 'test_data'):
                self.test_data[numeric_cols] = self.scaler.transform(self.test_data[numeric_cols])
    
    def handle_imbalance(self):
        """Handle class imbalance in training data using random undersampling"""
        print("Handling class imbalance...")
        
        # Separate features and target
        X = self.train_data.drop(['isFraud', 'TransactionID'], axis=1, errors='ignore')
        y = self.train_data['isFraud']
        
        # Print original class distribution
        print("\nOriginal class distribution:")
        orig_dist = y.value_counts()
        print(orig_dist)
        
        # Calculate sampling size
        n_fraud = orig_dist[1]  # Number of fraud cases
        n_non_fraud = min(orig_dist[0], n_fraud * 3)  # Keep 3x non-fraud cases
        
        print(f"\nSampling strategy:")
        print(f"- Fraud cases (class 1): {n_fraud}")
        print(f"- Non-fraud cases (class 0): {n_non_fraud}")
        
        # Get indices for each class
        fraud_idx = y[y == 1].index
        non_fraud_idx = y[y == 0].index
        
        # Randomly sample from non-fraud cases
        np.random.seed(42)
        sampled_non_fraud_idx = np.random.choice(non_fraud_idx, size=n_non_fraud, replace=False)
        
        # Combine indices
        balanced_idx = np.concatenate([fraud_idx, sampled_non_fraud_idx])
        
        # Create balanced dataset efficiently using indexing
        balanced_data = {
            'features': X.loc[balanced_idx],
            'isFraud': y[balanced_idx],
            'TransactionID': self.reference_cols['TransactionID'][balanced_idx]
        }
        
        if self.reference_cols['TransactionAmt'] is not None:
            balanced_data['OriginalTransactionAmt'] = self.reference_cols['TransactionAmt'][balanced_idx]
        
        # Combine all parts using concat
        self.train_data_balanced = pd.concat(balanced_data.values(), axis=1)
        
        print("\nFinal class distribution:")
        print(self.train_data_balanced['isFraud'].value_counts(normalize=True))
        print(f"Final dataset size: {len(self.train_data_balanced)}")
    
    def save_preprocessed_data(self, train_output_path, test_output_path=None):
        """Save preprocessed data and transformers"""
        print(f"Saving preprocessed training data to {train_output_path}")
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(train_output_path), exist_ok=True)
        
        # Define core columns that must be present and their order
        core_columns = [
            ('TransactionID', 'int'),
            ('TransactionAmt', 'float'),
            ('isFraud', 'int')
        ]
        
        # Prepare data for saving
        print("\nPreparing full balanced dataset...")
        data_to_save = self.train_data_balanced.copy()
        print(f"Total rows to save: {len(data_to_save):,}")
        
        # Convert all transaction amounts to absolute values
        if 'TransactionAmt' in data_to_save.columns:
            data_to_save['TransactionAmt'] = data_to_save['TransactionAmt'].abs()
        
        # Free up memory
        if hasattr(self, 'train_data_balanced'):
            del self.train_data_balanced
        
        print("\nInitial column types:")
        for col, dtype in data_to_save.dtypes.items():
            print(f"- {col}: {dtype}")
        
        # Convert core columns first
        print("\nValidating core columns...")
        for col, dtype in core_columns:
            try:
                if dtype == 'int':
                    data_to_save[col] = data_to_save[col].fillna(0).astype(np.int64)
                elif dtype == 'float':
                    data_to_save[col] = data_to_save[col].fillna(0).astype(float).round(4)
                print(f"Validated {col} as {dtype}")
            except Exception as e:
                print(f"Error converting {col} to {dtype}: {str(e)}")
                raise
        
        # Get remaining columns
        remaining_cols = [col for col in data_to_save.columns if col not in [c[0] for c in core_columns]]
        
        # Convert remaining columns
        print("\nConverting remaining columns...")
        for col in remaining_cols:
            curr_dtype = data_to_save[col].dtype
            if pd.api.types.is_numeric_dtype(curr_dtype):
                try:
                    data_to_save[col] = data_to_save[col].fillna(0).astype(float).round(4)
                    print(f"Converted {col} from {curr_dtype} to float")
                except Exception as e:
                    print(f"Warning: Could not convert {col} to float: {str(e)}")
            else:
                try:
                    data_to_save[col] = data_to_save[col].fillna('unknown').astype(str)
                    print(f"Converted {col} from {curr_dtype} to string")
                except Exception as e:
                    print(f"Warning: Could not convert {col} to string: {str(e)}")
        
        # Reorder columns to ensure core columns come first
        final_columns = [col for col, _ in core_columns] + remaining_cols
        data_to_save = data_to_save[final_columns]
        
        print("\nFinal column order:")
        for i, (col, dtype) in enumerate(data_to_save.dtypes.items()):
            print(f"{i+1}. {col}: {dtype}")
        
        print("\nSaving data...")
        try:
            # Save data in chunks to reduce memory usage
            chunk_size = 50000  # Increased chunk size for faster processing
            total_chunks = (len(data_to_save) + chunk_size - 1) // chunk_size
            
            for i in range(0, len(data_to_save), chunk_size):
                chunk_num = i // chunk_size + 1
                mode = 'w' if i == 0 else 'a'
                header = i == 0
                
                chunk = data_to_save.iloc[i:i + chunk_size]
                chunk.to_csv(
                    train_output_path,
                    mode=mode,
                    header=header,
                    index=False,
                    encoding='utf-8',
                    sep=',',
                    float_format='%.4f'
                )
                print(f"Saved chunk {chunk_num}/{total_chunks} ({i:,} to {i + len(chunk):,} rows)")
            
            print("\nVerifying saved data...")
            # Read back first few rows to verify format
            verification_data = pd.read_csv(train_output_path, nrows=5, encoding='utf-8')
            print("\nFirst few rows of saved data:")
            print(verification_data.head().to_string())
            print("\nVerification column order:")
            for i, (col, dtype) in enumerate(verification_data.dtypes.items()):
                print(f"{i+1}. {col}: {dtype}")
            
        except Exception as e:
            print(f"Error saving data: {str(e)}")
            raise
        
        print("\nSaving model files...")
        model_dir = os.path.dirname(train_output_path)
        
        # Save preprocessors
        joblib.dump(self.scaler, os.path.join(model_dir, 'scaler.joblib'))
        joblib.dump(self.label_encoders, os.path.join(model_dir, 'label_encoders.joblib'))
        joblib.dump(self.num_imputer, os.path.join(model_dir, 'numeric_imputer.joblib'))
        joblib.dump(self.cat_imputer, os.path.join(model_dir, 'categorical_imputer.joblib'))
        
        # Save column information
        print("\nSaving column information...")
        column_info = {
            'core_columns': [col for col, _ in core_columns],
            'numeric_columns': [col for col, dtype in data_to_save.dtypes.items() 
                              if pd.api.types.is_numeric_dtype(dtype)],
            'categorical_columns': [col for col, dtype in data_to_save.dtypes.items() 
                                  if not pd.api.types.is_numeric_dtype(dtype)]
        }
        
        for info_type, columns in column_info.items():
            filepath = os.path.join(model_dir, f'{info_type}.csv')
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write('column_name\n' + '\n'.join(columns))
            print(f"Saved {info_type} to {filepath}")
        
        # Save test data if available
        if test_output_path and hasattr(self, 'test_data'):
            print(f"\nSaving preprocessed test data to {test_output_path}")
            test_data_to_save = self.test_data.copy()  # Save all test data
            print(f"Total test rows to save: {len(test_data_to_save):,}")
            
            # Convert all transaction amounts to absolute values in test data
            if 'TransactionAmt' in test_data_to_save.columns:
                test_data_to_save['TransactionAmt'] = test_data_to_save['TransactionAmt'].abs()
            
            # Free up memory
            if hasattr(self, 'test_data'):
                del self.test_data
            
            # Ensure test data has same columns in same order
            missing_test_cols = [col for col in final_columns if col not in test_data_to_save.columns]
            for col in missing_test_cols:
                test_data_to_save[col] = 'unknown' if col not in [c[0] for c in core_columns] else 0
            
            test_data_to_save = test_data_to_save[final_columns]
            
            # Apply same conversions to test data
            for col, dtype in core_columns:
                if dtype == 'int':
                    test_data_to_save[col] = test_data_to_save[col].fillna(0).astype(np.int64)
                elif dtype == 'float':
                    test_data_to_save[col] = test_data_to_save[col].fillna(0).astype(float).round(4)
            
            for col in remaining_cols:
                curr_dtype = data_to_save[col].dtype
                if pd.api.types.is_numeric_dtype(curr_dtype):
                    test_data_to_save[col] = test_data_to_save[col].fillna(0).astype(float).round(4)
                else:
                    test_data_to_save[col] = test_data_to_save[col].fillna('unknown').astype(str)
            
            # Save test data in chunks
            chunk_size = 25000  # Smaller chunk size for test data
            total_chunks = (len(test_data_to_save) + chunk_size - 1) // chunk_size
            
            for i in range(0, len(test_data_to_save), chunk_size):
                chunk_num = i // chunk_size + 1
                mode = 'w' if i == 0 else 'a'
                header = i == 0
                
                chunk = test_data_to_save.iloc[i:i + chunk_size]
                chunk.to_csv(
                    test_output_path,
                    mode=mode,
                    header=header,
                    index=False,
                    encoding='utf-8',
                    sep=',',
                    float_format='%.4f'
                )
                print(f"Saved test chunk {chunk_num}/{total_chunks} ({i:,} to {i + len(chunk):,} rows)")
    
    def preprocess(self):
        """Run the complete preprocessing pipeline"""
        print("Starting preprocessing pipeline...")
        
        # Store initial fraud cases count
        self.load_data()
        initial_fraud_count = self.train_data['isFraud'].sum()
        print(f"Initial number of fraud cases: {initial_fraud_count}")
        
        # Run preprocessing steps
        self.handle_missing_values()
        self.encode_categorical()
        self.scale_features()
        self.handle_imbalance()
        
        # Verify no fraud cases were lost
        final_fraud_count = self.train_data_balanced['isFraud'].sum()
        print(f"Final number of fraud cases: {final_fraud_count}")
        
        if final_fraud_count < initial_fraud_count:
            raise ValueError(f"Lost {initial_fraud_count - final_fraud_count} fraud cases during preprocessing!")

if __name__ == "__main__":
    # Set paths
    train_transaction_path = "../data/train_transaction.csv"
    train_identity_path = "../data/train_identity.csv"
    test_transaction_path = "../data/test_transaction.csv"
    test_identity_path = "../data/test_identity.csv"
    
    train_output_path = "../models/preprocessed_train.csv"
    test_output_path = "../models/preprocessed_test.csv"
    
    # Initialize and run preprocessor
    preprocessor = DataPreprocessor(
        train_transaction_path, 
        train_identity_path,
        test_transaction_path,
        test_identity_path
    )
    preprocessor.preprocess()
    preprocessor.save_preprocessed_data(train_output_path, test_output_path)
    print("Preprocessing completed successfully!")
