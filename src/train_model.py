import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.ensemble import IsolationForest, VotingClassifier
import shap
import matplotlib.pyplot as plt
import os
import joblib
from model_utils import IsolationForestWrapper

class FraudDetectionModel:
    def __init__(self, data_path):
        """Initialize the model"""
        self.data_path = data_path
        self.model_path = os.path.join(os.path.dirname(data_path), 'models')
        os.makedirs(self.model_path, exist_ok=True)
        
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Initialize all models
        self.xgb_model = None
        self.lgb_model = None
        self.cat_model = None
        self.isolation_forest = None
        self.ensemble = None
        
    def engineer_features(self):
        """Add engineered features to improve model performance"""
        print("Engineering additional features...")
        
        # Aggregate features by TransactionID groups
        if 'TransactionID' in self.X.columns:
            group_features = self.X.groupby('TransactionID').agg({
                'TransactionAmt': ['mean', 'std', 'min', 'max'],
                'card1': 'nunique',
                'card2': 'nunique',
                'addr1': 'nunique',
                'addr2': 'nunique'
            }).fillna(0)
            
            # Flatten column names
            group_features.columns = [f'{col[0]}_{col[1]}' for col in group_features.columns]
            
            # Merge back to main features
            self.X = pd.merge(self.X, group_features, left_on='TransactionID', right_index=True, how='left')
        
        # Create interaction features
        if 'TransactionAmt' in self.X.columns and 'card1' in self.X.columns:
            self.X['amt_to_card_ratio'] = self.X['TransactionAmt'] / self.X.groupby('card1')['TransactionAmt'].transform('mean')
        
        # Time-based features
        if 'TransactionDT' in self.X.columns:
            self.X['transaction_hour'] = (self.X['TransactionDT'] / 3600) % 24
            self.X['transaction_day'] = (self.X['TransactionDT'] / (3600 * 24)) % 7
            self.X['is_weekend'] = self.X['transaction_day'].isin([5, 6]).astype(int)
        
        # Fill missing values with median
        self.X = self.X.fillna(self.X.median())
        
        print(f"Final feature count: {self.X.shape[1]}")
    
    def load_data(self):
        """Load preprocessed data"""
        print("Loading preprocessed data...")
        self.data = pd.read_csv(self.data_path)
        self.X = self.data.drop('isFraud', axis=1)
        self.y = self.data['isFraud'].astype(int)
        
        # Engineer additional features
        self.engineer_features()
        
        # Verify target values
        unique_classes = sorted(self.y.unique())
        print(f"\nUnique classes in target: {unique_classes}")
        print("Class distribution:")
        print(self.y.value_counts(normalize=True))
        
    def split_data(self):
        """Split data into train and test sets"""
        print("Splitting data...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
    def train_xgboost(self):
        """Train XGBoost model with optimized parameters"""
        print("Training XGBoost model...")
        
        # Define optimal parameters
        params = {
            'max_depth': 8,
            'learning_rate': 0.01,
            'n_estimators': 1000,
            'min_child_weight': 1,
            'gamma': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': 1,
            'objective': 'binary:logistic',
            'eval_metric': ['auc', 'logloss'],
            'use_label_encoder': False,
            'tree_method': 'hist',  # For faster training
            'random_state': 42
        }
        
        self.xgb_model = xgb.XGBClassifier(**params)
        
        # Train with early stopping
        self.xgb_model.fit(
            self.X_train, 
            self.y_train,
            eval_set=[(self.X_train, self.y_train), (self.X_test, self.y_test)],
            early_stopping_rounds=50,
            verbose=True
        )
        
    def train_lightgbm(self):
        """Train LightGBM model with optimized parameters"""
        print("Training LightGBM model...")
        
        params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 7,
            'num_leaves': 31,
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'auc',
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'random_state': 42,
            'verbose': -1
        }
        
        self.lgb_model = lgb.LGBMClassifier(**params)
        
        # Train with eval set for validation
        eval_set = [(self.X_test, self.y_test)]
        
        self.lgb_model.fit(
            self.X_train,
            self.y_train,
            eval_set=eval_set,
            eval_metric='auc'
        )
        
        print("LightGBM training completed!")
        
    def train_catboost(self):
        """Train CatBoost model with optimized parameters"""
        print("Training CatBoost model...")
        
        params = {
            'iterations': 100,
            'learning_rate': 0.1,
            'depth': 7,
            'l2_leaf_reg': 3,
            'bootstrap_type': 'Bernoulli',
            'subsample': 0.8,
            'eval_metric': 'AUC',
            'random_seed': 42,
            'verbose': 50,
            'task_type': 'CPU',
            'loss_function': 'Logloss'
        }
        
        self.cat_model = CatBoostClassifier(**params)
        
        # Train with validation set
        self.cat_model.fit(
            self.X_train,
            self.y_train,
            eval_set=[(self.X_test, self.y_test)],
            use_best_model=True,
            verbose=50
        )
        
        print("CatBoost training completed!")
        
    def train_isolation_forest(self):
        """Train Isolation Forest model with optimized parameters"""
        print("Training Isolation Forest model...")
        
        # Calculate actual fraud ratio for better contamination setting
        fraud_ratio = np.mean(self.y_train == 1)
        print(f"Actual fraud ratio in training data: {fraud_ratio:.3f}")
        
        # Initialize and train Isolation Forest
        self.isolation_forest = IsolationForest(
            n_estimators=100,          # Number of trees
            max_samples='auto',        # Automatically determine sample size
            contamination=fraud_ratio, # Use actual fraud ratio
            max_features=1.0,         # Use all features
            bootstrap=True,           # Enable bootstrap sampling
            n_jobs=-1,               # Use all CPU cores
            random_state=42,
            verbose=1                 # Show progress
        )
        
        print("Starting Isolation Forest training...")
        self.isolation_forest.fit(self.X_train)
        print("Isolation Forest training completed!")
        
        # Calculate threshold for binary predictions
        print("Calibrating scores...")
        scores = self.isolation_forest.score_samples(self.X_train)
        self.isolation_forest_threshold = np.percentile(scores, fraud_ratio * 100)
        print(f"Calibrated threshold: {self.isolation_forest_threshold:.3f}")
        print("Isolation Forest training and calibration completed successfully!")
        self.isolation_forest_threshold = np.percentile(scores, fraud_ratio * 100)
        print(f"Calibrated threshold: {self.isolation_forest_threshold:.3f}")
        print("Isolation Forest training and calibration completed successfully!")
        
    def create_ensemble(self):
        """Create and train ensemble model using multiple GBMs"""
        print("Creating advanced ensemble model...")
        
        try:
            # Verify all models are trained
            if not all([self.xgb_model, self.lgb_model, self.cat_model, self.isolation_forest]):
                raise ValueError("All models must be trained before creating ensemble")
            
            # Create Isolation Forest wrapper
            if_wrapper = IsolationForestWrapper(
                isolation_forest=self.isolation_forest,
                threshold=self.isolation_forest_threshold
            )
            
            # Create ensemble using VotingClassifier
            self.ensemble_model = VotingClassifier(
                estimators=[
                    ('xgboost', self.xgb_model),
                    ('lightgbm', self.lgb_model),
                    ('catboost', self.cat_model),
                    ('isolation_forest', if_wrapper)
                ],
                voting='soft',
                weights=[0.3, 0.3, 0.3, 0.1]  # Equal weights for GBMs, less for IF
            )
            
            # Train ensemble
            print("Training final ensemble...")
            self.ensemble_model.fit(self.X_train, self.y_train)
            print("Ensemble model training completed!")
            
        except Exception as e:
            print(f"Error creating ensemble: {str(e)}")
            raise
        print("Training CatBoost...")
        self.cat_model = CatBoostClassifier(
            iterations=100,
            learning_rate=0.1,
            depth=7,
            verbose=False,
            random_state=42
        )
        self.cat_model.fit(self.X_train, self.y_train)
        
        # Create Isolation Forest wrapper
        if_wrapper = IsolationForestWrapper(
            isolation_forest=self.isolation_forest,
            threshold=self.isolation_forest_threshold
        )
        
        # Create ensemble using VotingClassifier with all models
        self.ensemble = VotingClassifier(
            estimators=[
                ('xgboost', self.xgb_model),
                ('lightgbm', self.lgb_model),
                ('catboost', self.cat_model),
                ('isolation_forest', if_wrapper)
            ],
            voting='soft',
            weights=[0.3, 0.3, 0.3, 0.1]  # Equal weights for GBMs, less for IF
        )
        
        # Train ensemble
        print("Training final ensemble...")
        self.ensemble.fit(self.X_train, self.y_train)
        return self.ensemble
    
    def evaluate_models(self):
        """Evaluate trained models"""
        print("\nEvaluating models...")
        
        # Individual model evaluations
        models = {
            'XGBoost': self.xgb_model,
            'LightGBM': self.lgb_model,
            'CatBoost': self.cat_model
        }
        
        # Plot setup
        plt.figure(figsize=(12, 8))
        
        # Evaluate each model
        for name, model in models.items():
            print(f"\n{name} Results:")
            y_pred = model.predict(self.X_test)
            print(classification_report(self.y_test, y_pred))
            
            # Calculate ROC curve
            y_proba = model.predict_proba(self.X_test)[:, 1]
            fpr, tpr, _ = roc_curve(self.y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
        
        # Isolation Forest evaluation
        print("\nIsolation Forest Results:")
        if_pred = self.isolation_forest.predict(self.X_test)
        if_pred = np.where(if_pred == 1, 0, 1)  # Convert [-1, 1] to [1, 0]
        print(classification_report(self.y_test, if_pred))
        
        # Isolation Forest ROC
        if_scores = self.isolation_forest.score_samples(self.X_test)
        if_scores = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min())
        if_scores = 1 - if_scores
        if_fpr, if_tpr, _ = roc_curve(self.y_test, if_scores)
        if_auc = auc(if_fpr, if_tpr)
        plt.plot(if_fpr, if_tpr, label=f'Isolation Forest (AUC = {if_auc:.3f})')
        
        # Final Ensemble evaluation
        print("\nFinal Ensemble Results:")
        ensemble_pred = self.ensemble.predict(self.X_test)
        print(classification_report(self.y_test, ensemble_pred))
        
        # Ensemble ROC
        ensemble_proba = self.ensemble.predict_proba(self.X_test)[:, 1]
        ens_fpr, ens_tpr, _ = roc_curve(self.y_test, ensemble_proba)
        ens_auc = auc(ens_fpr, ens_tpr)
        plt.plot(ens_fpr, ens_tpr, label=f'Ensemble (AUC = {ens_auc:.3f})', linewidth=2, linestyle='--')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Fraud Detection Models')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        # Save ROC curves
        output_dir = os.path.dirname(self.model_path)
        plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_shap_values(self):
        """Generate SHAP values for feature importance using Tree Explainer"""
        print("Generating SHAP values...")
        try:
            # Convert data to numpy arrays with explicit dtypes
            X_train_arr = self.X_train.astype('float32').values
            feature_names = list(self.X_train.columns)
            
            # Use TreeExplainer which is much faster than KernelExplainer
            explainer = shap.TreeExplainer(
                self.xgb_model,
                feature_names=feature_names,
                model_output='probability'
            )
            
            # Calculate SHAP values on a subset for speed
            n_samples = min(1000, len(X_train_arr))
            np.random.seed(42)
            sample_indices = np.random.choice(len(X_train_arr), n_samples, replace=False)
            X_sample = X_train_arr[sample_indices]
            
            print(f"Calculating SHAP values on {n_samples} samples...")
            shap_values = explainer(X_sample)
            
            # Create and save summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                shap_values[:, :, 1],  # Select positive class
                X_sample,
                feature_names=feature_names,
                show=False,
                plot_size=(12, 8)
            )
            plt.title('SHAP Feature Importance')
            
            # Save plot
            output_dir = os.path.dirname(self.model_path)
            plt.savefig(
                os.path.join(output_dir, 'shap_importance.png'),
                dpi=300,
                bbox_inches='tight'
            )
            plt.close()
            
            # Calculate and save feature importance scores
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': np.abs(shap_values.values[:, :, 1]).mean(0)
            })
            feature_importance = feature_importance.sort_values('importance', ascending=False)
            
            # Save top 20 features plot
            plt.figure(figsize=(12, 6))
            plt.bar(
                range(min(20, len(feature_importance))),
                feature_importance['importance'][:20]
            )
            plt.xticks(
                range(min(20, len(feature_importance))),
                feature_importance['feature'][:20],
                rotation=45,
                ha='right'
            )
            plt.title('Top 20 Features by SHAP Importance')
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, 'top_20_features.png'),
                dpi=300,
                bbox_inches='tight'
            )
            plt.close()
            
            # Save feature importance to CSV
            feature_importance.to_csv(
                os.path.join(output_dir, 'feature_importance.csv'),
                index=False
            )
            
            print("SHAP analysis completed successfully!")
            
        except Exception as e:
            print(f"\nWarning: Could not generate SHAP values: {str(e)}")
            print("Using XGBoost's built-in feature importance instead...")
            
            importance = self.xgb_model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': importance
            })
            feature_importance = feature_importance.sort_values('importance', ascending=False)
            
            # Plot top 20 features
            plt.figure(figsize=(12, 6))
            plt.bar(
                range(min(20, len(feature_importance))),
                feature_importance['importance'][:20]
            )
            plt.xticks(
                range(min(20, len(feature_importance))),
                feature_importance['feature'][:20],
                rotation=45,
                ha='right'
            )
            plt.title('Top 20 Features by XGBoost Importance')
            plt.tight_layout()
            
            # Save results
            output_dir = os.path.dirname(self.model_path)
            plt.savefig(
                os.path.join(output_dir, 'xgb_importance.png'),
                dpi=300,
                bbox_inches='tight'
            )
            plt.close()
            
            feature_importance.to_csv(
                os.path.join(output_dir, 'feature_importance.csv'),
                index=False
            )
            
            print("\nTop 10 most important features (XGBoost method):")
            print(feature_importance.head(10))
        
    def save_models(self):
        """Save trained models"""
        print("\nSaving models...")
        models_dir = '../models'
        os.makedirs(models_dir, exist_ok=True)
        
        # Save the ensemble model
        joblib.dump(self.ensemble, os.path.join(models_dir, 'ensemble_model.joblib'))
        print("Ensemble model saved!")
        
        # Also save individual models for analysis
        joblib.dump(self.xgb_model, os.path.join(models_dir, 'xgboost_model.joblib'))
        print("XGBoost model saved!")
        
        joblib.dump(self.isolation_forest, os.path.join(models_dir, 'isolation_forest_model.joblib'))
        print("Isolation Forest model saved!")
        
    def create_ensemble(self):
        """Create and train ensemble model combining XGBoost and Isolation Forest"""
        print("\nCreating ensemble model...")
        try:
            # Create wrapper instance
            if_wrapper = IsolationForestWrapper(self.isolation_forest, self.isolation_forest_threshold)
            
            # Create ensemble with weighted voting
            self.ensemble = VotingClassifier(
                estimators=[
                    ('xgboost', self.xgb_model),
                    ('lightgbm', self.lgb_model),
                    ('catboost', self.cat_model),
                    ('isolation_forest', if_wrapper)
                ],
                voting='soft',
                weights=[0.3, 0.3, 0.3, 0.1]  # Equal weights for GBMs, less for Isolation Forest
            )
            
            # Fit the ensemble model
            self.ensemble.fit(self.X_train, self.y_train)
            print("Ensemble model training completed!")
            
        except Exception as e:
            print(f"Error creating ensemble model: {str(e)}")
            raise
    
    def train(self):
        """Run the complete training pipeline"""
        print("Starting model training pipeline...")
        
        # Step 1: Data preparation
        print("\nStep 1: Loading and preprocessing data...")
        self.load_data()
        self.split_data()
        
        # Step 2: Train individual models
        print("\nStep 2: Training individual models...")
        self.train_xgboost()
        self.train_lightgbm()
        self.train_catboost()
        self.train_isolation_forest()
        
        # Step 3: Create and train ensemble
        print("\nStep 3: Creating ensemble model...")
        self.create_ensemble()
        
        # Step 4: Evaluate and analyze
        print("\nStep 4: Evaluating models...")
        self.evaluate_models()
        self.generate_shap_values()
        
        # Step 5: Save models
        print("\nStep 5: Saving models...")
        self.save_models()
        
        print("\nTraining pipeline completed successfully!")

if __name__ == "__main__":
    # Initialize and train models using preprocessed training data
    model = FraudDetectionModel('../models/preprocessed_train.csv')
    model.train()
    print("Model training completed successfully!")