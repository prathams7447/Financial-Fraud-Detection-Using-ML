import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class IsolationForestWrapper(BaseEstimator, ClassifierMixin):
    """Wrapper class to make Isolation Forest compatible with sklearn's VotingClassifier"""
    def __init__(self, isolation_forest, threshold):
        self.isolation_forest = isolation_forest
        self.threshold = threshold
    
    def fit(self, X, y=None):
        """Fit the isolation forest if not already fitted"""
        if not hasattr(self.isolation_forest, 'offset_'):
            self.isolation_forest.fit(X)
        return self
    
    def predict_proba(self, X):
        scores = self.isolation_forest.score_samples(X)
        probs = 1 / (1 + np.exp(-(scores - self.threshold)))
        return np.vstack([1-probs, probs]).T
    
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.3).astype(int)  # Lower threshold to 0.3 for higher sensitivity
