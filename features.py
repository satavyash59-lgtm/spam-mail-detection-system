import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class ThreatSignalExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.suspicious_keywords = [
            'urgent', 'verify', 'account', 'login', 'suspended', 'password',
            'bank', 'claim', 'won', 'free', 'prize', 'winner', 'click', 'action required'
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for text in X:
            text_str = str(text) if pd.notnull(text) else ''
            length = len(text_str)
            url_count = len(re.findall(r'https?://\S+|www\.\S+|bit\.ly/\S+', text_str))
            text_lower = text_str.lower()
            kw_count = sum(1 for kw in self.suspicious_keywords if kw in text_lower)
            exclamations = text_str.count('!')
            questions = text_str.count('?')
            upper_ratio = sum(1 for c in text_str if c.isupper()) / max(length, 1)
            features.append([url_count, kw_count, exclamations, questions, upper_ratio, length])
        return np.array(features)
