import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from features import ThreatSignalExtractor

raw_data = {
    'text': [
        'URGENT: Your bank account has been locked. Verify identity at https://secure-login.com',
        'Hey, are we meeting at the college lab at 3 PM today?',
        'CONGRATULATIONS! You won a $1000 Amazon card! Text CLAIM to 8088 now!',
        'Can you review the PR I raised for the backend authentication routes?',
        'Alert: Unauthorized access detected. Change your password immediately: http://bit.ly/3x8',
        'Let us wrap up the assignments before tomorrow morning.',
        'Free entry into annual lucky draw! Reply WINNER to participate.',
        'Please find attached the slides from today computer networks class.',
        'Your tax refund is ready. Submit personal identity details here: https://tax-gov.info',
        'Are you coming to the gym this evening?'
    ] * 50,
    'label': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham'] * 50
}

df = pd.DataFrame(raw_data)
X = df['text']
y = df['label'].map({'spam': 1, 'ham': 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

pipeline = Pipeline([
    ('features', FeatureUnion([
        ('nlp_tfidf', TfidfVectorizer(ngram_range=(1, 2), stop_words='english', max_features=4000, sublinear_tf=True)),
        ('threat_heuristics', Pipeline([
            ('extractor', ThreatSignalExtractor()),
            ('scaler', StandardScaler())
        ]))
    ])),
    ('classifier', LogisticRegression(C=1.5, class_weight='balanced', max_iter=1000, random_state=42))
])

print('Training pipeline...')
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing/Spam']))
print(f'ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}')

os.makedirs('artifacts', exist_ok=True)
joblib.dump(pipeline, 'artifacts/spam_pipeline.pkl')
print('[SUCCESS] Model saved to artifacts/spam_pipeline.pkl')
