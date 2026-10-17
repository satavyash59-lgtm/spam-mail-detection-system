import os
import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from features import ThreatSignalExtractor

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

pipeline = joblib.load('artifacts/spam_pipeline.pkl')
extractor = ThreatSignalExtractor()

class ScanRequest(BaseModel):
    text: str

@app.post('/api/scan')
def scan(req: ScanRequest):
    t = req.text
    if not t.strip():
        return {'error': 'Empty'}
    p = float(pipeline.predict_proba([t])[0][1])
    is_threat = p >= 0.50
    h = extractor.transform([t])[0]
    return {
        'threat_score': round(p * 100, 1),
        'is_threat': is_threat,
        'threat_type': 'CREDENTIAL_THEFT' if (h[1] > 0 and h[0] > 0) else ('MALICIOUS_SPAM' if is_threat else 'CLEAN_HAM'),
        'signals': {
            'urls_detected': int(h[0]),
            'urgent_keywords': int(h[1]),
            'exclamations': int(h[2]),
            'uppercase_pct': round(float(h[4]) * 100, 1),
            'char_length': int(h[5])
        }
    }

app.mount('/', StaticFiles(directory='public', html=True), name='public')
