"""
Tests fuer die Fraud Detection API
"""
import sys
sys.path.insert(0, '..')
import joblib
import numpy as np

def test_model_exists():
    """Modell und Scaler muessen existieren"""
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    assert model is not None
    assert scaler is not None

def test_prediction():
    """Modell muss 0 oder 1 vorhersagen"""
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    features = [[500, 10, 200, 15, 2]]
    scaled = scaler.transform(features)
    pred = model.predict(scaled)[0]
    assert pred in [0, 1]

def test_high_risk():
    """Hoher Betrag + viele Laender = Fraud"""
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    features = [[5000, 30, 200, 1, 8]]  # verdaechtig
    scaled = scaler.transform(features)
    prob = model.predict_proba(scaled)[0][1]
    assert prob > 0.5
