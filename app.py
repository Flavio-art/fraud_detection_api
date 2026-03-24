"""
Schritt 2: Flask API fuer Fraud Detection
"""
from flask import Flask, request, jsonify
from datetime import datetime
import joblib
import numpy as np

app = Flask(__name__)

# Modell und Scaler laden
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Audit Log (in-memory, in Produktion waere das eine Datenbank)
audit_log = []

@app.route('/health', methods=['GET'])
def health():
    """Health Check – ist die API online?"""
    return jsonify({'status': 'ok', 'model': 'RandomForest v1'})

@app.route('/predict', methods=['POST'])
def predict():
    """Vorhersage: Ist die Transaktion Betrug?"""
    data = request.get_json()

    # Features extrahieren
    features = [
        data['amount'],
        data['num_transactions_30d'],
        data['avg_transaction'],
        data['days_since_last'],
        data['num_countries']
    ]

    # Skalieren und vorhersagen
    scaled = scaler.transform([features])
    prediction = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0][1]

    result = {
        'is_fraud': int(prediction),
        'fraud_probability': round(float(probability), 3),
        'risk_level': 'HIGH' if probability > 0.7 else 'MEDIUM' if probability > 0.3 else 'LOW'
    }

    # Audit Log
    audit_log.append({
        'timestamp': datetime.now().isoformat(),
        'input': data,
        'result': result,
        'model_version': 'v1'
    })

    return jsonify(result)

@app.route('/audit', methods=['GET'])
def get_audit():
    """Audit Trail – alle bisherigen Vorhersagen"""
    return jsonify(audit_log)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
