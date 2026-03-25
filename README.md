# Fraud Detection API

REST API zur Erkennung verdaechtiger Transaktionen. Random Forest Modell, deployed als Flask API mit Docker und CI/CD.

## Projektstruktur

```
fraud_detection_api/
├── train_model.py          # Modell trainieren und speichern
├── app.py                  # Flask API (Endpoints: /predict, /health, /audit)
├── model.pkl               # Trainiertes Random Forest Modell
├── scaler.pkl              # StandardScaler
├── requirements.txt        # Python Dependencies
├── Dockerfile              # Container Build
├── tests/
│   └── test_app.py         # Unit Tests
└── .github/
    └── workflows/
        └── ci.yml          # CI/CD Pipeline (Test + Docker Build)
```

## Setup

```bash
pip install -r requirements.txt
python train_model.py
python app.py
```

## API Endpoints

```bash
# Health Check
GET /health

# Fraud Prediction
POST /predict
{"amount": 5000, "num_transactions_30d": 30, "avg_transaction": 200, "days_since_last": 1, "num_countries": 8}

# Audit Trail
GET /audit
```

## Tech Stack

Python, Flask, Scikit-learn, Docker, GitHub Actions
