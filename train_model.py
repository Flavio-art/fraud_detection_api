"""
Schritt 1: Modell trainieren und speichern
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# === 1. Fake Daten erstellen ===
np.random.seed(42)
n = 1000

df = pd.DataFrame({
    'amount': np.random.exponential(500, n),
    'num_transactions_30d': np.random.poisson(10, n),
    'avg_transaction': np.random.normal(200, 100, n),
    'days_since_last': np.random.exponential(15, n),
    'num_countries': np.random.poisson(2, n),
})

# Fraud-Logik: hohe Betraege + viele Laender + viele Transaktionen = verdaechtig
fraud_score = (df['amount'] > 800).astype(int) + \
              (df['num_countries'] > 3).astype(int) + \
              (df['num_transactions_30d'] > 15).astype(int)
df['is_fraud'] = (fraud_score >= 2).astype(int)

print(f"Fraud Rate: {df['is_fraud'].mean():.1%}")
print(f"Datensatz: {len(df)} Zeilen\n")

# === 2. Features und Target ===
features = ['amount', 'num_transactions_30d', 'avg_transaction', 'days_since_last', 'num_countries']
X = df[features]
y = df['is_fraud']

# === 3. Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 4. Skalieren ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === 5. Trainieren ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# === 6. Evaluieren ===
y_pred = model.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}\n")
print(classification_report(y_test, y_pred))

# === 7. Feature Importance ===
importance = pd.Series(model.feature_importances_, index=features)
print("\nFeature Importance:")
print(importance.sort_values(ascending=False))

# === 8. Modell + Scaler speichern ===
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\nModell und Scaler gespeichert!")
