"""Project-wide constants and paths."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "raw" / "telco_churn.csv"
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
FEATURE_IMPORTANCE_PATH = ROOT / "models" / "feature_importance.csv"
TEST_PREDICTIONS_PATH = ROOT / "reports" / "evaluation" / "test_predictions.csv"
FIGURES_DIR = ROOT / "reports" / "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET = "Churn"
ID_COLUMN = "customerID"

NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "AvgMonthlySpend",
    "ServiceCount",
    "IsNewCustomer",
]

CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

RISK_BANDS = [
    {"name": "Low", "min": 0.00, "max": 0.30, "color": "#2A9D8F"},
    {"name": "Medium", "min": 0.30, "max": 0.55, "color": "#E9C46A"},
    {"name": "High", "min": 0.55, "max": 0.75, "color": "#F4A261"},
    {"name": "Critical", "min": 0.75, "max": 1.01, "color": "#E76F51"},
]

PALETTE = {
    "ink": "#0B1F33",
    "muted": "#5B7085",
    "teal": "#0E7C86",
    "mint": "#2A9D8F",
    "coral": "#E76F51",
    "amber": "#E9C46A",
    "cream": "#F7F3EA",
    "surface": "#FFFFFF",
}
