"""Preprocessing service — mirrors WEKA pipeline from the report."""

import numpy as np
import joblib

from app.config import ALL_V_FEATURES, ML_MODELS_DIR, SELECTED_FEATURES


# Global scaler (loaded at startup)
_scaler = None


def load_scaler():
    """Load the fitted StandardScaler from disk."""
    global _scaler
    scaler_path = ML_MODELS_DIR / "scaler.pkl"
    if scaler_path.exists():
        _scaler = joblib.load(scaler_path)
        print(f"  ✅ Scaler loaded from {scaler_path}")
    else:
        print(f"  ⚠️  Scaler not found at {scaler_path}")


def preprocess_transaction(transaction_dict: dict) -> np.ndarray:
    """
    Preprocess a single transaction for scoring.

    Steps (matching report Chapter 5):
    1. Validate fields
    2. Standardize Amount and Time (using training scaler)
    3. Select features (top 11 from Table 3)

    Note: SMOTE is training-only. Never applied to live data.
    """
    # Build feature array in correct order: Time, V1-V28, Amount
    raw = np.zeros(30)
    raw[0] = transaction_dict.get("time", 0)
    for i in range(1, 29):
        raw[i] = transaction_dict.get(f"v{i}", 0)
    raw[29] = transaction_dict.get("amount", 0)

    # Standardize Amount and Time using training stats
    if _scaler is not None:
        scaled_ta = _scaler.transform([[raw[0], raw[29]]])[0]
        raw[0] = scaled_ta[0]
        raw[29] = scaled_ta[1]

    # Build feature dict with standardized values
    feature_names = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    features = dict(zip(feature_names, raw))

    # Select top features for model input
    selected = np.array([features[f] for f in SELECTED_FEATURES])
    return selected


def get_key_features(transaction_dict: dict) -> dict:
    """Extract key features for explainability display."""
    return {
        "V17": transaction_dict.get("v17", 0),
        "V14": transaction_dict.get("v14", 0),
        "V12": transaction_dict.get("v12", 0),
        "V10": transaction_dict.get("v10", 0),
        "V4": transaction_dict.get("v4", 0),
        "V11": transaction_dict.get("v11", 0),
        "Amount": transaction_dict.get("amount", 0),
    }
