"""ML model scoring service — two-tier scoring from report Chapter 9."""

import warnings
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

# Suppress scikit-learn warnings about feature names when predicting with numpy arrays
warnings.filterwarnings("ignore", message="X does not have valid feature names")

from app.config import (
    DEFAULT_FN_COST, DEFAULT_FP_COST,
    GREY_ZONE_HIGH, GREY_ZONE_LOW,
    ML_MODELS_DIR, SELECTED_FEATURES,
)
from app.models.schemas import AlertStatus, RiskLevel, ScoringResult
from app.services.preprocessing import get_key_features, load_scaler, preprocess_transaction

# Global model references
_models = {}
_threshold_data = None


def load_models():
    """Load all trained models from disk."""
    global _models, _threshold_data
    load_scaler()

    model_files = {
        "j48_standard": "j48_standard.pkl",
        "j48_cost_sensitive": "j48_cost_sensitive.pkl",
        "logreg_standard": "logreg_standard.pkl",
    }

    for name, filename in model_files.items():
        path = ML_MODELS_DIR / filename
        if path.exists():
            _models[name] = joblib.load(path)
            print(f"  ✅ {name} loaded")
        else:
            print(f"  ⚠️  {name} not found at {path}")

    # Load threshold sweep data
    threshold_path = ML_MODELS_DIR / "threshold_sweep.pkl"
    if threshold_path.exists():
        _threshold_data = joblib.load(threshold_path)
        print("  ✅ Threshold sweep data loaded")

    # Load optimal threshold
    optimal_path = ML_MODELS_DIR / "optimal_threshold.pkl"
    if optimal_path.exists():
        _models["optimal_threshold"] = joblib.load(optimal_path)
        print(f"  ✅ Optimal threshold: {_models['optimal_threshold']}")


def _extract_j48_rule_path(model, features: np.ndarray) -> str:
    """Extract the decision path from the J48 (Decision Tree) model."""
    if model is None or not hasattr(model, 'tree_'):
        return "Model not available"

    tree = model.tree_
    node = 0
    path_parts = []
    feature_names = SELECTED_FEATURES

    while tree.children_left[node] != tree.children_right[node]:
        feature_idx = tree.feature[node]
        threshold = tree.threshold[node]
        feature_name = feature_names[feature_idx] if feature_idx < len(feature_names) else f"feature_{feature_idx}"

        if features[feature_idx] <= threshold:
            path_parts.append(f"{feature_name} ≤ {threshold:.4f}")
            node = tree.children_left[node]
        else:
            path_parts.append(f"{feature_name} > {threshold:.4f}")
            node = tree.children_right[node]

    return " → ".join(path_parts)


def score_transaction(transaction_dict: dict) -> ScoringResult:
    """
    Score a single transaction using the two-tier system from Chapter 9:
    1. LogReg scores for a fast probability
    2. If grey-zone, apply J48 rules and attach the reason
    """
    # Preprocess
    features = preprocess_transaction(transaction_dict)
    key_features = get_key_features(transaction_dict)

    # Tier 1: Logistic Regression fast score
    logreg_prob = 0.5
    if "logreg_standard" in _models:
        model = _models["logreg_standard"]
        proba = model.predict_proba(features.reshape(1, -1))[0]
        logreg_prob = float(proba[1])  # Probability of fraud (class=1)

    # Determine risk level and decision
    optimal_threshold = _models.get("optimal_threshold", 0.5)

    if logreg_prob < GREY_ZONE_LOW:
        risk_level = RiskLevel.LOW
        decision = "allow"
    elif logreg_prob < optimal_threshold:
        risk_level = RiskLevel.MEDIUM
        decision = "flag"
    elif logreg_prob < GREY_ZONE_HIGH:
        risk_level = RiskLevel.HIGH
        decision = "flag"
    else:
        risk_level = RiskLevel.CRITICAL
        decision = "block"

    # Tier 2: J48 rules (always compute for explainability)
    j48_prediction = 0
    j48_rule_path = None

    if "j48_standard" in _models:
        j48_model = _models["j48_standard"]
        j48_prediction = int(j48_model.predict(features.reshape(1, -1))[0])
        j48_rule_path = _extract_j48_rule_path(j48_model, features)

        # If in grey zone and J48 says fraud, escalate
        if GREY_ZONE_LOW <= logreg_prob <= GREY_ZONE_HIGH and j48_prediction == 1:
            risk_level = RiskLevel.HIGH
            decision = "block"

    # Cost estimates
    cost_fn = DEFAULT_FN_COST if logreg_prob > 0.3 else 0
    cost_fp = DEFAULT_FP_COST if decision in ("flag", "block") else 0

    return ScoringResult(
        fraud_probability=logreg_prob,
        decision=decision,
        risk_level=risk_level,
        j48_rule_path=j48_rule_path,
        j48_prediction=j48_prediction,
        logreg_probability=logreg_prob,
        cost_estimate_fn=cost_fn,
        cost_estimate_fp=cost_fp,
        key_features=key_features,
        status=AlertStatus.NEW,
    )


def score_batch_transactions(transactions_list: list[dict]) -> list[ScoringResult]:
    """Score a batch of transactions using vectorized predictions to bypass Python/scikit-learn overhead."""
    if not transactions_list:
        return []

    X_list = [preprocess_transaction(tx) for tx in transactions_list]
    X_batch = np.vstack(X_list)

    logreg_probs = np.full(len(transactions_list), 0.5)
    if "logreg_standard" in _models:
        model = _models["logreg_standard"]
        probas = model.predict_proba(X_batch)
        logreg_probs = probas[:, 1]

    j48_preds = np.zeros(len(transactions_list), dtype=int)
    if "j48_standard" in _models:
        j48_model = _models["j48_standard"]
        j48_preds = j48_model.predict(X_batch).astype(int)

    optimal_threshold = _models.get("optimal_threshold", 0.5)
    j48_model = _models.get("j48_standard")

    results = []
    for i, tx_dict in enumerate(transactions_list):
        prob = float(logreg_probs[i])
        j48_pred = int(j48_preds[i])
        features = X_batch[i]

        if prob < GREY_ZONE_LOW:
            risk_level = RiskLevel.LOW
            decision = "allow"
        elif prob < optimal_threshold:
            risk_level = RiskLevel.MEDIUM
            decision = "flag"
        elif prob < GREY_ZONE_HIGH:
            risk_level = RiskLevel.HIGH
            decision = "flag"
        else:
            risk_level = RiskLevel.CRITICAL
            decision = "block"

        j48_rule_path = _extract_j48_rule_path(j48_model, features) if j48_model else None

        if GREY_ZONE_LOW <= prob <= GREY_ZONE_HIGH and j48_pred == 1:
            risk_level = RiskLevel.HIGH
            decision = "block"

        cost_fn = DEFAULT_FN_COST if prob > 0.3 else 0
        cost_fp = DEFAULT_FP_COST if decision in ("flag", "block") else 0

        res = ScoringResult(
            fraud_probability=prob,
            decision=decision,
            risk_level=risk_level,
            j48_rule_path=j48_rule_path,
            j48_prediction=j48_pred,
            logreg_probability=prob,
            cost_estimate_fn=cost_fn,
            cost_estimate_fp=cost_fp,
            key_features=get_key_features(tx_dict),
            status=AlertStatus.NEW,
        )
        results.append(res)

    return results


def get_threshold_sweep_data() -> Optional[dict]:
    """Return threshold sweep data for the interactive slider."""
    return _threshold_data


def get_model_comparison_data() -> dict:
    """Return pre-computed model comparison data for all 4 variants."""
    comparison_path = ML_MODELS_DIR / "model_comparison.pkl"
    if comparison_path.exists():
        return joblib.load(comparison_path)
    # Fallback to report values
    return {
        "j48_standard": {
            "accuracy": 98.12, "precision": 92.6, "recall": 86.0,
            "f1": 89.1, "roc_auc": 0.938, "pr_auc": 0.817,
            "tp": 423, "fp": 34, "fn": 69, "tn": 284281,
            "fpr": 34 / 284315,
            "total_cost": 17760,
        },
        "j48_cost_sensitive": {
            "accuracy": 0, "precision": 0, "recall": 0,
            "f1": 0, "roc_auc": 0, "pr_auc": 0,
            "tp": 0, "fp": 0, "fn": 0, "tn": 0,
            "fpr": 0, "total_cost": 0,
            "_note": "Will be populated after training",
        },
        "logreg_standard": {
            "accuracy": 98.36, "precision": 96.3, "recall": 85.0,
            "f1": 90.3, "roc_auc": 0.980, "pr_auc": 0.937,
            "tp": 418, "fp": 16, "fn": 74, "tn": 284299,
            "fpr": 16 / 284315,
            "total_cost": 18740,
        },
        "logreg_threshold_tuned": {
            "accuracy": 0, "precision": 0, "recall": 0,
            "f1": 0, "roc_auc": 0, "pr_auc": 0,
            "tp": 0, "fp": 0, "fn": 0, "tn": 0,
            "fpr": 0, "total_cost": 0,
            "_note": "Will be populated after training",
        },
    }
