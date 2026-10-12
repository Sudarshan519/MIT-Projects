"""Pydantic schemas for request/response validation."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------- Auth ----------
class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    username: Optional[str] = None


class UserLogin(BaseModel):
    username: str
    password: str


class UserOut(BaseModel):
    username: str
    full_name: str
    email: str
    role: str


# ---------- Transaction Scoring ----------
class TransactionInput(BaseModel):
    """Single transaction to score."""
    time: float = Field(..., description="Seconds from first transaction in dataset")
    amount: float = Field(..., ge=0, description="Transaction amount in euros")
    v1: float = 0.0
    v2: float = 0.0
    v3: float = 0.0
    v4: float = 0.0
    v5: float = 0.0
    v6: float = 0.0
    v7: float = 0.0
    v8: float = 0.0
    v9: float = 0.0
    v10: float = 0.0
    v11: float = 0.0
    v12: float = 0.0
    v13: float = 0.0
    v14: float = 0.0
    v15: float = 0.0
    v16: float = 0.0
    v17: float = 0.0
    v18: float = 0.0
    v19: float = 0.0
    v20: float = 0.0
    v21: float = 0.0
    v22: float = 0.0
    v23: float = 0.0
    v24: float = 0.0
    v25: float = 0.0
    v26: float = 0.0
    v27: float = 0.0
    v28: float = 0.0


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    NEW = "new"
    REVIEWING = "reviewing"
    CONFIRMED_FRAUD = "confirmed_fraud"
    RELEASED = "released"


class ScoringResult(BaseModel):
    """Response from scoring a transaction."""
    id: Optional[int] = None
    fraud_probability: float
    decision: str  # "allow", "flag", "block"
    risk_level: RiskLevel
    j48_rule_path: Optional[str] = None
    j48_prediction: Optional[int] = None
    logreg_probability: float
    cost_estimate_fn: float  # Cost if this is a missed fraud
    cost_estimate_fp: float  # Cost if this is a false alarm
    key_features: dict  # Top feature values for explainability
    status: AlertStatus = AlertStatus.NEW
    scored_at: Optional[datetime] = None


class StatusUpdate(BaseModel):
    status: AlertStatus
    reviewer: Optional[str] = None
    notes: Optional[str] = None


# ---------- Dashboard ----------
class DashboardStats(BaseModel):
    total_scored: int
    active_alerts: int
    confirmed_frauds: int
    released: int
    estimated_loss_avoided: float
    false_alarm_rate: float
    false_positive_rate: float  # Gap Fix #3: FPR metric
    model_comparison: dict


class CostAnalysis(BaseModel):
    fn_cost_per_case: float
    fp_cost_per_case: float
    models: dict  # Per-model breakdown


class ThresholdPoint(BaseModel):
    """A single point in the threshold sweep."""
    threshold: float
    precision: float
    recall: float
    f1: float
    fpr: float
    fp_count: int
    fn_count: int
    total_cost: float


class ThresholdSweepResult(BaseModel):
    """Full threshold sweep for LogReg."""
    points: list[ThresholdPoint]
    optimal_threshold: float
    optimal_cost: float
