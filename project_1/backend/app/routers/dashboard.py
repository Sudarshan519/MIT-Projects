"""Dashboard stats and cost analysis router."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.config import DEFAULT_FN_COST, DEFAULT_FP_COST
from app.models.database import TransactionLog, get_db
from app.models.schemas import UserOut
from app.routers.auth import get_current_user
from app.services.cost_analysis import calculate_fpr
from app.services.scoring import get_model_comparison_data, get_threshold_sweep_data

router = APIRouter()


@router.get("/stats")
async def dashboard_stats(
    db: Session = Depends(get_db),
    current_user: UserOut = Depends(get_current_user),
):
    """Get dashboard statistics."""
    total_scored = db.query(TransactionLog).count()
    active_alerts = db.query(TransactionLog).filter(
        TransactionLog.status.in_(["new", "reviewing"])
    ).count()
    confirmed_frauds = db.query(TransactionLog).filter(
        TransactionLog.status == "confirmed_fraud"
    ).count()
    released = db.query(TransactionLog).filter(
        TransactionLog.status == "released"
    ).count()

    # Estimated loss avoided = confirmed frauds × average fraud cost
    loss_avoided = confirmed_frauds * DEFAULT_FN_COST

    # False alarm rate = released / (released + confirmed_frauds) when both are flagged
    flagged_total = confirmed_frauds + released
    false_alarm_rate = (released / flagged_total * 100) if flagged_total > 0 else 0

    # FPR for scored transactions
    total_legitimate = db.query(TransactionLog).filter(
        TransactionLog.status.in_(["released"])
    ).count()
    false_alarms = db.query(TransactionLog).filter(
        TransactionLog.status == "released",
        TransactionLog.decision.in_(["flag", "block"]),
    ).count()
    fpr = calculate_fpr(false_alarms, total_legitimate - false_alarms) if total_legitimate > 0 else 0

    # Model comparison data
    model_comparison = get_model_comparison_data()

    return {
        "total_scored": total_scored,
        "active_alerts": active_alerts,
        "confirmed_frauds": confirmed_frauds,
        "released": released,
        "estimated_loss_avoided": loss_avoided,
        "false_alarm_rate": round(false_alarm_rate, 2),
        "false_positive_rate": round(fpr * 100, 4),
        "model_comparison": model_comparison,
    }


@router.get("/cost-analysis")
async def cost_analysis(
    fn_cost: float = Query(DEFAULT_FN_COST, description="Cost per missed fraud (€)"),
    fp_cost: float = Query(DEFAULT_FP_COST, description="Cost per false alarm (€)"),
    current_user: UserOut = Depends(get_current_user),
):
    """
    Get cost analysis for all 4 model variants.
    Supports custom cost values for the interactive slider.
    """
    comparison = get_model_comparison_data()

    # Recalculate costs with custom values
    for name, data in comparison.items():
        if isinstance(data, dict) and "fn" in data:
            data["fn_total_cost"] = data["fn"] * fn_cost
            data["fp_total_cost"] = data["fp"] * fp_cost
            data["total_cost"] = data["fn"] * fn_cost + data["fp"] * fp_cost
            data["fpr"] = calculate_fpr(data["fp"], data["tn"])

    return {
        "fn_cost_per_case": fn_cost,
        "fp_cost_per_case": fp_cost,
        "models": comparison,
    }


@router.get("/threshold-sweep")
async def threshold_sweep(
    current_user: UserOut = Depends(get_current_user),
):
    """Get LogReg threshold sweep data for interactive slider."""
    data = get_threshold_sweep_data()
    if data is None:
        return {
            "message": "Threshold sweep data not yet generated. Run model training first.",
            "points": [],
        }
    return data
