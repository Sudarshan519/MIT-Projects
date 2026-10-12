"""Cost analysis service — implements report Section 8.6 cost matrix."""

from app.config import DEFAULT_FN_COST, DEFAULT_FP_COST


def calculate_cost(fn_count: int, fp_count: int,
                   fn_cost: float = DEFAULT_FN_COST,
                   fp_cost: float = DEFAULT_FP_COST) -> dict:
    """
    Calculate total operating cost.

    From report: Total cost = (FN × €250) + (FP × €15)

    Args:
        fn_count: Number of false negatives (missed fraud)
        fp_count: Number of false positives (false alarms)
        fn_cost: Cost per missed fraud (default €250)
        fp_cost: Cost per false alarm (default €15)
    """
    fn_total = fn_count * fn_cost
    fp_total = fp_count * fp_cost
    return {
        "fn_count": fn_count,
        "fp_count": fp_count,
        "fn_cost_per_case": fn_cost,
        "fp_cost_per_case": fp_cost,
        "fn_total_cost": fn_total,
        "fp_total_cost": fp_total,
        "total_cost": fn_total + fp_total,
    }


def calculate_fpr(fp_count: int, tn_count: int) -> float:
    """
    Calculate False Positive Rate.

    FPR = FP / (FP + TN)

    Gap Fix #3: This was missing from the original report.
    """
    if (fp_count + tn_count) == 0:
        return 0.0
    return fp_count / (fp_count + tn_count)


def compare_models(model_results: dict,
                   fn_cost: float = DEFAULT_FN_COST,
                   fp_cost: float = DEFAULT_FP_COST) -> dict:
    """
    Compare all 4 model variants on cost.

    model_results: {
        "j48_standard": {"fn": 69, "fp": 34, "tn": 284281, ...},
        "j48_cost_sensitive": {"fn": ..., "fp": ..., "tn": ..., ...},
        "logreg_standard": {"fn": 74, "fp": 16, "tn": 284299, ...},
        "logreg_threshold_tuned": {"fn": ..., "fp": ..., "tn": ..., ...},
    }
    """
    comparison = {}
    for name, results in model_results.items():
        cost = calculate_cost(results["fn"], results["fp"], fn_cost, fp_cost)
        fpr = calculate_fpr(results["fp"], results["tn"])
        comparison[name] = {
            **cost,
            "fpr": fpr,
            "fpr_pct": f"{fpr * 100:.4f}%",
            "precision": results.get("precision", 0),
            "recall": results.get("recall", 0),
            "f1": results.get("f1", 0),
        }

    # Find cheapest model
    cheapest = min(comparison, key=lambda k: comparison[k]["total_cost"])
    comparison["_winner"] = cheapest
    comparison["_savings"] = (
        max(c["total_cost"] for c in comparison.values() if isinstance(c, dict))
        - comparison[cheapest]["total_cost"]
    )

    return comparison
