"""Ranking and probability metrics for a fixed outreach capacity."""
import math
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def top_mask(scores, fraction=0.10):
    if not 0 < fraction <= 1 or len(scores) == 0:
        raise ValueError("Capacity must be in (0, 1] and scores must be nonempty.")
    k = math.ceil(len(scores) * fraction)
    selected = np.zeros(len(scores), dtype=bool)
    # Stable tie-breaking: input rows are ordered by encounter ID before scoring.
    selected[np.argsort(-np.asarray(scores), kind="stable")[:k]] = True
    return selected


def metrics(y, scores, fraction=0.10):
    y, scores = np.asarray(y), np.asarray(scores)
    selected = top_mask(scores, fraction)
    rate = float(y.mean())
    precision = float(y[selected].mean())
    observed, predicted = calibration_curve(y, scores, n_bins=10, strategy="quantile")
    return {
        "n": len(y), "positive_count": int(y.sum()), "positive_rate": rate,
        "roc_auc": float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else None,
        "average_precision": float(average_precision_score(y, scores)) if y.sum() else None,
        "brier_score": float(brier_score_loss(y, scores)),
        "outreach_count": int(selected.sum()), "capacity_fraction": fraction,
        "readmissions_in_top_capacity": int(y[selected].sum()),
        "precision_at_capacity": precision,
        "recall_at_capacity": float(y[selected].sum() / y.sum()) if y.sum() else None,
        "lift_at_capacity": precision / rate if rate else None,
        "calibration": [{"predicted": float(p), "observed": float(o)} for p, o in zip(predicted, observed)],
    }


def subgroup_audit(frame, scores, fraction=0.10):
    """Use one overall capacity allocation, not a different threshold per group."""
    selected = top_mask(scores, fraction)
    result = []
    for column in ["age", "race", "gender"]:
        for group in sorted(frame[column].unique()):
            mask = (frame[column] == group).to_numpy()
            y = frame.loc[mask, "label"].to_numpy()
            assigned = selected[mask]
            result.append({
                "attribute": column, "group": group, "n": int(mask.sum()),
                "positives": int(y.sum()), "positive_rate": float(y.mean()),
                "selection_rate": float(assigned.mean()),
                "recall_at_global_capacity": float(y[assigned].sum() / y.sum()) if y.sum() else None,
                "precision_at_global_capacity": float(y[assigned].mean()) if assigned.sum() else None,
                "brier_score": float(brier_score_loss(y, np.asarray(scores)[mask])),
                "small_group": bool(len(y) < 200 or y.sum() < 20),
            })
    return result
