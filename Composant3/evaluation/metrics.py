import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def _mae_rmse(y_true, y_pred):
    return {
        "mae":  round(mean_absolute_error(y_true, y_pred), 4),
        "rmse": round(root_mean_squared_error(y_true, y_pred), 4),
        "n":    len(y_true),
    }


def by_age_bucket(y_true, y_pred, ages) -> pd.DataFrame:
    bins   = [0, 22, 25, 28, 32, 99]
    labels = ["≤22", "23-25", "26-28", "29-32", "33+"]
    buckets = pd.cut(ages, bins=bins, labels=labels)

    records = []
    for label in labels:
        mask = buckets == label
        if mask.sum() < 5:
            continue
        row = _mae_rmse(y_true[mask], y_pred[mask])
        row["age_bucket"] = label
        records.append(row)

    return pd.DataFrame(records)[["age_bucket", "n", "mae", "rmse"]]


def by_tier(y_true, y_pred, tiers) -> pd.DataFrame:
    records = []
    for tier in sorted(tiers.unique()):
        mask = tiers == tier
        if mask.sum() < 5:
            continue
        row = _mae_rmse(y_true[mask], y_pred[mask])
        row["valuation_tier"] = tier
        records.append(row)

    return pd.DataFrame(records)[["valuation_tier", "n", "mae", "rmse"]]

def coverage(y_true, q10, q90) -> dict:
    """Un intervalle 80% bien calibré doit couvrir ~80% des cas."""
    covered = ((y_true >= q10) & (y_true <= q90)).mean()
    width   = (q90 - q10).mean()
    return {
        "coverage": round(float(covered), 4),
        "avg_width": round(float(width), 4),
    }