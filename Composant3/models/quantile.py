# models/quantile.py
import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor
from sklearn.pipeline import Pipeline

from Composant3.models.preprocessing import (
    build_preprocessor, FEATURES_TO_DROP, CAT_FEATURES, TREND_FEATURES
)

QUANTILES = [0.10, 0.90]

def _get_X_y(df: pd.DataFrame):
    drop = FEATURES_TO_DROP + CAT_FEATURES + TREND_FEATURES
    feature_cols = [c for c in df.columns if c not in drop]
    all_cols = feature_cols + CAT_FEATURES + TREND_FEATURES
    X = df[all_cols]
    y = df["target_pie"].values
    return X, y

def train(train_df: pd.DataFrame) -> dict:
    X_train, y_train = _get_X_y(train_df)
    preprocessor = build_preprocessor(train_df)

    models = {}
    for q in QUANTILES:
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor(train_df)),
            ("model", QuantileRegressor(quantile=q, alpha=0.1, solver="highs")),
        ])
        pipeline.fit(X_train, y_train)
        models[q] = pipeline
        print(f"  Quantile {q:.0%} trained")

    return models

def predict(models: dict, test_df: pd.DataFrame) -> dict:
    X_test, _ = _get_X_y(test_df)
    return {
        "q10": models[0.10].predict(X_test),
        "q90": models[0.90].predict(X_test),
    }