# models/bootstrap.py
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from Composant3.models.preprocessing import (
    build_preprocessor, FEATURES_TO_DROP, CAT_FEATURES, TREND_FEATURES
)

N_BOOTSTRAP = 200

def _get_X_y(df: pd.DataFrame):
    drop = FEATURES_TO_DROP + CAT_FEATURES + TREND_FEATURES
    feature_cols = [c for c in df.columns if c not in drop]
    all_cols = feature_cols + CAT_FEATURES + TREND_FEATURES
    X = df[all_cols]
    y = df["target_pie"].values
    return X, y

def train_and_predict(train_df: pd.DataFrame, test_df: pd.DataFrame, alpha: int = 200) -> dict:
    rng = np.random.default_rng(42)
    n_train = len(train_df)
    preds = np.zeros((N_BOOTSTRAP, len(test_df)))

    X_train, y_train = _get_X_y(train_df)
    X_test, _       = _get_X_y(test_df)

    for i in range(N_BOOTSTRAP):
        idx = rng.integers(0, n_train, size=n_train)
        X_s = X_train.iloc[idx]
        y_s = y_train[idx]

        pipe = Pipeline([
            ("preprocessor", build_preprocessor(train_df)),
            ("model", Ridge(alpha=alpha)),
        ])
        pipe.fit(X_s, y_s)
        preds[i] = pipe.predict(X_test)

    return {
        "q10": np.percentile(preds, 10, axis=0),
        "q90": np.percentile(preds, 90, axis=0),
    }