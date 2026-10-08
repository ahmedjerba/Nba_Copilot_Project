# models/elastic_net.py
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from Composant3.models.preprocessing import (
    build_preprocessor, FEATURES_TO_DROP, CAT_FEATURES, TREND_FEATURES
)

PARAM_GRID = {
    "model__alpha":   [0.001, 0.01, 0.1, 1.0],
    "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
}

def _get_X_y(df: pd.DataFrame):
    drop = FEATURES_TO_DROP + CAT_FEATURES + TREND_FEATURES
    feature_cols = [c for c in df.columns if c not in drop]
    all_cols = feature_cols + CAT_FEATURES + TREND_FEATURES
    X = df[all_cols]
    y = df["target_pie"].values
    return X, y

def train(train_df: pd.DataFrame) -> GridSearchCV:
    X_train, y_train = _get_X_y(train_df)
    preprocessor = build_preprocessor(train_df)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", ElasticNet(max_iter=5000)),
    ])

    grid = GridSearchCV(
        pipeline,
        param_grid=PARAM_GRID,
        cv=TimeSeriesSplit(n_splits=3),
        scoring="neg_mean_absolute_error",
        refit=True,
        n_jobs=-1,
    )

    grid.fit(X_train, y_train)
    print(f"  Best params: {grid.best_params_} "
          f"| CV MAE: {-grid.best_score_:.4f}")
    return grid

def predict(model: GridSearchCV, test_df: pd.DataFrame) -> np.ndarray:
    X_test, _ = _get_X_y(test_df)
    return model.predict(X_test)