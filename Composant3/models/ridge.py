import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from Composant3.models.preprocessing import build_preprocessor, FEATURES_TO_DROP, CAT_FEATURES, TREND_FEATURES

ALPHA_GRID = [0.01, 0.1, 1, 10, 50, 100, 200, 500]


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
        ("model", Ridge()),
    ])

    grid = GridSearchCV(
        pipeline,
        param_grid={"model__alpha": ALPHA_GRID},
        cv=TimeSeriesSplit(n_splits=3),
        scoring="neg_mean_absolute_error",
        refit=True,
        n_jobs=-1,
    )

    grid.fit(X_train, y_train)
    print(f"  Best alpha: {grid.best_params_['model__alpha']} "
          f"| CV MAE: {-grid.best_score_:.4f}")
    return grid


def predict(model: GridSearchCV, test_df: pd.DataFrame) -> np.ndarray:
    X_test, _ = _get_X_y(test_df)
    return model.predict(X_test)