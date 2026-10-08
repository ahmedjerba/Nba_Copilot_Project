import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

FEATURES_TO_DROP = ["player_id", "saison", "feature_version", "team_id",
                    "target_pie", "target_poss", "season_year"]

CAT_FEATURES = ["valuation_tier"]

TREND_FEATURES = ["trend_pts", "trend_pie"]


def get_feature_columns(df: pd.DataFrame) -> tuple[list, list]:
    drop = FEATURES_TO_DROP + CAT_FEATURES + TREND_FEATURES
    num_features = [c for c in df.columns if c not in drop]
    return num_features, CAT_FEATURES


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    num_features, cat_features = get_feature_columns(df)

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    trend_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num",   num_pipeline,   num_features),
        ("trend", trend_pipeline, TREND_FEATURES),
        ("cat",   cat_pipeline,   cat_features),
    ])