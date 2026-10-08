import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from Composant3.baselines.last_year import predict as last_year_predict
from Composant3.baselines.age_curve import predict as age_curve_predict
from Composant3.baselines.mean_reversion import predict as mean_reversion_predict
from Composant3.models import ridge
from Composant3.models import lgbm  # import
from Composant3.models import elastic_net

from Composant3.models import quantile
from Composant3.evaluation.metrics import coverage

from Composant3.models import bootstrap
from Composant3.evaluation.metrics import coverage

FOLDS = [
    {"train": [2021], "test": 2022},
    {"train": [2021, 2022], "test": 2023},
    {"train": [2021, 2022, 2023], "test": 2024},
]

BASELINES = {
    "last_year":      last_year_predict,
    "age_curve":      age_curve_predict,
    "mean_reversion": mean_reversion_predict,
}


def _score(y_true, y_pred, name, test_season, n) -> dict:
    return {
        "model":       name,
        "test_season": test_season,
        "n":           n,
        "mae":         round(mean_absolute_error(y_true, y_pred), 4),
        "rmse":        round(root_mean_squared_error(y_true, y_pred), 4),
    }


def run(df: pd.DataFrame) -> pd.DataFrame:
    records = []

    for fold in FOLDS:
        train = df[df["season_year"].isin(fold["train"])]
        test  = df[df["season_year"] == fold["test"]]
        y_true = test["target_pie"].values
        n = len(test)

        print(f"\n--- Pli test {fold['test']} | train {fold['train']} ---")

        # Baselines
        for name, predict_fn in BASELINES.items():
            preds = predict_fn(train, test)
            records.append(_score(y_true, preds, name, fold["test"], n))

        # Ridge
        model = ridge.train(train)
        preds = ridge.predict(model, test)
        records.append(_score(y_true, preds, "ridge", fold["test"], n))

        # LightGBM
        #model_lgbm = lgbm.train(train)
        #preds = lgbm.predict(model_lgbm, test)
        #records.append(_score(y_true, preds, "lgbm", fold["test"], n))

        # Elastic NeT
        model_en = elastic_net.train(train)
        preds = elastic_net.predict(model_en, test)
        records.append(_score(y_true, preds, "elastic_net", fold["test"], n))

        # Quantile Regression (Interval)
        models_q = quantile.train(train)
        preds_q  = quantile.predict(models_q, test)
        cov      = coverage(y_true, preds_q["q10"], preds_q["q90"])
        print(f"  Quantile coverage: {cov}")

        # Bootstrap
        #preds_bs = bootstrap.train_and_predict(train, test)
        #cov_bs   = coverage(y_true, preds_bs["q10"], preds_bs["q90"])
        #print(f"  Bootstrap coverage: {cov_bs}")


    return pd.DataFrame(records).sort_values(["test_season", "mae"])