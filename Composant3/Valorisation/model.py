# ─────────────────────────────────────────────
# Composant3 / model.py
# Beta Regression — entraînement, évaluation, prédiction
# ─────────────────────────────────────────────

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error

from Composant3.Valorisation.config import SALARY_CAP_REF


class BetaValuationModel:
    """
    Beta Regression (GLM Binomial + lien Logit) pour prédire cap_hit_pct.
    Conçue pour les cibles bornées entre 0 et 1.

    Usage typique :
        model = BetaValuationModel()
        model.fit(pairs, num_features, cat_features, sample_weight_col="sample_weight")
        result = model.predict_player(player_row, num_features, cat_features)
    """

    def __init__(self):
        self.glm_model    = None
        self.imputer      = None
        self.encoder      = None
        self.alpha        = 0.01
        self.num_features: list[str] = []
        self.cat_features: list[str] = []
        self.feature_names: list[str] = []

    # ──────────────────────────────────────────────────────────────────────
    # Préparation interne
    # ──────────────────────────────────────────────────────────────────────

    def _prepare_X(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        X_num = df[self.num_features].copy()

        if fit:
            self.imputer = SimpleImputer(strategy="median")
            X_num_imp = self.imputer.fit_transform(X_num)
        else:
            X_num_imp = self.imputer.transform(X_num)

        X_num_df = pd.DataFrame(X_num_imp, columns=self.num_features,
                                index=df.index)

        X_cat_raw = df[self.cat_features].fillna("unknown")
        if fit:
            self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            X_cat_arr = self.encoder.fit_transform(X_cat_raw)
        else:
            X_cat_arr = self.encoder.transform(X_cat_raw)

        cat_names = self.encoder.get_feature_names_out(self.cat_features).tolist()
        X_cat_df  = pd.DataFrame(X_cat_arr, columns=cat_names, index=df.index)

        return pd.concat([X_num_df, X_cat_df], axis=1)

    def _build_formula(self, feature_names: list[str], target: str) -> str:
        terms = []
        for c in feature_names:
            if any(s in c for s in [" ", "-", "/", "("]):
                terms.append(f"Q('{c}')")
            else:
                terms.append(c)
        return f"{target} ~ " + " + ".join(terms)

    # ──────────────────────────────────────────────────────────────────────
    # Entraînement
    # ──────────────────────────────────────────────────────────────────────

    def fit(
        self,
        df: pd.DataFrame,
        num_features: list[str],
        cat_features: list[str],
        sample_weight_col: str = "sample_weight",
        alpha: float = 0.01,
    ) -> "BetaValuationModel":
        """
        Entraîne la Beta Regression sur le dataset df.
        alpha > 0 : régularisation Ridge (L2) pour éviter l'overfit.
        alpha = 0 : GLM standard sans régularisation.
        """
        self.num_features = num_features
        self.cat_features = cat_features
        self.alpha        = alpha

        X_full = self._prepare_X(df, fit=True)
        self.feature_names = X_full.columns.tolist()

        y = df["cap_hit_pct"].clip(1e-6, 1 - 1e-6).values

        X_model = X_full.copy().reset_index(drop=True)
        X_model["cap_hit_pct"] = y

        weights = df[sample_weight_col].values if sample_weight_col in df.columns else None

        formula = self._build_formula(self.feature_names, "cap_hit_pct")

        glm = smf.glm(
            formula=formula,
            data=X_model,
            family=sm.families.Binomial(link=sm.families.links.Logit()),
            freq_weights=weights,
        )

        if alpha > 0:
            self.glm_model = glm.fit_regularized(alpha=alpha, L1_wt=0.0)
        else:
            self.glm_model = glm.fit(disp=0)

        return self

    # ──────────────────────────────────────────────────────────────────────
    # Évaluation
    # ──────────────────────────────────────────────────────────────────────

    def evaluate(self, df: pd.DataFrame) -> dict:
        """
        Calcule MAE en % du cap et en $ sur le dataset df.
        df doit contenir 'cap_hit_pct'.
        """
        X_full  = self._prepare_X(df, fit=False)
        X_model = X_full.copy().reset_index(drop=True)
        y_true  = df["cap_hit_pct"].values

        y_pred = self.glm_model.predict(X_model)

        mae_pct     = mean_absolute_error(y_true, y_pred)
        mae_dollars = mae_pct * SALARY_CAP_REF
        baseline    = mean_absolute_error(y_true, np.full(len(y_true), y_true.mean()))
        gain        = (baseline - mae_pct) / baseline

        return {
            "mae_pct":              round(mae_pct, 4),
            "mae_dollars":          round(mae_dollars),
            "baseline_mae_dollars": round(baseline * SALARY_CAP_REF),
            "gain_vs_baseline":     round(gain, 4),
            "n":                    len(df),
        }

    # ──────────────────────────────────────────────────────────────────────
    # Prédiction individuelle
    # ──────────────────────────────────────────────────────────────────────

    def predict_player(
        self,
        player_row: pd.DataFrame,
        salary_cap: float = SALARY_CAP_REF,
    ) -> dict:
        """
        Prédit la valeur marché d'un joueur à partir d'une ligne de features.
        """
        X_full  = self._prepare_X(player_row, fit=False)
        X_model = X_full.reset_index(drop=True)

        val_pct = float(self.glm_model.predict(X_model).iloc[0])
        val_usd = val_pct * salary_cap

        try:
            pred_obj = self.glm_model.get_prediction(X_model)
            ci       = pred_obj.summary_frame(alpha=0.20)
            low_pct  = float(ci["mean_ci_lower"].iloc[0])
            high_pct = float(ci["mean_ci_upper"].iloc[0])
        except Exception:
            low_pct  = val_pct * 0.85
            high_pct = val_pct * 1.15

        return {
            "valeur_pct":     round(val_pct, 4),
            "valeur_dollars": round(val_usd),
            "p10_dollars":    round(low_pct  * salary_cap),
            "p90_dollars":    round(high_pct * salary_cap),
            "salary_cap":     salary_cap,
        }

    def predict_surplus(
        self,
        player_row: pd.DataFrame,
        cap_hit: float,
        salary_cap: float = SALARY_CAP_REF,
    ) -> dict:
        """
        Calcule le surplus value = valeur estimée − salaire actuel.
        cap_hit : salaire actuel du joueur en dollars.
        """
        pred    = self.predict_player(player_row, salary_cap)
        surplus = pred["valeur_dollars"] - cap_hit
        signal  = "SOUS-PAYÉ 🟢" if surplus > 0 else "SUR-PAYÉ 🔴"
        return {
            **pred,
            "cap_hit": round(cap_hit),
            "surplus": round(surplus),
            "signal":  signal,
        }