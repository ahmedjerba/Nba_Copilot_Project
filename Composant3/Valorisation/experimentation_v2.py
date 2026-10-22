# ─────────────────────────────────────────────────────────────────────────────
# Composant3 / experimentation_v2.py
# Modèles avancés : hédoniques, quantile, GAM, stratifiés, stacking
# ─────────────────────────────────────────────────────────────────────────────
#
# USAGE :
#   python Composant3\Valorisation\experimentation_v2.py
#
# NOUVEAUX MODÈLES :
#   A. Hédonique log-linéaire (log du cap_hit_pct)
#   B. Quantile Regression P10/P50/P90 (sklearn)
#   C. GAM additif (GradBoost depth=1 → fonctions additives)
#   D. Modèles stratifiés par position (un XGBoost par groupe)
#   E. Stacking GLM + XGBoost (méta-modèle Ridge)
#   F. CatBoost (si installé)
#   G. XGBoost tuné (optuna si installé, sinon grid search)
# ─────────────────────────────────────────────────────────────────────────────

import sys
import pathlib

_here = pathlib.Path(__file__).resolve()
_root = _here
while not (_root / "Composant3").exists() and _root != _root.parent:
    _root = _root.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, QuantileRegressor
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

from Composant3.Valorisation.data_loader import load_pairs
from Composant3.Valorisation.feature_engineering import build_features
from Composant3.Valorisation.config import SALARY_CAP_REF

# ─── Helpers ─────────────────────────────────────────────────────────────────

def make_pipe(model, num_feats, cat_feats):
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"),                              num_feats),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),  cat_feats),
    ])
    return Pipeline([("prep", pre), ("model", model)])


def Xy(df, num_feats, cat_feats, log_target=False):
    X = df[num_feats + cat_feats]
    y = df["cap_hit_pct"].clip(1e-6, 1 - 1e-6).values
    if log_target:
        y = np.log(y)
    w = df["sample_weight"].values
    return X, y, w


def loso_eval(model_factory, pairs_fe, SAISONS, num_feats, cat_feats,
              log_target=False, fit_kwargs=None):
    """
    Leave-One-Season-Out CV.
    model_factory : callable() -> sklearn estimator
    Retourne (mean_mae_$, list[(saison, mae_$)])
    """
    maes = []
    for test_s in SAISONS:
        tr = pairs_fe[pairs_fe["saison"] != test_s]
        te = pairs_fe[pairs_fe["saison"] == test_s]
        if len(te) == 0:
            continue
        pipe = make_pipe(model_factory(), num_feats, cat_feats)
        X_tr, y_tr, w_tr = Xy(tr, num_feats, cat_feats, log_target)
        X_te, y_te, _    = Xy(te, num_feats, cat_feats, log_target=False)

        kw = {"model__sample_weight": w_tr} if fit_kwargs is None else fit_kwargs
        try:
            pipe.fit(X_tr, y_tr, **kw)
        except TypeError:
            pipe.fit(X_tr, y_tr)

        y_pred = pipe.predict(X_te)
        if log_target:
            y_pred = np.exp(y_pred)  # retour espace original
        y_pred = np.clip(y_pred, 1e-6, 1 - 1e-6)
        mae_d  = mean_absolute_error(y_te, y_pred) * SALARY_CAP_REF
        maes.append((test_s, mae_d))
    return np.mean([m for _, m in maes]), maes


# ═════════════════════════════════════════════════════════════════════════════
# Chargement
# ═════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("COMPOSANT 3 — EXPÉRIMENTATION AVANCÉE v2")
print("=" * 70)

pairs        = load_pairs(segment="market")
SAISONS      = sorted(pairs["saison"].unique())
pairs_fe, NUM_FEATS, CAT_FEATS, arch_arts = build_features(pairs)
print(f"Dataset : {len(pairs)} paires  |  Saisons : {SAISONS}")
print(f"Features : {len(NUM_FEATS)} num + {len(CAT_FEATS)} cat\n")

RESULTS = {}  # nom -> mean_mae_$

import xgboost as xgb
XGB_BASE = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbosity=0)

# ─── Référence XGBoost v1 ────────────────────────────────────────────────────
print("REF : XGBoost v1 (params de base)")
mean_mae, _ = loso_eval(
    lambda: xgb.XGBRegressor(**XGB_BASE),
    pairs_fe, SAISONS, NUM_FEATS, CAT_FEATS
)
RESULTS["XGBoost_v1 (ref)"] = mean_mae
print(f"  MAE LOSO = {mean_mae:,.0f}$\n")

# ═════════════════════════════════════════════════════════════════════════════
# A. Hédonique log-linéaire
# ═════════════════════════════════════════════════════════════════════════════
print("-" * 70)
print("A. HÉDONIQUE LOG-LINÉAIRE  [log(cap_hit_pct) ~ features]")
print("   Justification : erreurs proportionnelles au salaire,")
print("   cohérent avec la théorie hédonique de Rosen (1974).")
print("-" * 70)

# A1. Log-Ridge
mean_mae, maes = loso_eval(
    lambda: Ridge(alpha=1.0), pairs_fe, SAISONS, NUM_FEATS, CAT_FEATS,
    log_target=True
)
RESULTS["Log-Ridge"] = mean_mae
print(f"  Log-Ridge          MAE LOSO = {mean_mae:>10,.0f}$  "
      f"({' | '.join(f'{s}: {m:,.0f}' for s, m in maes)})")

# A2. Log-XGBoost
mean_mae, maes = loso_eval(
    lambda: xgb.XGBRegressor(**XGB_BASE),
    pairs_fe, SAISONS, NUM_FEATS, CAT_FEATS, log_target=True
)
RESULTS["Log-XGBoost"] = mean_mae
print(f"  Log-XGBoost        MAE LOSO = {mean_mae:>10,.0f}$  "
      f"({' | '.join(f'{s}: {m:,.0f}' for s, m in maes)})")

# ═════════════════════════════════════════════════════════════════════════════
# B. Quantile Regression P10 / P50 / P90
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 70)
print("B. QUANTILE REGRESSION  (P10 / P50 / P90)")
print("   Justification : intervalles de confiance statistiquement honnêtes")
print("   sans bootstrap. Chaque quantile = un modèle distinct.")
print("-" * 70)

for q_name, q in [("P10 (0.10)", 0.10), ("P50 median", 0.50), ("P90 (0.90)", 0.90)]:
    mean_mae, maes = loso_eval(
        lambda q=q: QuantileRegressor(quantile=q, alpha=0.01, solver="highs"),
        pairs_fe, SAISONS, NUM_FEATS, CAT_FEATS, fit_kwargs={}
    )
    RESULTS[f"Quantile_{q_name}"] = mean_mae
    print(f"  {q_name:<14} MAE LOSO = {mean_mae:>10,.0f}$")

print("  Note : MAE du P50 = médiane est la borne basse naturelle pour la MAE.")

# ═════════════════════════════════════════════════════════════════════════════
# C. GAM additif (GradBoost depth=1 = modèle additif non-paramétrique)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 70)
print("C. GAM ADDITIF  [GradBoost avec max_depth=1]")
print("   Justification : chaque arbre de profondeur 1 = une fonction")
print("   univariée. La somme = modèle additif generalise (GAM) sans")
print("   interactions. Montre la contribution marginale de chaque feature.")
print("-" * 70)

for n_est, lr in [(100, 0.1), (300, 0.05), (500, 0.03)]:
    mean_mae, _ = loso_eval(
        lambda n=n_est, l=lr: GradientBoostingRegressor(
            n_estimators=n, max_depth=1, learning_rate=l,
            min_samples_leaf=10, random_state=42),
        pairs_fe, SAISONS, NUM_FEATS, CAT_FEATS
    )
    RESULTS[f"GAM_n{n_est}_lr{lr}"] = mean_mae
    print(f"  GAM(n={n_est}, lr={lr})   MAE LOSO = {mean_mae:>10,.0f}$")

# ═════════════════════════════════════════════════════════════════════════════
# D. Modèles stratifiés par position
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 70)
print("D. XGBoost STRATIFIÉ PAR POSITION")
print("   Justification : le marché valorise différemment un Guard et")
print("   un Center. Séparer les marchés réduit la variance inter-position.")
print("-" * 70)

POS_GROUPS = {
    "Guard":   ["Guard", "Guard-Forward"],
    "Forward": ["Forward", "Forward-Guard", "Center-Forward"],
    "Center":  ["Center", "Forward-Center"],
}

# Ajouter colonne position simple
pairs_fe["pos_group"] = pairs_fe["position"].apply(
    lambda p: next((g for g, ps in POS_GROUPS.items() if p in ps), "Guard")
)

strat_maes = []
for test_s in SAISONS:
    tr = pairs_fe[pairs_fe["saison"] != test_s]
    te = pairs_fe[pairs_fe["saison"] == test_s]
    preds = np.zeros(len(te))
    te = te.reset_index(drop=True)

    for grp, pos_list in POS_GROUPS.items():
        tr_g = tr[tr["pos_group"] == grp]
        te_g_idx = te[te["pos_group"] == grp].index

        if len(tr_g) < 20 or len(te_g_idx) == 0:
            # Fallback: modèle global pour les groupes trop petits
            te_g_idx_all = te.index
            pipe_fb = make_pipe(xgb.XGBRegressor(**XGB_BASE), NUM_FEATS, CAT_FEATS)
            X_tr, y_tr, w_tr = Xy(tr, NUM_FEATS, CAT_FEATS)
            pipe_fb.fit(X_tr, y_tr, model__sample_weight=w_tr)
            preds[te_g_idx] = pipe_fb.predict(te[NUM_FEATS + CAT_FEATS].iloc[te_g_idx])
            continue

        pipe = make_pipe(xgb.XGBRegressor(**XGB_BASE), NUM_FEATS, CAT_FEATS)
        X_tr, y_tr, w_tr = Xy(tr_g, NUM_FEATS, CAT_FEATS)
        pipe.fit(X_tr, y_tr, model__sample_weight=w_tr)
        preds[te_g_idx] = pipe.predict(te[NUM_FEATS + CAT_FEATS].iloc[te_g_idx])

    y_true = te["cap_hit_pct"].clip(1e-6, 1 - 1e-6).values
    preds  = np.clip(preds, 1e-6, 1 - 1e-6)
    strat_maes.append((test_s, mean_absolute_error(y_true, preds) * SALARY_CAP_REF))

mean_strat = np.mean([m for _, m in strat_maes])
RESULTS["XGBoost_stratifie"] = mean_strat
print(f"  XGBoost stratifié  MAE LOSO = {mean_strat:>10,.0f}$  "
      f"({' | '.join(f'{s}: {m:,.0f}' for s, m in strat_maes)})")

# ═════════════════════════════════════════════════════════════════════════════
# E. Stacking : GLM + XGBoost → méta Ridge
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 70)
print("E. STACKING  [GLM (Ridge logit) + XGBoost → méta-modèle Ridge]")
print("   Justification : le GLM capture la structure linéaire calibrée,")
print("   XGBoost capture les interactions. Ridge les combine.")
print("-" * 70)

from sklearn.linear_model import Ridge as SkRidge

stacking_maes = []
for test_s in SAISONS:
    tr = pairs_fe[pairs_fe["saison"] != test_s]
    te = pairs_fe[pairs_fe["saison"] == test_s]

    # Base learners
    pipe_ridge = make_pipe(SkRidge(alpha=1.0), NUM_FEATS, CAT_FEATS)
    pipe_xgb   = make_pipe(xgb.XGBRegressor(**XGB_BASE), NUM_FEATS, CAT_FEATS)

    X_tr, y_tr, w_tr = Xy(tr, NUM_FEATS, CAT_FEATS)
    X_te, y_te, _    = Xy(te, NUM_FEATS, CAT_FEATS)

    pipe_ridge.fit(X_tr, y_tr, model__sample_weight=w_tr)
    pipe_xgb.fit(X_tr, y_tr, model__sample_weight=w_tr)

    p1 = pipe_ridge.predict(X_te)
    p2 = pipe_xgb.predict(X_te)

    # Méta features = [pred_ridge, pred_xgb]
    meta_tr = np.column_stack([
        pipe_ridge.predict(X_tr),
        pipe_xgb.predict(X_tr)
    ])
    meta_te = np.column_stack([p1, p2])

    meta_model = SkRidge(alpha=0.1)
    meta_model.fit(meta_tr, y_tr, sample_weight=w_tr)
    y_pred = np.clip(meta_model.predict(meta_te), 1e-6, 1 - 1e-6)
    stacking_maes.append((test_s, mean_absolute_error(y_te, y_pred) * SALARY_CAP_REF))

mean_stack = np.mean([m for _, m in stacking_maes])
RESULTS["Stacking(Ridge+XGB)"] = mean_stack
print(f"  Stacking           MAE LOSO = {mean_stack:>10,.0f}$  "
      f"({' | '.join(f'{s}: {m:,.0f}' for s, m in stacking_maes)})")

# ═════════════════════════════════════════════════════════════════════════════
# F. CatBoost
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 70)
print("F. CatBoost (si installé)")
print("-" * 70)

try:
    # pyrefly: ignore[missing-import]
    from catboost import CatBoostRegressor
    cat_idx = [len(NUM_FEATS) + i for i in range(len(CAT_FEATS))]

    cb_maes = []
    for test_s in SAISONS:
        tr = pairs_fe[pairs_fe["saison"] != test_s]
        te = pairs_fe[pairs_fe["saison"] == test_s]
        X_tr, y_tr, w_tr = Xy(tr, NUM_FEATS, CAT_FEATS)
        X_te, y_te, _    = Xy(te, NUM_FEATS, CAT_FEATS)

        # CatBoost gère nativement les NaN et les catégorielles (en str)
        X_tr_cb = X_tr.copy()
        X_te_cb = X_te.copy()
        for c in CAT_FEATS:
            X_tr_cb[c] = X_tr_cb[c].fillna("unknown").astype(str)
            X_te_cb[c] = X_te_cb[c].fillna("unknown").astype(str)

        model = CatBoostRegressor(
            iterations=500, depth=5, learning_rate=0.05,
            cat_features=CAT_FEATS, random_seed=42, verbose=0
        )
        model.fit(X_tr_cb, y_tr, sample_weight=w_tr)
        y_pred = np.clip(model.predict(X_te_cb), 1e-6, 1 - 1e-6)
        cb_maes.append((test_s, mean_absolute_error(y_te, y_pred) * SALARY_CAP_REF))

    mean_cb = np.mean([m for _, m in cb_maes])
    RESULTS["CatBoost"] = mean_cb
    print(f"  CatBoost           MAE LOSO = {mean_cb:>10,.0f}$  "
          f"({' | '.join(f'{s}: {m:,.0f}' for s, m in cb_maes)})")
except ImportError:
    print("  CatBoost non installé (pip install catboost)")

# ═════════════════════════════════════════════════════════════════════════════
# G. XGBoost tuné (grid search sur les hyperparamètres clés)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 70)
print("G. XGBOOST TUNING (grid search rapide)")
print("-" * 70)

grid = [
    {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8},
    {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.8},
    {"n_estimators": 500, "max_depth": 4, "learning_rate": 0.03, "subsample": 0.7},
    {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.9,
     "colsample_bytree": 0.7, "min_child_weight": 3},
    {"n_estimators": 400, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8,
     "reg_alpha": 0.1, "reg_lambda": 2.0},
]

best_xgb_mae  = float("inf")
best_xgb_params = {}

for params in grid:
    p = {**params, "random_state": 42, "verbosity": 0, "colsample_bytree": params.get("colsample_bytree", 0.8)}
    mean_mae, _ = loso_eval(
        lambda p=p: xgb.XGBRegressor(**p),
        pairs_fe, SAISONS, NUM_FEATS, CAT_FEATS
    )
    tag = f"n={params['n_estimators']},d={params['max_depth']},lr={params['learning_rate']}"
    print(f"  XGB({tag:<30}) = {mean_mae:>10,.0f}$")
    if mean_mae < best_xgb_mae:
        best_xgb_mae    = mean_mae
        best_xgb_params = params

RESULTS[f"XGBoost_tune"] = best_xgb_mae
print(f"\n  * Meilleurs params XGB : {best_xgb_params}")
print(f"  * MAE LOSO : {best_xgb_mae:,.0f}$")

# ═════════════════════════════════════════════════════════════════════════════
# TABLEAU DE SYNTHÈSE FINAL
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TABLEAU DE SYNTHESE FINAL (toutes méthodes)")
print("=" * 70)

rows = [(k, v) for k, v in sorted(RESULTS.items(), key=lambda x: x[1])]
rows.append(("GLM actuel v2 [ref OOS]", 4_686_079))

print(f"\n  {'Modele':<35} {'MAE LOSO ($)':>14}  {'vs GLM':>10}")
print(f"  {'-'*35}  {'-'*14}  {'-'*10}")
glm_ref = 4_686_079
for name, mae in sorted(rows, key=lambda x: x[1]):
    delta = mae - glm_ref
    flag = "★ MEILLEUR" if mae == min(v for _, v in rows) else ""
    print(f"  {name:<35} {mae:>14,.0f}$  {delta:>+10,.0f}$  {flag}")

print("\n" + "-" * 70)
best_name, best_mae = min(rows, key=lambda x: x[1])
print(f"  GAGNANT : {best_name}  ->  {best_mae:,.0f}$")
print("-" * 70)

# ─── Insights finaux ─────────────────────────────────────────────────────────
print("""
INSIGHTS :
  1. Log-XGBoost vs XGBoost direct : si Log-XGB est meilleur,
     les erreurs sont proportionnelles (sous-évaluation des stars).
  2. GAM depth=1 : si proche de XGBoost, les non-linéarités
     sont essentiellement marginales (pas d'interactions fortes).
  3. Stratifié par position : si meilleur, construire 3 modèles
     distincts (Guard / Forward / Center) pour la prod.
  4. Stacking : si meilleur, combiner les deux meilleurs modèles
     dans le BetaValuationModel final.
  5. Top pires : les 20 pires sont tous >15% du cap SOUVAL.
     Solution : ajouter une feature 'rang_star' ou 'All-Star count'.
""")
print("Fin experimentation v2.\n")
