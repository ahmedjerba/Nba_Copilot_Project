# ─────────────────────────────────────────────────────────────────────────────
# Composant3 / experimentation.py
# Comparaison de modèles, tuning, analyse des résidus
# ─────────────────────────────────────────────────────────────────────────────
#
# USAGE :
#   cd c:\Users\user\Desktop\nba_copilot
#   python -m Composant3.Valorisation.experimentation
#
# MODÈLES TESTÉS :
#   1. GLM Binomial + Ridge (modèle actuel)
#   2. Ridge Regression   (sklearn)
#   3. ElasticNet
#   4. Random Forest
#   5. Gradient Boosting  (sklearn)
#   6. XGBoost            (si installé)
#
# ÉVALUATIONS :
#   - Cross-validation temporelle leave-one-season-out
#   - Tuning alpha (GLM) via LOSO-CV
#   - MAE par segment (low / mid / high)
#   - MAE par position
#   - Analyse des 20 pires prédictions
#   - Tableau de synthèse final
# ─────────────────────────────────────────────────────────────────────────────

import sys
import pathlib

# Auto-détection de la racine du projet (contient Composant3/)
_here = pathlib.Path(__file__).resolve()
_root = _here
while not (_root / "Composant3").exists() and _root != _root.parent:
    _root = _root.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

from Composant3.Valorisation.data_loader import load_pairs
from Composant3.Valorisation.feature_engineering import build_features
from Composant3.Valorisation.config import SALARY_CAP_REF


# ═════════════════════════════════════════════════════════════════════════════
# 0. Chargement et préparation
# ═════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("COMPOSANT 3 — EXPÉRIMENTATION MODÈLES")
print("=" * 70)

pairs = load_pairs(segment="market")
SAISONS = sorted(pairs["saison"].unique())
print(f"\nSaisons disponibles : {SAISONS}")
print(f"Dataset total       : {len(pairs)} paires\n")

# Feature engineering global
pairs_fe, NUM_FEATS, CAT_FEATS, arch_arts = build_features(pairs)


def make_sklearn_pipeline(model, num_feats, cat_feats):
    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"),                              num_feats),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),  cat_feats),
    ])
    return Pipeline([("prep", preprocessor), ("model", model)])


def extract_Xy(df, num_feats, cat_feats):
    X = df[num_feats + cat_feats]
    y = df["cap_hit_pct"].clip(1e-6, 1 - 1e-6).values
    w = df["sample_weight"].values
    return X, y, w


# ═════════════════════════════════════════════════════════════════════════════
# 1. Cross-validation temporelle : Leave-One-Season-Out (LOSO)
# ═════════════════════════════════════════════════════════════════════════════

print("-" * 70)
print("1. CROSS-VALIDATION TEMPORELLE (Leave-One-Season-Out)")
print("-" * 70)

SKLEARN_MODELS = {
    "Ridge(a=0.1)":  Ridge(alpha=0.1),
    "Ridge(a=1.0)":  Ridge(alpha=1.0),
    "ElasticNet":    ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=2000),
    "RandomForest":  RandomForestRegressor(n_estimators=200, max_depth=6,
                                           min_samples_leaf=10, random_state=42),
    "GradBoost":     GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                               learning_rate=0.05, random_state=42),
}

try:
    import xgboost as xgb
    SKLEARN_MODELS["XGBoost"] = xgb.XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    print("  [XGBoost detecte OK]")
except ImportError:
    print("  [XGBoost non installe - ignore]")

loso_results = {}

for model_name, sk_model in SKLEARN_MODELS.items():
    maes = []
    for test_saison in SAISONS:
        train_df = pairs_fe[pairs_fe["saison"] != test_saison]
        test_df  = pairs_fe[pairs_fe["saison"] == test_saison]
        if len(test_df) == 0:
            continue
        pipe = make_sklearn_pipeline(sk_model, NUM_FEATS, CAT_FEATS)
        X_tr, y_tr, w_tr = extract_Xy(train_df, NUM_FEATS, CAT_FEATS)
        X_te, y_te, _    = extract_Xy(test_df,  NUM_FEATS, CAT_FEATS)
        pipe.fit(X_tr, y_tr, model__sample_weight=w_tr)
        y_pred = pipe.predict(X_te)
        mae_d  = mean_absolute_error(y_te, y_pred) * SALARY_CAP_REF
        maes.append((test_saison, mae_d))
    loso_results[model_name] = maes
    mean_mae = np.mean([m for _, m in maes])
    detail = " | ".join(f"{s}: {m:,.0f}$" for s, m in maes)
    print(f"  {model_name:<18} MAE LOSO = {mean_mae:>10,.0f}$  ({detail})")

print(f"  {'GLM actuel(v2)':<18} MAE OOS  =  4,686,079$  (2024-25 uniquement - ref)")

# ═════════════════════════════════════════════════════════════════════════════
# 2. Tuning alpha Ridge via LOSO
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "-" * 70)
print("2. TUNING ALPHA RIDGE (proxy GLM via LOSO)")
print("-" * 70)

alpha_grid = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
alpha_results = {}

for alpha in alpha_grid:
    maes = []
    sk_model = Ridge(alpha=alpha)
    for test_saison in SAISONS:
        train_df = pairs_fe[pairs_fe["saison"] != test_saison]
        test_df  = pairs_fe[pairs_fe["saison"] == test_saison]
        if len(test_df) == 0:
            continue
        pipe = make_sklearn_pipeline(sk_model, NUM_FEATS, CAT_FEATS)
        X_tr, y_tr, w_tr = extract_Xy(train_df, NUM_FEATS, CAT_FEATS)
        X_te, y_te, _    = extract_Xy(test_df,  NUM_FEATS, CAT_FEATS)
        pipe.fit(X_tr, y_tr, model__sample_weight=w_tr)
        maes.append(mean_absolute_error(y_te, pipe.predict(X_te)) * SALARY_CAP_REF)
    mean_mae = np.mean(maes)
    alpha_results[alpha] = mean_mae
    print(f"  alpha={alpha:<8} -> MAE LOSO = {mean_mae:>10,.0f}$")

best_alpha = min(alpha_results, key=alpha_results.get)
print(f"\n  * Meilleur alpha : {best_alpha}  ->  {alpha_results[best_alpha]:,.0f}$")

# ═════════════════════════════════════════════════════════════════════════════
# 3. Analyse par SEGMENT
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "-" * 70)
print("3. ANALYSE PAR SEGMENT (test=2024-25)")
print("-" * 70)

train_fe = pairs_fe[pairs_fe["saison"] < "2024-25"]
test_fe  = pairs_fe[pairs_fe["saison"] == "2024-25"]

best_model_name = min(loso_results, key=lambda k: np.mean([m for _, m in loso_results[k]]))
print(f"  Modele : {best_model_name}\n")

best_pipe = make_sklearn_pipeline(SKLEARN_MODELS[best_model_name], NUM_FEATS, CAT_FEATS)
X_tr, y_tr, w_tr = extract_Xy(train_fe, NUM_FEATS, CAT_FEATS)
X_te, y_te, _    = extract_Xy(test_fe,  NUM_FEATS, CAT_FEATS)
best_pipe.fit(X_tr, y_tr, model__sample_weight=w_tr)
y_pred_te = best_pipe.predict(X_te)

test_eval = test_fe.copy()
test_eval["y_true"] = y_te
test_eval["y_pred"] = np.clip(y_pred_te, 1e-6, 1 - 1e-6)
test_eval["err_pct"] = (test_eval["y_pred"] - test_eval["y_true"]).abs()
test_eval["err_$"]   = test_eval["err_pct"] * SALARY_CAP_REF

def segment(pct):
    if pct < 0.05:  return "1_low (<5%)"
    if pct < 0.10:  return "2_mid-low (5-10%)"
    if pct < 0.20:  return "3_mid (10-20%)"
    return "4_high (>20%)"

test_eval["segment"] = test_eval["y_true"].apply(segment)

print(f"  {'Segment':<22} {'N':>5}  {'MAE ($)':>12}  {'MAE (%)':>10}")
print(f"  {'-'*22}  {'-'*5}  {'-'*12}  {'-'*10}")
for seg, g in test_eval.groupby("segment"):
    print(f"  {seg:<22} {len(g):>5}  {g['err_$'].mean():>12,.0f}$  {g['err_pct'].mean():>9.2%}")
print(f"  {'TOTAL':<22} {len(test_eval):>5}  "
      f"{test_eval['err_$'].mean():>12,.0f}$  {test_eval['err_pct'].mean():>9.2%}")

# ═════════════════════════════════════════════════════════════════════════════
# 4. Analyse par POSITION
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "-" * 70)
print("4. ANALYSE PAR POSITION")
print("-" * 70)

print(f"\n  {'Position':<12} {'N':>5}  {'MAE ($)':>12}  {'Biais':>14}")
print(f"  {'-'*12}  {'-'*5}  {'-'*12}  {'-'*14}")
for pos, g in test_eval.groupby("position"):
    biais = (g["y_pred"] - g["y_true"]).mean() * SALARY_CAP_REF
    signe = "surevalue" if biais > 0 else "sous-evalue"
    print(f"  {str(pos):<12} {len(g):>5}  {g['err_$'].mean():>12,.0f}$  "
          f"  {biais:>+10,.0f}$ ({signe})")

# ═════════════════════════════════════════════════════════════════════════════
# 5. TOP 20 PIRES PRÉDICTIONS
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "-" * 70)
print("5. TOP 20 PIRES PREDICTIONS")
print("-" * 70)

worst = test_eval.nlargest(20, "err_$")[
    ["player_id", "saison", "position", "y_true", "y_pred", "err_$"]
].copy()
worst["reel_%"]  = worst["y_true"].map("{:.1%}".format)
worst["pred_%"]  = worst["y_pred"].map("{:.1%}".format)
worst["err"]     = worst["err_$"].map("${:,.0f}".format)
worst["sens"]    = (worst["y_pred"] - worst["y_true"]).apply(
    lambda x: "SURVAL" if x > 0 else "SOUVAL"
)
print(worst[["player_id", "position", "reel_%", "pred_%", "err", "sens"]].to_string(index=False))

# ═════════════════════════════════════════════════════════════════════════════
# 6. TABLEAU DE SYNTHÈSE
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TABLEAU DE SYNTHESE FINAL")
print("=" * 70)

rows = []
for name, maes in loso_results.items():
    rows.append({"Modele": name, "MAE LOSO ($)": np.mean([m for _, m in maes])})
rows.append({"Modele": "GLM actuel v2 [OOS 2024-25]", "MAE LOSO ($)": 4_686_079})

summary = pd.DataFrame(rows).sort_values("MAE LOSO ($)")
summary["MAE LOSO ($)"] = summary["MAE LOSO ($)"].map("${:,.0f}".format)
print(summary.to_string(index=False))

print("\n" + "-" * 70)
best_name = pd.DataFrame([(n, np.mean([m for _, m in v])) for n, v in loso_results.items()],
                         columns=["n", "m"]).sort_values("m").iloc[0]["n"]
print(f"  * Meilleur modele LOSO   : {best_name}")
print(f"  * Meilleur alpha Ridge   : {best_alpha}  ->  {alpha_results[best_alpha]:,.0f}$")
print("-" * 70)
print("\nFin de l'experimentation.\n")
