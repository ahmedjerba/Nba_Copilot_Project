import sys
import os
from pathlib import Path
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r"c:\Users\user\Desktop\nba_copilot")

import pandas as pd
import numpy as np
from Composant3.Valorisation.data_loader import load_pairs
from Composant3.Valorisation.feature_engineering import build_features, CAT_FEATURES
from Composant3.Valorisation.model import BetaValuationModel
from Composant3.Valorisation.model_registry import save_model
from Composant3.Valorisation.config import SALARY_CAP, SALARY_CAP_REF
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

print("=" * 80)
print("COMPOSANT 3 : ENTRAÎNEMENT SUR 4 SAISONS & ÉVALUATION TOUS JOUEURS 2025-26")
print("=" * 80)

# ── 1. CHARGEMENT DU DATASET D'ENTRAÎNEMENT (MARKET - 4 SAISONS) ────────────
pairs_market = load_pairs(segment="market")

# 4 saisons historiques strictes d'entraînement
train_mask = pairs_market["saison"].isin(["2021-22", "2022-23", "2023-24", "2024-25"])
train_pairs = pairs_market[train_mask].copy()

print(f"\n[1] Dataset d'entraînement (market, 4 saisons) : {len(train_pairs)} paires")
print(train_pairs["saison"].value_counts().sort_index().to_string())

# ── 2. FEATURE ENGINEERING ÉPURÉ ───────────────────────────────────────────
train_fe, num_cols, cat_cols, _ = build_features(
    train_pairs,
    use_archetype=False,
    use_interactions=True,
    use_polyvalence=True,
    use_modern_nba=True,
    use_age_peak=True,
    use_scoring_eff=True,
    use_clean_pm=True,
    use_bankability=True,
)
print(f"\n[2] Features sélectionnées : {len(num_cols)} numériques + {len(cat_cols)} catégorielles")

# ── 3. ENTRAÎNEMENT DU MODÈLE DE MARCHÉ ────────────────────────────────────
print("\n[3] Entraînement de BetaValuationModel (GLM Binomial Logit + Ridge)...")
beta_model = BetaValuationModel()
beta_model.fit(
    train_fe,
    num_features=num_cols,
    cat_features=cat_cols,
    sample_weight_col="sample_weight",
    alpha=0.01
)

# Benchmark OOS sur la saison 2024-25
eval_2024 = train_fe[train_fe["saison"] == "2024-25"]
metrics_oos = beta_model.evaluate(eval_2024)
print(f"    MAE OOS 2024-25 : {metrics_oos['mae_dollars']:,.0f}$ (Gain vs baseline: {metrics_oos['gain_vs_baseline']:.1%})")

# Modèle complémentaire Gradient Boosting (pour validation croisée)
pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
])
gb_model = Pipeline([
    ("prep", pre),
    ("model", GradientBoostingRegressor(random_state=42, n_estimators=160, max_depth=3, learning_rate=0.04))
])
gb_model.fit(train_fe[num_cols + cat_cols], train_fe["cap_hit_pct"], model__sample_weight=train_fe["sample_weight"])
preds_gb_2024 = gb_model.predict(eval_2024[num_cols + cat_cols])
mae_gb = mean_absolute_error(eval_2024["cap_hit_pct"], preds_gb_2024) * SALARY_CAP_REF
print(f"    GradBoost MAE OOS 2024-25 : {mae_gb:,.0f}$")

# ── 4. SAUVEGARDE DU MODÈLE DE PRODUCTION ─────────────────────────────────
fe_flags = {
    "use_archetype": False,
    "use_interactions": True,
    "use_polyvalence": True,
    "use_modern_nba": True,
    "use_age_peak": True,
    "use_scoring_eff": True,
    "use_clean_pm": True,
    "use_bankability": True,
}
pkl_path = save_model(
    model=beta_model,
    fe_flags=fe_flags,
    num_features=num_cols,
    mae_oos=mae_gb,
    version="market_v2"
)
print(f"\n[4] Modèle enregistré avec succès : {pkl_path}")

# ── 5. CHARGEMENT ET ÉVALUATION SUR TOUS LES JOUEURS 2025-26 ───────────────
pairs_all = load_pairs(segment="all")
target_pairs = pairs_all[pairs_all["saison"] == "2025-26"].copy()
print(f"\n[5] Évaluation de TOUS les joueurs pour 2025-26 : {len(target_pairs)} joueurs (Rookies + Superstars + Vétérans)")

# Feature engineering sur la saison cible
target_fe, _, _, _ = build_features(
    target_pairs,
    use_archetype=False,
    use_interactions=True,
    use_polyvalence=True,
    use_modern_nba=True,
    use_age_peak=True,
    use_scoring_eff=True,
    use_clean_pm=True,
    use_bankability=True,
)

# Prédiction via le modèle de marché performant (GradBoost)
cap_2025 = SALARY_CAP["2025-26"]
target_fe["pred_pct"] = gb_model.predict(target_fe[num_cols + cat_cols])
target_fe["pred_value"] = target_fe["pred_pct"] * cap_2025
target_fe["cap_hit_real"] = target_fe["cap_hit"]
target_fe["delta_dollars"] = target_fe["pred_value"] - target_fe["cap_hit_real"]

cols_display = [
    "player_name", "position", "team_name", "age_at_date", "saisons_experience",
    "cap_hit_real", "pred_value", "delta_dollars"
]

# ── 6. EXTRACTION DU TOP 15 SOUS-ÉVALUÉS & SUR-ÉVALUÉS ────────────────────
print("\n" + "=" * 95)
print("TOP 15 JOUEURS LES PLUS SOUS-ÉVALUÉS (BARGAINS / BONNES AFFAIRES POUR LEUR CLUB)")
print("Juste valeur marchande largement supérieure au salaire versé")
print("=" * 95)
top_undervalued = target_fe.sort_values("delta_dollars", ascending=False).head(15)[cols_display].copy()
top_undervalued["cap_hit_real"]   = top_undervalued["cap_hit_real"].apply(lambda x: f"{x:,.0f} $")
top_undervalued["pred_value"]     = top_undervalued["pred_value"].apply(lambda x: f"{x:,.0f} $")
top_undervalued["delta_dollars"]  = top_undervalued["delta_dollars"].apply(lambda x: f"+{x:,.0f} $")
print(top_undervalued.to_string(index=False))

print("\n" + "=" * 95)
print("TOP 15 JOUEURS LES PLUS SUR-ÉVALUÉS (OVERPAID / MAUVAIS CONTRATS)")
print("Salaire réel largement supérieur à leur production sportive marchande")
print("=" * 95)
top_overvalued = target_fe.sort_values("delta_dollars", ascending=True).head(15)[cols_display].copy()
top_overvalued["cap_hit_real"]   = top_overvalued["cap_hit_real"].apply(lambda x: f"{x:,.0f} $")
top_overvalued["pred_value"]     = top_overvalued["pred_value"].apply(lambda x: f"{x:,.0f} $")
top_overvalued["delta_dollars"]  = top_overvalued["delta_dollars"].apply(lambda x: f"{x:,.0f} $")
print(top_overvalued.to_string(index=False))

# Sauvegarde des résultats complets en CSV pour exploitation
output_csv = Path(__file__).parent / "valorisations_2025_26_all.csv"
target_fe.sort_values("delta_dollars", ascending=False).to_csv(output_csv, index=False)
print(f"\n[OK] Rapport complet exporté dans : {output_csv}")
