# Composant 3 — Valorisation : résultats de modélisation

Résultats du backtest walk-forward (3 plis, saisons test 2022 / 2023 / 2024).
Note : `season_year` = 4 premiers caractères de `saison` → à confirmer selon le
format exact de la colonne (`print(df["saison"].unique())`), mais si `saison`
suit la convention `"2022-23"`, alors `season_year = 2022` correspond à la
**saison 2022-2023**.

## Choix du modèle

| Modèle | MAE 2022 | MAE 2023 | MAE 2024 | Verdict |
|---|---|---|---|---|
| **Ridge** | 0.0143 | 0.0133 | 0.0133 | ✅ **Retenu — production** |
| Elastic Net | 0.0148 | 0.0135 | 0.0134 | Écarté — quasi-identique à Ridge, `l1_ratio` optimal proche de 0 |
| LightGBM | 0.0156 | 0.0142 | 0.0146 | Écarté — moins bon que Ridge sur les 3 plis |
| mean_reversion (baseline) | 0.0154 | 0.0144 | 0.0154 | Battu par Ridge |
| last_year (baseline) | 0.0156 | 0.0154 | 0.0162 | Battu par Ridge |
| age_curve (baseline) | 0.399 | 0.385 | 0.368 | Hors-jeu, inutilisable |

**Conclusion.** Le dataset est trop linéaire et trop petit (~400 lignes/pli)
pour que le gradient boosting apporte un gain mesurable. La pénalité L1
d'Elastic Net n'apporte rien non plus (`l1_ratio` optimal ≈ 0.1-0.3,
proche d'un Ridge pur). **Ridge est le modèle de production.**

Meilleur alpha trouvé par pli (GridSearchCV, TimeSeriesSplit) : 100-200
selon le pli, stable dans le temps → pas de sur-apprentissage visible.

## Incertitude

| Méthode | Coverage visé 80% | Width moyenne | Verdict |
|---|---|---|---|
| **Régression quantile** (q10/q90) | 79.6 % – 83.7 % (bien calibré) | ~0.063 – 0.068 | ✅ **Retenue — production** |
| Bootstrap (200 resamples, Ridge alpha=200) | 16.2 % – 19.2 % (sous-calibré) | ~0.006 – 0.008 | ❌ Écarté |

**Conclusion.** Le bootstrap échoue car un Ridge fortement régularisé produit
des coefficients très stables d'un resample à l'autre : il capture
l'incertitude des paramètres mais pas l'incertitude résiduelle (le bruit
irréductible du modèle). Un bootstrap résiduel corrigerait ça, mais l'effort
n'est pas justifié — la régression quantile est déjà bien calibrée et bien
plus simple. **Régression quantile (q10/q90) retenue.**

## État du Composant 3

- [x] Baselines (last_year, age_curve, mean_reversion)
- [x] Walk-forward backtest
- [x] Ridge — modèle de production
- [x] Elastic Net / LightGBM — testés, écartés
- [x] Incertitude — régression quantile
- [ ] Conversion production → dollars
- [ ] SHAP pour l'explication
- [ ] Clarifier la construction de `training_pairs`