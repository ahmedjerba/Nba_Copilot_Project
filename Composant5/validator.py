"""Valide que les comparables prédisent mieux la trajectoire qu'une baseline naïve."""

import pandas as pd
import numpy as np
from sqlalchemy import text
from dataclasses import dataclass
from typing import Optional

from similarity import find_historical_comparables

# ──────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────

TARGET_METRIC  = "pie"
HORIZONS       = [1, 2, 3]
K_COMPARABLES  = 5
MIN_POSS       = 0

SAISON_ORDER = [
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26",
]


# ──────────────────────────────────────────────────────────────────
# Structures de résultat
# ──────────────────────────────────────────────────────────────────

@dataclass
class HorizonResult:
    horizon        : int
    n_samples      : int
    mae_comparables: float
    mae_baseline   : float
    improvement    : float


@dataclass
class BacktestResult:
    horizons      : list[HorizonResult]
    target_metric : str
    k_comparables : int
    seasons_tested: list[str]


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def _saison_offset(saison: str, offset: int) -> Optional[str]:
    if saison not in SAISON_ORDER:
        return None
    idx = SAISON_ORDER.index(saison) + offset
    if idx >= len(SAISON_ORDER):
        return None
    return SAISON_ORDER[idx]


def _load_all_features(engine) -> pd.DataFrame:
    query = f"""
        SELECT player_id, saison, age_at_date, valuation_tier, {TARGET_METRIC}
        FROM player_features
        WHERE valuation_tier IN ('reliable', 'replacement_level')
        ORDER BY player_id, saison
    """
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn)


def _baseline_prediction(
    all_features : pd.DataFrame,
    saison_n     : str,
    saison_target: str,
    tier         : str,
) -> Optional[float]:
    subset = all_features[
        (all_features["saison"] == saison_target) &
        (all_features["valuation_tier"] == tier)
    ][TARGET_METRIC].dropna()

    if subset.empty:
        return None
    return float(subset.mean())


def _comparable_prediction(
    engine       : object,
    player_id    : int,
    saison_n     : str,
    saison_target: str,
    all_features : pd.DataFrame,
    h            : int = 1,
    k            : int = K_COMPARABLES,
) -> Optional[float]:
    comparables = find_historical_comparables(
        engine    = engine,
        player_id = player_id,
        season    = saison_n,
        age_window= 2,
        k         = k,
    )

    if not comparables:
        return None

    values = []
    for comp in comparables:
        comp_season_target = _saison_offset(comp.season, h)
        if not comp_season_target:
            continue

        comp_future = all_features[
            (all_features["player_id"] == comp.player_id) &
            (all_features["saison"] == comp_season_target)
        ][TARGET_METRIC]

        if not comp_future.empty:
            values.append(float(comp_future.iloc[0]))

    if not values:
        return None

    return float(np.mean(values))


# ──────────────────────────────────────────────────────────────────
# Backtest principal
# ──────────────────────────────────────────────────────────────────

def run_backtest(engine) -> BacktestResult:
    print("[validator] chargement des features...")
    all_features = _load_all_features(engine)
    print(f"[validator] {len(all_features)} joueurs-saisons chargés.")

    test_seasons = SAISON_ORDER[:-1]
    print(f"[validator] saisons testées : {test_seasons}")

    errors: dict[int, dict[str, list[float]]] = {
        h: {"comparables": [], "baseline": []} for h in HORIZONS
    }

    seasons_with_data = []

    for saison_n in test_seasons:
        print(f"\n[validator] saison N = {saison_n}")

        players_n = all_features[
            all_features["saison"] == saison_n
        ][["player_id", "valuation_tier", TARGET_METRIC]].dropna()

        print(f"[validator] {len(players_n)} joueurs à tester")

        for _, player_row in players_n.iterrows():
            pid  = int(player_row["player_id"])
            tier = player_row["valuation_tier"]

            for h in HORIZONS:
                saison_target = _saison_offset(saison_n, h)
                if saison_target is None:
                    continue

                true_future = all_features[
                    (all_features["player_id"] == pid) &
                    (all_features["saison"] == saison_target)
                ][TARGET_METRIC]

                if true_future.empty:
                    continue

                true_val = float(true_future.iloc[0])

                pred_comp = _comparable_prediction(
                    engine        = engine,
                    player_id     = pid,
                    saison_n      = saison_n,
                    saison_target = saison_target,
                    all_features  = all_features,
                    h             = h,
                    k             = K_COMPARABLES,
                )

                pred_base = _baseline_prediction(
                    all_features  = all_features,
                    saison_n      = saison_n,
                    saison_target = saison_target,
                    tier          = tier,
                )

                if pred_comp is not None:
                    errors[h]["comparables"].append(abs(true_val - pred_comp))

                if pred_base is not None:
                    errors[h]["baseline"].append(abs(true_val - pred_base))

        seasons_with_data.append(saison_n)

    horizon_results = []
    for h in HORIZONS:
        errs_comp = errors[h]["comparables"]
        errs_base = errors[h]["baseline"]

        if not errs_comp or not errs_base:
            print(f"[validator] ⚠️  horizon N+{h} : pas assez de données")
            continue

        mae_comp = float(np.mean(errs_comp))
        mae_base = float(np.mean(errs_base))
        improv   = (mae_base - mae_comp) / mae_base * 100

        horizon_results.append(HorizonResult(
            horizon         = h,
            n_samples       = len(errs_comp),
            mae_comparables = round(mae_comp, 4),
            mae_baseline    = round(mae_base, 4),
            improvement     = round(improv, 2),
        ))

        sign = "✅" if improv > 0 else "❌"
        print(
            f"\n[validator] N+{h} {sign} "
            f"MAE comparables={mae_comp:.4f} "
            f"vs baseline={mae_base:.4f} "
            f"→ {improv:+.1f}%"
        )

    return BacktestResult(
        horizons       = horizon_results,
        target_metric  = TARGET_METRIC,
        k_comparables  = K_COMPARABLES,
        seasons_tested = seasons_with_data,
    )


# ──────────────────────────────────────────────────────────────────
# Affichage des résultats
# ──────────────────────────────────────────────────────────────────

def print_backtest_report(result: BacktestResult) -> None:
    print("\n")
    print("=" * 60)
    print(f"BACKTEST VALIDATOR — {result.target_metric.upper()}")
    print(f"k={result.k_comparables} comparables")
    print(f"Saisons testées : {result.seasons_tested}")
    print("=" * 60)

    print(f"\n{'Horizon':<10} {'N':<8} {'MAE Comp':<12} {'MAE Base':<12} {'Amélio':<10} Verdict")
    print("-" * 60)

    for h in result.horizons:
        sign = "✅ meilleur" if h.improvement > 0 else "❌ moins bon"
        print(
            f"N+{h.horizon:<8} {h.n_samples:<8} "
            f"{h.mae_comparables:<12.4f} {h.mae_baseline:<12.4f} "
            f"{h.improvement:+.1f}%     {sign}"
        )

    print("-" * 60)

    if result.horizons:
        avg_improv = np.mean([h.improvement for h in result.horizons])
        print(f"\nAmélioration moyenne : {avg_improv:+.1f}%")
        if avg_improv > 0:
            print("→ Le système de comparables apporte de la valeur.")
        else:
            print("→ Le système ne bat pas la baseline — à investiguer.")


# ──────────────────────────────────────────────────────────────────
# Analyse par archétype
# ──────────────────────────────────────────────────────────────────

def run_backtest_by_archetype(engine) -> pd.DataFrame:
    """
    Même backtest mais ventilé par archétype.
    Permet d'identifier sur quels profils le système fonctionne bien.
    """
    print("[validator] backtest par archétype...")

    all_features = _load_all_features(engine)

    # ← archetype_name → archetype (nouveau schema)
    with engine.connect() as conn:
        archetypes_df = pd.read_sql(
            text("""
                SELECT player_id, saison,
                       archetype,
                       archetype_conf,
                       archetype_secondary,
                       is_soft_assigned,
                       is_reclassified
                FROM player_archetypes
            """),
            conn,
        )

    all_features = all_features.merge(
        archetypes_df, on=["player_id", "saison"], how="left"
    )
    # ← versatile_hybrid → unclassified (nouveau nom)
    all_features["archetype"] = all_features["archetype"].fillna("unclassified")

    test_seasons = SAISON_ORDER[:-1]

    rows = []

    for saison_n in test_seasons:
        players_n = all_features[
            all_features["saison"] == saison_n
        ][["player_id", "valuation_tier", "archetype",
           "archetype_conf", "is_soft_assigned", TARGET_METRIC]].dropna(
            subset=["player_id", TARGET_METRIC]
        )

        for _, player_row in players_n.iterrows():
            pid       = int(player_row["player_id"])
            tier      = player_row["valuation_tier"]
            archetype = player_row["archetype"]  # ← archetype_name → archetype

            for h in HORIZONS:
                saison_target = _saison_offset(saison_n, h)
                if saison_target is None:
                    continue

                true_future = all_features[
                    (all_features["player_id"] == pid) &
                    (all_features["saison"] == saison_target)
                ][TARGET_METRIC]

                if true_future.empty:
                    continue

                true_val = float(true_future.iloc[0])

                pred_comp = _comparable_prediction(
                    engine        = engine,
                    player_id     = pid,
                    saison_n      = saison_n,
                    saison_target = saison_target,
                    all_features  = all_features,
                    h             = h,
                    k             = K_COMPARABLES,
                )
                pred_base = _baseline_prediction(
                    all_features  = all_features,
                    saison_n      = saison_n,
                    saison_target = saison_target,
                    tier          = tier,
                )

                if pred_comp is not None and pred_base is not None:
                    rows.append({
                        "archetype"      : archetype,
                        "archetype_conf" : player_row.get("archetype_conf"),
                        "is_soft_assigned": player_row.get("is_soft_assigned"),
                        "horizon"        : h,
                        "err_comparables": abs(true_val - pred_comp),
                        "err_baseline"   : abs(true_val - pred_base),
                    })

    df = pd.DataFrame(rows)
    if df.empty:
        print("[validator] ⚠️  aucune donnée pour le backtest par archétype.")
        return df

    summary = (
        df.groupby(["archetype", "horizon"])
        .agg(
            n        = ("err_comparables", "count"),
            mae_comp = ("err_comparables", "mean"),
            mae_base = ("err_baseline", "mean"),
        )
        .reset_index()
    )
    summary["improvement"] = (
        (summary["mae_base"] - summary["mae_comp"]) / summary["mae_base"] * 100
    ).round(2)
    summary["mae_comp"] = summary["mae_comp"].round(4)
    summary["mae_base"] = summary["mae_base"].round(4)

    return summary.sort_values(["horizon", "improvement"], ascending=[True, False])


# ──────────────────────────────────────────────────────────────────
# Analyse soft assignment
# ──────────────────────────────────────────────────────────────────

def run_backtest_soft_vs_hard(engine) -> pd.DataFrame:
    """
    Compare la qualité des prédictions entre :
    - joueurs assignés directement par HDBSCAN (is_soft_assigned=False)
    - joueurs soft-assignés (is_soft_assigned=True)
    - joueurs reclassifiés (is_reclassified=True)

    Permet de vérifier que le soft assignment ne dégrade pas les scores.
    """
    print("[validator] backtest soft vs hard assignment...")

    all_features = _load_all_features(engine)

    with engine.connect() as conn:
        archetypes_df = pd.read_sql(
            text("""
                SELECT player_id, saison,
                       archetype, archetype_conf,
                       is_soft_assigned, is_reclassified
                FROM player_archetypes
            """),
            conn,
        )

    all_features = all_features.merge(
        archetypes_df, on=["player_id", "saison"], how="left"
    )
    all_features["archetype"]       = all_features["archetype"].fillna("unclassified")
    all_features["is_soft_assigned"]= all_features["is_soft_assigned"].fillna(False)
    all_features["is_reclassified"] = all_features["is_reclassified"].fillna(False)

    test_seasons = SAISON_ORDER[:-1]
    rows = []

    for saison_n in test_seasons:
        players_n = all_features[
            all_features["saison"] == saison_n
        ].dropna(subset=["player_id", TARGET_METRIC])

        for _, player_row in players_n.iterrows():
            pid  = int(player_row["player_id"])
            tier = player_row["valuation_tier"]

            # Catégorie d'assignment
            if player_row["is_reclassified"]:
                assignment_type = "reclassified"
            elif player_row["is_soft_assigned"]:
                assignment_type = "soft"
            else:
                assignment_type = "hard"

            for h in [1, 2]:  # N+1 et N+2 suffisent pour cette analyse
                saison_target = _saison_offset(saison_n, h)
                if saison_target is None:
                    continue

                true_future = all_features[
                    (all_features["player_id"] == pid) &
                    (all_features["saison"] == saison_target)
                ][TARGET_METRIC]

                if true_future.empty:
                    continue

                true_val  = float(true_future.iloc[0])
                pred_comp = _comparable_prediction(
                    engine        = engine,
                    player_id     = pid,
                    saison_n      = saison_n,
                    saison_target = saison_target,
                    all_features  = all_features,
                    h             = h,
                    k             = K_COMPARABLES,
                )
                pred_base = _baseline_prediction(
                    all_features  = all_features,
                    saison_n      = saison_n,
                    saison_target = saison_target,
                    tier          = tier,
                )

                if pred_comp is not None and pred_base is not None:
                    rows.append({
                        "assignment_type" : assignment_type,
                        "archetype_conf"  : player_row.get("archetype_conf"),
                        "horizon"         : h,
                        "err_comparables" : abs(true_val - pred_comp),
                        "err_baseline"    : abs(true_val - pred_base),
                    })

    df = pd.DataFrame(rows)
    if df.empty:
        print("[validator] ⚠️  aucune donnée.")
        return df

    summary = (
        df.groupby(["assignment_type", "horizon"])
        .agg(
            n        = ("err_comparables", "count"),
            mae_comp = ("err_comparables", "mean"),
            mae_base = ("err_baseline", "mean"),
        )
        .reset_index()
    )
    summary["improvement"] = (
        (summary["mae_base"] - summary["mae_comp"]) / summary["mae_base"] * 100
    ).round(2)
    summary["mae_comp"] = summary["mae_comp"].round(4)
    summary["mae_base"] = summary["mae_base"].round(4)

    return summary.sort_values(["horizon", "assignment_type"])