"""Moteur de recherche de joueurs similaires par similarité cosinus."""

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sqlalchemy import text
from typing import Optional
from dataclasses import dataclass

# ──────────────────────────────────────────────────────────────────
# Config — doit être identique à role_engine.py
# ──────────────────────────────────────────────────────────────────

SIMILARITY_FEATURES = [
    "pts_per100_shrunk",
    "ast_per100_shrunk",
    "stl_per100_shrunk",
    "blk_per100_shrunk",
    "ts_pct",
    "usg_pct",
    "three_rate",
    "free_throw_rate",
    "oreb_pct",
    "dreb_pct",
    "trend_pts",
    "trend_pie",
    "pie",
    "net_rating",
]

COMPARABLE_TIERS   = ["reliable", "replacement_level"]
DEFAULT_K          = 10
DEFAULT_AGE_WINDOW = 2


# ──────────────────────────────────────────────────────────────────
# Structures de résultat
# ──────────────────────────────────────────────────────────────────

@dataclass
class ComparableResult:
    player_id             : int
    player_name           : str
    season                : str
    age                   : float
    valuation_tier        : str
    archetype             : str        # ← archetype_name → archetype
    archetype_conf        : float      # ← nouveau
    archetype_secondary   : str        # ← nouveau
    distance              : float
    similarity_pct        : float
    top_shared_features   : list[str]
    top_diverging_features: list[str]


# ──────────────────────────────────────────────────────────────────
# Chargement des données
# ──────────────────────────────────────────────────────────────────

def _load_pool(
    engine,
    season  : Optional[str]   = None,
    age_min : Optional[float] = None,
    age_max : Optional[float] = None,
) -> pd.DataFrame:
    cols_sql = ", ".join([f"pf.{f}" for f in SIMILARITY_FEATURES])

    query = f"""
        SELECT pf.player_id, pf.saison, pf.valuation_tier,
               pf.age_at_date, {cols_sql},
               COALESCE(pa.archetype, 'unclassified')      AS archetype,
               COALESCE(pa.archetype_conf, 0.0)            AS archetype_conf,
               pa.archetype_secondary
        FROM player_features pf
        LEFT JOIN player_archetypes pa
               ON pa.player_id = pf.player_id AND pa.saison = pf.saison
        WHERE pf.valuation_tier IN ('reliable', 'replacement_level')
    """
    params: dict = {}

    if season:
        query += " AND pf.saison = :saison"
        params["saison"] = season

    if age_min is not None:
        query += " AND pf.age_at_date >= :age_min"
        params["age_min"] = age_min

    if age_max is not None:
        query += " AND pf.age_at_date <= :age_max"
        params["age_max"] = age_max

    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn, params=params)

    return df.reset_index(drop=True)


def _load_target(
    engine   ,
    player_id: int,
    season   : str,
) -> Optional[pd.Series]:
    cols_sql = ", ".join([f"pf.{f}" for f in SIMILARITY_FEATURES])

    query = f"""
        SELECT pf.player_id, pf.saison, pf.valuation_tier,
               pf.age_at_date, {cols_sql},
               COALESCE(pa.archetype, 'unclassified')      AS archetype,
               COALESCE(pa.archetype_conf, 0.0)            AS archetype_conf,
               pa.archetype_secondary
        FROM player_features pf
        LEFT JOIN player_archetypes pa
               ON pa.player_id = pf.player_id AND pa.saison = pf.saison
        WHERE pf.player_id = :pid AND pf.saison = :saison
    """
    with engine.connect() as conn:
        df = pd.read_sql(
            text(query), conn,
            params={"pid": player_id, "saison": season},
        )

    if df.empty:
        return None
    return df.iloc[0]


def _load_player_names(engine) -> dict[int, str]:
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT player_id, player_name FROM players")
        ).fetchall()
    return {row.player_id: row.player_name for row in rows}


# ──────────────────────────────────────────────────────────────────
# Standardisation
# ──────────────────────────────────────────────────────────────────

def _standardize(
    matrix     : np.ndarray,
    pool_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.preprocessing import RobustScaler
    scaler        = RobustScaler()
    pool_scaled   = scaler.fit_transform(pool_matrix)
    target_scaled = scaler.transform(matrix)
    return target_scaled, pool_scaled


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def _apply_weights(
    matrix       : np.ndarray,
    feature_names: list[str],
    weights      : Optional[dict[str, float]],
) -> np.ndarray:
    if not weights:
        return matrix
    weight_vector = np.array([weights.get(f, 1.0) for f in feature_names])
    return matrix * weight_vector


def _explain_similarity(
    target_vec   : np.ndarray,
    candidate_vec: np.ndarray,
    feature_names: list[str],
    top_n        : int = 3,
) -> tuple[list[str], list[str]]:
    # Normalisation L2 pour aligner les écarts par coordonnée sur la distance cosinus
    norm_t = np.linalg.norm(target_vec)
    norm_c = np.linalg.norm(candidate_vec)
    u_norm = target_vec / norm_t if norm_t > 0 else target_vec
    v_norm = candidate_vec / norm_c if norm_c > 0 else candidate_vec

    diffs         = np.abs(u_norm - v_norm)
    shared_idx    = np.argsort(diffs)[:top_n]
    diverging_idx = np.argsort(diffs)[-top_n:][::-1]
    return (
        [feature_names[i] for i in shared_idx],
        [feature_names[i] for i in diverging_idx],
    )


def _build_index(
    pool_matrix: np.ndarray,
    k          : int,
    weights    : Optional[dict[str, float]] = None,
) -> NearestNeighbors:
    matrix = _apply_weights(pool_matrix, SIMILARITY_FEATURES, weights)
    model  = NearestNeighbors(
        n_neighbors=min(k + 1, len(matrix)),
        metric="cosine",
        algorithm="brute",
    )
    model.fit(matrix)
    return model


def _build_results(
    target_vec  : np.ndarray,
    pool_df     : pd.DataFrame,
    pool_scaled : np.ndarray,
    distances   : np.ndarray,
    indices     : np.ndarray,
    player_names: dict[int, str],
    exclude     : tuple[int, str],
    k           : int,
) -> list[ComparableResult]:
    results = []

    for dist, idx in zip(distances[0], indices[0]):
        row    = pool_df.iloc[idx]
        pid    = int(row["player_id"])
        season = row["saison"]

        if (pid, season) == exclude:
            continue

        candidate_vec = pool_scaled[idx]
        shared, diverging = _explain_similarity(
            target_vec, candidate_vec, SIMILARITY_FEATURES
        )

        results.append(ComparableResult(
            player_id              = pid,
            player_name            = player_names.get(pid, f"player_{pid}"),
            season                 = season,
            age                    = float(row["age_at_date"]),
            valuation_tier         = row["valuation_tier"],
            archetype              = row.get("archetype", "unclassified"),
            archetype_conf         = float(row.get("archetype_conf", 0.0)),
            archetype_secondary    = row.get("archetype_secondary"),
            distance               = round(float(dist), 4),
            similarity_pct         = round((1 - float(dist)) * 100, 2),
            top_shared_features    = shared,
            top_diverging_features = diverging,
        ))

        if len(results) == k:
            break

    return results


# ──────────────────────────────────────────────────────────────────
# Interface publique
# ──────────────────────────────────────────────────────────────────

def find_similar_players(
    engine   ,
    player_id: int,
    season   : str,
    k        : int = DEFAULT_K,
    weights  : Optional[dict[str, float]] = None,
) -> list[ComparableResult]:
    """
    Trouve les k joueurs les plus similaires à (player_id, season)
    dans le pool comparable (reliable + replacement_level).
    Le joueur cible peut être n'importe quel tier.
    """
    target_row = _load_target(engine, player_id, season)
    if target_row is None:
        print(f"[similarity] joueur {player_id} / {season} introuvable.")
        return []

    pool_df = _load_pool(engine)
    if pool_df.empty:
        print("[similarity] pool vide.")
        return []

    player_names = _load_player_names(engine)

    target_raw = target_row[SIMILARITY_FEATURES].values.astype(float)
    pool_raw   = pool_df[SIMILARITY_FEATURES].values.astype(float)

    target_scaled_arr, pool_scaled = _standardize(
        np.nan_to_num(target_raw).reshape(1, -1),
        np.nan_to_num(pool_raw),
    )

    target_weighted    = _apply_weights(target_scaled_arr, SIMILARITY_FEATURES, weights)
    model              = _build_index(pool_scaled, k=k, weights=weights)
    distances, indices = model.kneighbors(target_weighted)

    return _build_results(
        target_vec   = target_scaled_arr[0],
        pool_df      = pool_df,
        pool_scaled  = pool_scaled,
        distances    = distances,
        indices      = indices,
        player_names = player_names,
        exclude      = (player_id, season),
        k            = k,
    )


def find_historical_comparables(
    engine    ,
    player_id : int,
    season    : str,
    age_window: int = DEFAULT_AGE_WINDOW,
    k         : int = 5,
    weights   : Optional[dict[str, float]] = None,
) -> list[ComparableResult]:
    """
    Trouve les joueurs qui ressemblaient à (player_id, season)
    au même âge dans l'histoire (toutes saisons confondues).
    Filtre le pool sur age_at_date ∈ [age_cible - window, age_cible + window].
    """
    target_row = _load_target(engine, player_id, season)
    if target_row is None:
        print(f"[similarity] joueur {player_id} / {season} introuvable.")
        return []

    target_age = float(target_row["age_at_date"])

    pool_df = _load_pool(
        engine,
        season  = None,
        age_min = target_age - age_window,
        age_max = target_age + age_window,
    )

    if pool_df.empty:
        print(
            f"[similarity] aucun comparable dans la fenêtre d'âge "
            f"{target_age} ± {age_window}."
        )
        return []

    player_names = _load_player_names(engine)

    target_raw = target_row[SIMILARITY_FEATURES].values.astype(float)
    pool_raw   = pool_df[SIMILARITY_FEATURES].values.astype(float)

    target_scaled_arr, pool_scaled = _standardize(
        np.nan_to_num(target_raw).reshape(1, -1),
        np.nan_to_num(pool_raw),
    )

    target_weighted    = _apply_weights(target_scaled_arr, SIMILARITY_FEATURES, weights)
    model              = _build_index(pool_scaled, k=k, weights=weights)
    distances, indices = model.kneighbors(target_weighted)

    return _build_results(
        target_vec   = target_scaled_arr[0],
        pool_df      = pool_df,
        pool_scaled  = pool_scaled,
        distances    = distances,
        indices      = indices,
        player_names = player_names,
        exclude      = (player_id, season),
        k            = k,
    )