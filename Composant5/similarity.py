"""Calcul des scores de similarité vectorielle et métriques de distance entre profils de joueurs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional
"""Moteur de recherche de joueurs similaires par similarité cosinus."""

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sqlalchemy import text
from typing import Optional
from dataclasses import dataclass, field

from .feature_store import (
    get_feature_matrix,
    get_feature_vector,
    load_scalers,
    SIMILARITY_FEATURES,
    SCALER_PATH,
)

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

DEFAULT_K = 10
DEFAULT_AGE_WINDOW = 2


# ------------------------------------------------------------------
# Structures de résultat
# ------------------------------------------------------------------

@dataclass
class ComparableResult:
    player_id: int
    player_name: str
    season: str
    age: float
    valuation_tier: str
    distance: float                        # 0 = identique, 1 = opposé
    similarity_pct: float                  # (1 - distance) * 100
    top_shared_features: list[str]         # features qui les rapprochent
    top_diverging_features: list[str]      # features qui les séparent


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _apply_weights(
    matrix: np.ndarray,
    feature_names: list[str],
    weights: Optional[dict[str, float]],
) -> np.ndarray:
    """
    Multiplie chaque colonne par son poids.
    Les features absentes du dict weights gardent un poids de 1.0.
    """
    if not weights:
        return matrix

    weight_vector = np.array([
        weights.get(feat, 1.0) for feat in feature_names
    ])
    return matrix * weight_vector


def _explain_similarity(
    target_vec: np.ndarray,
    candidate_vec: np.ndarray,
    feature_names: list[str],
    top_n: int = 3,
) -> tuple[list[str], list[str]]:
    """
    Retourne les features qui rapprochent et éloignent deux joueurs.
    Basé sur la différence absolue par feature.
    """
    diffs = np.abs(target_vec - candidate_vec)

    # Features les plus proches = différence minimale
    shared_idx = np.argsort(diffs)[:top_n]
    # Features les plus divergentes = différence maximale
    diverging_idx = np.argsort(diffs)[-top_n:][::-1]

    shared = [feature_names[i] for i in shared_idx]
    diverging = [feature_names[i] for i in diverging_idx]

    return shared, diverging


def _load_player_names(engine) -> dict[int, str]:
    """Charge le mapping player_id → player_name depuis la DB."""
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT player_id, player_name FROM players")
        ).fetchall()
    return {row.player_id: row.player_name for row in rows}


# ------------------------------------------------------------------
# Moteur principal
# ------------------------------------------------------------------

def _build_index(
    feature_matrix: pd.DataFrame,
    weights: Optional[dict[str, float]] = None,
) -> tuple[NearestNeighbors, np.ndarray]:
    """
    Construit l'index kNN sur le pool de comparables.
    Retourne (modèle entraîné, matrice numpy utilisée).
    """
    matrix = feature_matrix[SIMILARITY_FEATURES].values
    matrix = _apply_weights(matrix, SIMILARITY_FEATURES, weights)

    model = NearestNeighbors(
        n_neighbors=min(DEFAULT_K + 1, len(matrix)),
        metric="cosine",
        algorithm="brute",   # suffisant pour < 5k lignes
    )
    model.fit(matrix)
    return model, matrix


def _build_results(
    target_vec: np.ndarray,
    pool_df: pd.DataFrame,
    distances: np.ndarray,
    indices: np.ndarray,
    player_names: dict[int, str],
    exclude: tuple[int, str],
    k: int,
) -> list[ComparableResult]:
    """
    Construit la liste de ComparableResult depuis les sorties kNN.
    Exclut le joueur-saison exact de la cible.
    """
    results = []

    for dist, idx in zip(distances[0], indices[0]):
        row = pool_df.iloc[idx]
        pid = int(row["player_id"])
        season = row["saison"]

        # Exclure le joueur-saison exact
        if (pid, season) == exclude:
            continue

        candidate_vec = pool_df.iloc[idx][SIMILARITY_FEATURES].values
        shared, diverging = _explain_similarity(
            target_vec, candidate_vec, SIMILARITY_FEATURES
        )

        results.append(ComparableResult(
            player_id=pid,
            player_name=player_names.get(pid, f"player_{pid}"),
            season=season,
            age=float(row["age_at_date"]),
            valuation_tier=row["valuation_tier"],
            distance=round(float(dist), 4),
            similarity_pct=round((1 - float(dist)) * 100, 2),
            top_shared_features=shared,
            top_diverging_features=diverging,
        ))

        if len(results) == k:
            break

    return results


# ------------------------------------------------------------------
# Interface publique
# ------------------------------------------------------------------

def find_similar_players(
    engine,
    player_id: int,
    season: str,
    k: int = DEFAULT_K,
    weights: Optional[dict[str, float]] = None,
    scalers: Optional[dict] = None,
) -> list[ComparableResult]:
    """
    Trouve les k joueurs les plus similaires à (player_id, season)
    dans le pool comparable (reliable + replacement_level).

    Le joueur cible peut être low_sample — il est positionné dans
    l'espace mais les résultats viennent uniquement du pool comparable.
    """
    if scalers is None:
        scalers = load_scalers()

    # Vecteur cible (tous tiers)
    target_row = get_feature_vector(engine, player_id, season, scalers=scalers)
    if target_row is None:
        print(f"[similarity] joueur {player_id} / {season} introuvable.")
        return []

    target_vec = target_row[SIMILARITY_FEATURES].values

    # Pool de comparables (sans low_sample)
    pool_df = get_feature_matrix(
        engine,
        comparable_pool_only=True,
        scalers=scalers,
    ).reset_index(drop=True)

    player_names = _load_player_names(engine)

    # Application des poids
    target_weighted = _apply_weights(
        target_vec.reshape(1, -1), SIMILARITY_FEATURES, weights
    )

    model, _ = _build_index(pool_df, weights=weights)
    distances, indices = model.kneighbors(target_weighted)

    return _build_results(
        target_vec=target_vec,
        pool_df=pool_df,
        distances=distances,
        indices=indices,
        player_names=player_names,
        exclude=(player_id, season),
        k=k,
    )


def find_historical_comparables(
    engine,
    player_id: int,
    season: str,
    age_window: int = DEFAULT_AGE_WINDOW,
    k: int = 5,
    weights: Optional[dict[str, float]] = None,
    scalers: Optional[dict] = None,
) -> list[ComparableResult]:
    """
    Trouve les joueurs qui ressemblaient à (player_id, season)
    au même âge dans l'histoire.

    Filtre le pool sur age_at_date ∈ [age_cible - window, age_cible + window].
    """
    if scalers is None:
        scalers = load_scalers()

    target_row = get_feature_vector(engine, player_id, season, scalers=scalers)
    if target_row is None:
        print(f"[similarity] joueur {player_id} / {season} introuvable.")
        return []

    target_age = float(target_row["age_at_date"])
    target_vec = target_row[SIMILARITY_FEATURES].values

    # Pool historique filtré par âge (toutes saisons)
    pool_df = get_feature_matrix(
        engine,
        season=None,              # toutes les saisons
        comparable_pool_only=True,
        scalers=scalers,
    ).reset_index(drop=True)

    pool_df = pool_df[
        (pool_df["age_at_date"] >= target_age - age_window) &
        (pool_df["age_at_date"] <= target_age + age_window)
    ].reset_index(drop=True)

    if pool_df.empty:
        print(f"[similarity] aucun comparable dans la fenêtre d'âge {target_age} ± {age_window}")
        return []

    player_names = _load_player_names(engine)

    target_weighted = _apply_weights(
        target_vec.reshape(1, -1), SIMILARITY_FEATURES, weights
    )

    model, _ = _build_index(pool_df, weights=weights)
    distances, indices = model.kneighbors(target_weighted)

    return _build_results(
        target_vec=target_vec,
        pool_df=pool_df,
        distances=distances,
        indices=indices,
        player_names=player_names,
        exclude=(player_id, season),
        k=k,
    )