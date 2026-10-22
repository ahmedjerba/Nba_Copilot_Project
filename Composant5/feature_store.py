"""Extraction, chargement et mise en cache des features de joueurs pour le calcul de comparables."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional
"""Charge et standardise les features de player_features pour C5."""

import pandas as pd
from sklearn.preprocessing import RobustScaler
from sqlalchemy import text
import pickle
from pathlib import Path
from typing import Optional

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

SIMILARITY_FEATURES = [
    "pts_per100_shrunk",
    "ast_per100_shrunk",
    "reb_per100_shrunk",
    "stl_per100_shrunk",
    "blk_per100_shrunk",
    "ts_pct",
    "efg_pct",
    "usg_pct",
    "three_rate",
    "free_throw_rate",
    "ast_pct",
    "oreb_pct",
    "dreb_pct",
    "trend_pts",
    "trend_pie",
    "pie",
    "net_rating",
]

VALID_TIERS = ["reliable", "replacement_level", "low_sample"]
COMPARABLE_TIERS = ["reliable", "replacement_level"]
SCALER_PATH = Path("models/c5_scalers.pkl")


# ------------------------------------------------------------------
# Chargement
# ------------------------------------------------------------------

def load_features(
    engine,
    season: Optional[str] = None,
) -> pd.DataFrame:
    """
    Charge player_features depuis la DB sans filtre de possessions.
    Le filtrage se fait par tier en aval.
    """
    cols = (
        ["player_id", "saison", "age_at_date", "valuation_tier"]
        + SIMILARITY_FEATURES
    )
    cols_sql = ", ".join(cols)

    query = f"SELECT {cols_sql} FROM player_features WHERE 1=1"
    params = {}

    if season:
        query += " AND saison = :saison"
        params["saison"] = season

    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn, params=params)

    # Vérifie qu'on ne reçoit que des tiers connus
    unknown = set(df["valuation_tier"].unique()) - set(VALID_TIERS)
    if unknown:
        print(f"[feature_store] ⚠️ tiers inconnus détectés : {unknown}")

    return df


# ------------------------------------------------------------------
# Standardisation (Scaler Global Unifié - C5-2 Isométrie)
# ------------------------------------------------------------------

def fit_scalers(df: pd.DataFrame) -> RobustScaler:
    """
    Entraîne un RobustScaler global unique sur l'ensemble de la population
    pour préserver l'isométrie et les distances réelles entre tous les joueurs.
    """
    scaler = RobustScaler()
    scaler.fit(df[SIMILARITY_FEATURES].fillna(0))
    print(f"[feature_store] scaler global unifié entraîné sur {len(df)} lignes.")
    return scaler


def save_scalers(scaler: Any, path: Path = SCALER_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scaler, f)


def load_scalers(path: Path = SCALER_PATH) -> Any:
    if not path.exists():
        raise FileNotFoundError(
            f"[feature_store] scalers introuvables : {path}\n"
            "Lance build_and_save_scalers() d'abord."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def apply_scalers(
    df: pd.DataFrame,
    scalers: Any,
) -> pd.DataFrame:
    """
    Applique le scaler global unifié à toutes les lignes (ou par tier si dictionnaire legacy).
    """
    if isinstance(scalers, RobustScaler):
        scaled = df.copy()
        scaled[SIMILARITY_FEATURES] = scalers.transform(df[SIMILARITY_FEATURES].fillna(0))
        return scaled

    # Support de repli si dict
    scaled_parts = []
    for tier, group in df.groupby("valuation_tier"):
        if tier not in scalers:
            print(f"[feature_store] ⚠️ tier sans scaler ignoré : {tier}")
            continue
        scaled = group.copy()
        scaled[SIMILARITY_FEATURES] = scalers[tier].transform(
            group[SIMILARITY_FEATURES].fillna(0)
        )
        scaled_parts.append(scaled)

    if not scaled_parts:
        raise ValueError("[feature_store] Aucune ligne après application des scalers.")

    return pd.concat(scaled_parts).sort_index()


# ------------------------------------------------------------------
# Interface principale
# ------------------------------------------------------------------

def get_feature_matrix(
    engine,
    season: Optional[str] = None,
    comparable_pool_only: bool = False,
    scalers: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Point d'entrée principal pour similarity.py et embedder.py.

    comparable_pool_only=True  → exclut low_sample du résultat.
                                 Utilisé pour construire le pool de recherche.
    comparable_pool_only=False → retourne tous les tiers.
                                 Utilisé quand le joueur cible est low_sample.
    """
    df = load_features(engine, season=season)

    if scalers is None:
        scalers = load_scalers()

    df_scaled = apply_scalers(df, scalers)

    if comparable_pool_only:
        df_scaled = df_scaled[
            df_scaled["valuation_tier"].isin(COMPARABLE_TIERS)
        ].copy()

    return df_scaled


def get_feature_vector(
    engine,
    player_id: int,
    season: str,
    scalers: Optional[dict] = None,
) -> Optional[pd.Series]:
    """
    Retourne le vecteur standardisé d'un joueur-saison spécifique.
    Retourne None si le joueur est introuvable ou sous le seuil.
    Inclut low_sample — le joueur cible peut être n'importe quel tier.
    """
    if scalers is None:
        scalers = load_scalers()

    df = get_feature_matrix(
        engine,
        season=season,
        comparable_pool_only=False,
        scalers=scalers,
    )

    row = df[
        (df["player_id"] == player_id) &
        (df["saison"] == season)
    ]

    if row.empty:
        print(
            f"[feature_store] joueur {player_id} / saison {season} "
            "introuvable ou low_sample exclu."
        )
        return None

    return row.iloc[0]


# ------------------------------------------------------------------
# Construction initiale des scalers
# ------------------------------------------------------------------

def build_and_save_scalers(engine) -> dict:
    """
    À appeler une fois après chaque mise à jour majeure des données.
    Entraîne les scalers sur toutes les saisons et tous les tiers,
    y compris low_sample, pour pouvoir positionner n'importe quel joueur.
    """
    print("[feature_store] chargement de toutes les saisons...")
    df = load_features(engine, season=None)

    print(f"[feature_store] {len(df)} lignes chargées")
    print(f"[feature_store] distribution des tiers :\n{df['valuation_tier'].value_counts()}")

    missing_tiers = set(VALID_TIERS) - set(df["valuation_tier"].unique())
    if missing_tiers:
        print(f"[feature_store] ⚠️ tiers absents des données : {missing_tiers}")

    scalers = fit_scalers(df)
    save_scalers(scalers)

    print(f"[feature_store] ✓ scalers sauvegardés dans {SCALER_PATH}")
    return scalers