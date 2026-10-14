"""Classification et identification des rôles tactiques et archétypes de jeu des joueurs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional
"""Découvre et nomme les archétypes de joueurs par clustering HDBSCAN."""

import pandas as pd
import numpy as np
import hdbscan
import pickle
from pathlib import Path
from typing import Optional
from sqlalchemy import text
from config import DB_URL

from .feature_store import (
    get_feature_matrix,
    SIMILARITY_FEATURES,
)

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

MIN_CLUSTER_SIZE = 40      # taille minimale d'un archétype
MIN_SAMPLES = 5            # tolérance au bruit
OUTLIER_LABEL = -1
OUTLIER_ARCHETYPE = "versatile_hybrid"

CLUSTERER_PATH = Path("models/c5_clusterer.pkl")
ARCHETYPES_PATH = Path("models/c5_archetypes.pkl")

# Règles de nommage : (feature dominante, seuil relatif, nom partiel)
# Utilisées pour construire le nom automatiquement
NAMING_RULES = [
    ("pts_per100_shrunk",  0.5,  "scoring"),
    ("ast_per100_shrunk",  0.5,  "playmaking"),
    ("reb_per100_shrunk",  0.5,  "rebounding"),
    ("blk_per100_shrunk",  0.5,  "rim_protecting"),
    ("three_rate",         0.5,  "shooting"),
    ("stl_per100_shrunk",  0.5,  "perimeter_defending"),
    ("usg_pct",            0.5,  "high_usage"),
    ("oreb_pct",           0.5,  "crashing"),
]


# ------------------------------------------------------------------
# Clustering
# ------------------------------------------------------------------

def fit_clusterer(
    feature_matrix: np.ndarray,
    min_cluster_size: int = MIN_CLUSTER_SIZE,
    min_samples: int = MIN_SAMPLES,
) -> hdbscan.HDBSCAN:
    """
    Entraîne HDBSCAN sur la matrice de features.
    Utilise la distance euclidienne sur l'espace standardisé.
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",   # excess of mass : clusters stables
        prediction_data=True,             # nécessaire pour predict() sur nouveaux points
    )
    clusterer.fit(feature_matrix)
    return clusterer


def predict_archetype(
    clusterer: hdbscan.HDBSCAN,
    vector: np.ndarray,
) -> tuple[int, float]:
    """
    Prédit le cluster d'un nouveau vecteur.
    Retourne (label, probabilité d'appartenance).
    Les outliers reçoivent label=-1.
    """
    labels, probs = hdbscan.approximate_predict(
        clusterer, vector.reshape(1, -1)
    )
    return int(labels[0]), float(probs[0])


# ------------------------------------------------------------------
# Nommage des archétypes
# ------------------------------------------------------------------

def _name_cluster(
    cluster_mean: pd.Series,
    global_mean: pd.Series,
) -> str:
    """
    Construit un nom d'archétype en comparant la moyenne du cluster
    à la moyenne globale feature par feature.
    """
    tags = []
    for feature, threshold, tag in NAMING_RULES:
        if feature not in cluster_mean.index:
            continue
        # Le cluster est au-dessus de la moyenne globale de plus de threshold
        if cluster_mean[feature] - global_mean[feature] > threshold:
            tags.append(tag)

    if not tags:
        return "balanced"

    return "_".join(tags[:3])   # max 3 tags pour rester lisible


def build_archetype_names(
    df: pd.DataFrame,
    labels: np.ndarray,
) -> dict[int, str]:
    """
    Construit le mapping {label_int → nom_string} pour tous les clusters.
    Les outliers (label -1) reçoivent OUTLIER_ARCHETYPE.
    """
    df = df.copy()
    df["_cluster"] = labels

    global_mean = df[SIMILARITY_FEATURES].mean()
    archetype_names = {OUTLIER_LABEL: OUTLIER_ARCHETYPE}

    unique_labels = [l for l in np.unique(labels) if l != OUTLIER_LABEL]

    for label in unique_labels:
        cluster_df = df[df["_cluster"] == label]
        cluster_mean = cluster_df[SIMILARITY_FEATURES].mean()
        name = _name_cluster(cluster_mean, global_mean)
        # Évite les noms dupliqués en ajoutant le label
        archetype_names[label] = f"{name}_{label}"

    return archetype_names


# ------------------------------------------------------------------
# Persistence
# ------------------------------------------------------------------

def save_clusterer(clusterer: hdbscan.HDBSCAN, path: Path = CLUSTERER_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(clusterer, f)


def load_clusterer(path: Path = CLUSTERER_PATH) -> hdbscan.HDBSCAN:
    if not path.exists():
        raise FileNotFoundError(
            f"[role_engine] clusterer introuvable : {path}\n"
            "Lance build_and_save_roles() d'abord."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def save_archetypes(archetypes: dict, path: Path = ARCHETYPES_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(archetypes, f)


def load_archetypes(path: Path = ARCHETYPES_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"[role_engine] archetypes introuvables : {path}\n"
            "Lance build_and_save_roles() d'abord."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


# ------------------------------------------------------------------
# Écriture en base
# ------------------------------------------------------------------

def save_assignments_to_db(
    engine,
    df: pd.DataFrame,
    labels: np.ndarray,
    archetype_names: dict[int, str],
    probabilities: np.ndarray,
) -> None:
    """
    Persiste les assignments dans une table player_archetypes.
    Crée la table si elle n'existe pas.
    """
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS player_archetypes (
                player_id       INTEGER,
                saison          VARCHAR,
                cluster_label   INTEGER,
                archetype_name  VARCHAR,
                membership_prob FLOAT,
                PRIMARY KEY (player_id, saison)
            )
        """))

        rows = []
        for i, (_, row) in enumerate(df.iterrows()):
            rows.append({
                "player_id": int(row["player_id"]),
                "saison": row["saison"],
                "cluster_label": int(labels[i]),
                "archetype_name": archetype_names.get(int(labels[i]), "unknown"),
                "membership_prob": round(float(probabilities[i]), 4),
            })

        # Upsert
        conn.execute(text("""
            INSERT INTO player_archetypes
                (player_id, saison, cluster_label, archetype_name, membership_prob)
            VALUES
                (:player_id, :saison, :cluster_label, :archetype_name, :membership_prob)
            ON CONFLICT (player_id, saison) DO UPDATE SET
                cluster_label   = EXCLUDED.cluster_label,
                archetype_name  = EXCLUDED.archetype_name,
                membership_prob = EXCLUDED.membership_prob
        """), rows)
        conn.commit()

    print(f"[role_engine] ✓ {len(rows)} assignments écrits en base.")


# ------------------------------------------------------------------
# Interface publique
# ------------------------------------------------------------------

def build_and_save_roles(engine) -> dict:
    """
    Pipeline complet :
    1. Charge les features (pool comparable uniquement)
    2. Entraîne HDBSCAN
    3. Nomme les archétypes
    4. Sauvegarde clusterer + noms en fichier
    5. Persiste les assignments en base

    Retourne le mapping {label → nom}.
    """
    print("[role_engine] chargement des features...")
    df = get_feature_matrix(
        engine,
        comparable_pool_only=True,
    ).reset_index(drop=True)

    matrix = df[SIMILARITY_FEATURES].values

    print(f"[role_engine] clustering sur {len(df)} joueurs-saisons...")
    clusterer = fit_clusterer(matrix)

    labels = clusterer.labels_
    probabilities = clusterer.probabilities_

    n_clusters = len(set(labels)) - (1 if OUTLIER_LABEL in labels else 0)
    n_outliers = (labels == OUTLIER_LABEL).sum()
    print(f"[role_engine] {n_clusters} archétypes trouvés, {n_outliers} outliers")

    archetype_names = build_archetype_names(df, labels)
    print(f"[role_engine] archétypes : {archetype_names}")

    save_clusterer(clusterer)
    save_archetypes(archetype_names)
    save_assignments_to_db(engine, df, labels, archetype_names, probabilities)

    return archetype_names


def get_player_archetype(
    engine,
    player_id: int,
    season: str,
) -> Optional[dict]:
    """
    Retourne l'archétype d'un joueur-saison depuis la base.
    Retourne None si le joueur n'a pas été assigné.
    """
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT cluster_label, archetype_name, membership_prob
            FROM player_archetypes
            WHERE player_id = :pid AND saison = :saison
        """), {"pid": player_id, "saison": season}).fetchone()

    if row is None:
        return None

    return {
        "cluster_label": row.cluster_label,
        "archetype_name": row.archetype_name,
        "membership_prob": row.membership_prob,
    }


def get_archetype_players(
    engine,
    archetype_name: str,
    season: Optional[str] = None,
) -> pd.DataFrame:
    """
    Retourne tous les joueurs d'un archétype donné.
    Utile pour l'agent : 'montre moi tous les shooting_playmaker'.
    """
    query = """
        SELECT pa.player_id, p.player_name, pa.saison,
               pa.membership_prob
        FROM player_archetypes pa
        JOIN players p ON p.player_id = pa.player_id
        WHERE pa.archetype_name = :archetype
    """
    params = {"archetype": archetype_name}

    if season:
        query += " AND pa.saison = :saison"
        params["saison"] = season

    query += " ORDER BY pa.membership_prob DESC"

    with engine.connect() as conn:
        return pd.read_sql(text(query), conn, params=params)