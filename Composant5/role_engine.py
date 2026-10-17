"""Découvre et nomme les archétypes de joueurs via UMAP + HDBSCAN."""

import pandas as pd
import numpy as np
import hdbscan
from hdbscan import all_points_membership_vectors
import pickle
from pathlib import Path
from typing import Optional
from sqlalchemy import text
from umap import UMAP
from sklearn.preprocessing import RobustScaler

# ──────────────────────────────────────────────────────────────────
# Config
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

UMAP_PARAMS = {
    "n_components" : 2,
    "n_neighbors"  : 20,       # ← 30 → 20
    "min_dist"     : 0.1,
    "metric"       : "cosine",
    "random_state" : 42,
}

HDBSCAN_PARAMS = {
    "min_cluster_size"        : 40,
    "min_samples"             : 3,  # ← 5 → 3
    "metric"                  : "euclidean",
    "cluster_selection_method": "eom",
    "prediction_data"         : True,
}

# ── Seuils soft assignment ─────────────────────────────────────────
# Pour les outliers HDBSCAN (label == -1)
SOFT_PRIMARY_THRESHOLD       = 0.10   # proba min pour recevoir un label
SOFT_SECONDARY_THRESHOLD     = 0.06   # proba min pour un label secondaire
# Pour les points assignés par HDBSCAN
ASSIGNED_CONF_MIN            = 0.15   # si conf < 0.15 → reclassification
ASSIGNED_SECONDARY_THRESHOLD = 0.15   # proba min pour un label secondaire

OUTLIER_LABEL     = -1
OUTLIER_ARCHETYPE = "unclassified"

ARCHETYPE_NAMES = {
    -1: "unclassified",
     0: "fringe_rotation",
     1: "replacement_level",
     2: "raw_prospect",
     3: "developmental_big",
     4: "rim_protector",
     5: "shot_blocker",
     6: "physical_big",
     7: "scoring_big",
     8: "declining_vet",
     9: "solid_rotation",
    10: "ascending_wing",
    11: "3and_d_wing",
    12: "superstar",
    13: "declining_star",
    14: "star_fading",
    15: "franchise_player",
    16: "3and_d_specialist",
    17: "spot_up_shooter",
}

UMAP_PATH      = Path("models/c5_umap.pkl")
CLUSTERER_PATH = Path("models/c5_clusterer.pkl")
SCALER_PATH    = Path("models/c5_role_scaler.pkl")


# ──────────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────────

def _save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _load(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"[role_engine] fichier introuvable : {path}\n"
            "Lance build_and_save_roles() d'abord."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


# ──────────────────────────────────────────────────────────────────
# Soft assignment
# ──────────────────────────────────────────────────────────────────

def _compute_soft_labels(
    labels       : np.ndarray,
    soft_vectors : np.ndarray,
) -> list[dict]:
    """
    Calcule les labels soft pour tous les points.

    Logique :
    - Point assigné par HDBSCAN (label != -1) :
        * Si archetype_conf >= ASSIGNED_CONF_MIN → on garde le label HDBSCAN
        * Si archetype_conf < ASSIGNED_CONF_MIN  → reclassification vers
          le cluster avec la plus haute proba dans soft_vectors
        * Label secondaire si un autre cluster dépasse ASSIGNED_SECONDARY_THRESHOLD
    - Outlier (label == -1) :
        * Si top1_prob >= SOFT_PRIMARY_THRESHOLD → label soft primaire
        * Si top2_prob >= SOFT_SECONDARY_THRESHOLD → label soft secondaire
        * Sinon → "unclassified"

    Retourne une liste de dicts dans le même ordre que labels.
    """
    results = []

    for i, label in enumerate(labels):
        probs      = soft_vectors[i]
        is_outlier = (label == -1)
        ranked     = sorted(enumerate(probs), key=lambda x: -x[1])
        top1_idx, top1_prob = ranked[0]
        top2_idx, top2_prob = ranked[1] if len(ranked) > 1 else (None, 0.0)

        if not is_outlier:
            hdbscan_prob = float(probs[label])

            if hdbscan_prob >= ASSIGNED_CONF_MIN:
                # Point bien assigné — on garde le label HDBSCAN
                primary_idx  = label
                primary_prob = hdbscan_prob
                is_reclassified = False
            else:
                # Conf trop basse → reclassification vers le meilleur cluster
                primary_idx  = top1_idx
                primary_prob = float(top1_prob)
                is_reclassified = True

            primary_name = ARCHETYPE_NAMES.get(primary_idx, f"cluster_{primary_idx}")

            # Label secondaire
            others = [(j, p) for j, p in ranked if j != primary_idx]
            sec_name, sec_prob = None, None
            if others and others[0][1] >= ASSIGNED_SECONDARY_THRESHOLD:
                sec_name = ARCHETYPE_NAMES.get(others[0][0], f"cluster_{others[0][0]}")
                sec_prob = float(others[0][1])

        else:
            is_reclassified = False

            if top1_prob >= SOFT_PRIMARY_THRESHOLD:
                primary_name = ARCHETYPE_NAMES.get(top1_idx, f"cluster_{top1_idx}")
                primary_prob = float(top1_prob)

                sec_name, sec_prob = None, None
                if top2_prob >= SOFT_SECONDARY_THRESHOLD and top2_idx != top1_idx:
                    sec_name = ARCHETYPE_NAMES.get(top2_idx, f"cluster_{top2_idx}")
                    sec_prob = float(top2_prob)
            else:
                primary_name = "unclassified"
                primary_prob = float(top1_prob)
                sec_name, sec_prob = None, None

        results.append({
            "archetype":                primary_name,
            "archetype_conf":           round(primary_prob, 4),
            "archetype_secondary":      sec_name,
            "archetype_secondary_conf": round(sec_prob, 4) if sec_prob else None,
            "is_soft_assigned":         bool(is_outlier),        # ← bool() natif
            "is_reclassified":          bool(is_reclassified),   # ← bool() natif
            "original_label":           int(label),
        })

    return results


# ──────────────────────────────────────────────────────────────────
# Pipeline d'entraînement
# ──────────────────────────────────────────────────────────────────

def _fit_umap(matrix: np.ndarray) -> tuple[UMAP, np.ndarray]:
    umap_model = UMAP(**UMAP_PARAMS)
    coords     = umap_model.fit_transform(matrix)
    return umap_model, coords


def _fit_clusterer(coords: np.ndarray) -> hdbscan.HDBSCAN:
    clusterer = hdbscan.HDBSCAN(**HDBSCAN_PARAMS)
    clusterer.fit(coords)
    return clusterer


# ──────────────────────────────────────────────────────────────────
# Écriture en base
# ──────────────────────────────────────────────────────────────────

def _save_assignments_to_db(
    engine,
    df          : pd.DataFrame,
    soft_labels : list[dict],
    umap_coords : np.ndarray,
) -> None:
    """
    Persiste les assignments dans player_archetypes.
    Inclut les colonnes soft assignment et double label.
    """
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS player_archetypes (
                player_id               INTEGER,
                saison                  VARCHAR,
                cluster_label           INTEGER,
                archetype               VARCHAR,
                archetype_conf          FLOAT,
                archetype_secondary     VARCHAR,
                archetype_secondary_conf FLOAT,
                is_soft_assigned        BOOLEAN,
                is_reclassified         BOOLEAN,
                umap_x                  FLOAT,
                umap_y                  FLOAT,
                PRIMARY KEY (player_id, saison)
            )
        """))

        rows = []
        for i, (_, row) in enumerate(df.iterrows()):
            sl = soft_labels[i]
            rows.append({
                "player_id":               int(row["player_id"]),
                "saison":                  row["saison"],
                "cluster_label":           int(sl["original_label"]),
                "archetype":               sl["archetype"],
                "archetype_conf":          sl["archetype_conf"],
                "archetype_secondary":     sl["archetype_secondary"],
                "archetype_secondary_conf": sl["archetype_secondary_conf"],
                "is_soft_assigned":        bool(sl["is_soft_assigned"]),   # ← bool() natif
                "is_reclassified":         bool(sl["is_reclassified"]),    # ← bool() natif
                "umap_x":                  round(float(umap_coords[i, 0]), 4),
                "umap_y":                  round(float(umap_coords[i, 1]), 4),
            })

        conn.execute(text("""
            INSERT INTO player_archetypes (
                player_id, saison, cluster_label,
                archetype, archetype_conf,
                archetype_secondary, archetype_secondary_conf,
                is_soft_assigned, is_reclassified,
                umap_x, umap_y
            )
            VALUES (
                :player_id, :saison, :cluster_label,
                :archetype, :archetype_conf,
                :archetype_secondary, :archetype_secondary_conf,
                :is_soft_assigned, :is_reclassified,
                :umap_x, :umap_y
            )
            ON CONFLICT (player_id, saison) DO UPDATE SET
                cluster_label            = EXCLUDED.cluster_label,
                archetype                = EXCLUDED.archetype,
                archetype_conf           = EXCLUDED.archetype_conf,
                archetype_secondary      = EXCLUDED.archetype_secondary,
                archetype_secondary_conf = EXCLUDED.archetype_secondary_conf,
                is_soft_assigned         = EXCLUDED.is_soft_assigned,
                is_reclassified          = EXCLUDED.is_reclassified,
                umap_x                   = EXCLUDED.umap_x,
                umap_y                   = EXCLUDED.umap_y
        """), rows)
        conn.commit()

    print(f"[role_engine] ✓ {len(rows)} assignments écrits en base.")


# ──────────────────────────────────────────────────────────────────
# Interface publique
# ──────────────────────────────────────────────────────────────────

def build_and_save_roles(engine) -> dict:
    """
    Pipeline complet :
    1. Charge les données depuis player_features
    2. Standardise avec RobustScaler
    3. UMAP (n_neighbors=20, cosine)
    4. HDBSCAN (mcs=40, ms=3, euclidean, eom)
    5. Soft assignment avec double label et reclassification
    6. Sauvegarde modèles + base
    """
    print("[role_engine] chargement des features...")

    cols_sql = ", ".join([f"pf.{f}" for f in SIMILARITY_FEATURES])
    query = f"""
        SELECT pf.player_id, pf.saison, pf.valuation_tier,
               pf.age_at_date, {cols_sql}
        FROM player_features pf
        WHERE pf.valuation_tier IN ('reliable', 'replacement_level')
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)

    df = df.reset_index(drop=True)
    print(f"[role_engine] {len(df)} joueurs-saisons chargés.")

    # Standardisation
    scaler = RobustScaler()
    matrix = scaler.fit_transform(df[SIMILARITY_FEATURES].fillna(0))
    _save(scaler, SCALER_PATH)
    print("[role_engine] scaler sauvegardé.")

    # UMAP
    print("[role_engine] calcul UMAP...")
    umap_model, coords = _fit_umap(matrix)
    _save(umap_model, UMAP_PATH)
    print("[role_engine] UMAP sauvegardé.")

    # HDBSCAN
    print("[role_engine] clustering HDBSCAN...")
    clusterer = _fit_clusterer(coords)
    _save(clusterer, CLUSTERER_PATH)

    labels = clusterer.labels_
    probs  = clusterer.probabilities_

    n_clusters = len(set(labels)) - (1 if OUTLIER_LABEL in labels else 0)
    n_outliers = (labels == OUTLIER_LABEL).sum()
    print(f"[role_engine] {n_clusters} clusters, {n_outliers} outliers.")

    unknown = set(labels) - set(ARCHETYPE_NAMES.keys())
    if unknown:
        print(f"[role_engine] ⚠️  labels sans nom : {unknown}")

    # Soft assignment
    print("[role_engine] calcul soft membership vectors...")
    soft_vectors = all_points_membership_vectors(clusterer)
    soft_labels  = _compute_soft_labels(labels, soft_vectors)

    n_reclassified  = sum(1 for s in soft_labels if s["is_reclassified"])
    n_soft_assigned = sum(1 for s in soft_labels if s["is_soft_assigned"] and s["archetype"] != "unclassified")
    n_unclassified  = sum(1 for s in soft_labels if s["archetype"] == "unclassified")
    n_double        = sum(1 for s in soft_labels if s["archetype_secondary"] is not None)

    print(f"[role_engine] reclassifiés (conf<0.15) : {n_reclassified}")
    print(f"[role_engine] soft assignés (outliers)  : {n_soft_assigned}")
    print(f"[role_engine] unclassified              : {n_unclassified}")
    print(f"[role_engine] double label              : {n_double}")

    _save_assignments_to_db(engine, df, soft_labels, coords)

    return ARCHETYPE_NAMES


def predict_archetype(
    vector     : np.ndarray,
    umap_model : Optional[UMAP]            = None,
    clusterer  : Optional[hdbscan.HDBSCAN] = None,
    scaler     : Optional[RobustScaler]    = None,
) -> dict:
    """
    Prédit l'archétype d'un nouveau vecteur de features (joueur hors DB).
    Pas de soft assignment ici — approximate_predict suffit pour un point unique.
    """
    if scaler     is None: scaler     = _load(SCALER_PATH)
    if umap_model is None: umap_model = _load(UMAP_PATH)
    if clusterer  is None: clusterer  = _load(CLUSTERER_PATH)

    scaled    = scaler.transform(vector.reshape(1, -1))
    coords_2d = umap_model.transform(scaled)

    pred_labels, pred_probs = hdbscan.approximate_predict(clusterer, coords_2d)
    label = int(pred_labels[0])
    prob  = float(pred_probs[0])

    archetype = ARCHETYPE_NAMES.get(label, OUTLIER_ARCHETYPE)

    # Si conf trop basse, on marque unclassified
    if prob < ASSIGNED_CONF_MIN and label != -1:
        archetype = OUTLIER_ARCHETYPE

    return {
        "cluster_label"  : label,
        "archetype"      : archetype,
        "archetype_conf" : round(prob, 4),
        "umap_x"         : round(float(coords_2d[0, 0]), 4),
        "umap_y"         : round(float(coords_2d[0, 1]), 4),
    }


def get_player_archetype(
    engine,
    player_id : int,
    season    : str,
) -> Optional[dict]:
    """Retourne l'archétype complet d'un joueur-saison depuis la base."""
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT cluster_label, archetype, archetype_conf,
                   archetype_secondary, archetype_secondary_conf,
                   is_soft_assigned, is_reclassified,
                   umap_x, umap_y
            FROM player_archetypes
            WHERE player_id = :pid AND saison = :saison
        """), {"pid": player_id, "saison": season}).fetchone()

    if row is None:
        return None

    return {
        "cluster_label":           row.cluster_label,
        "archetype":               row.archetype,
        "archetype_conf":          row.archetype_conf,
        "archetype_secondary":     row.archetype_secondary,
        "archetype_secondary_conf": row.archetype_secondary_conf,
        "is_soft_assigned":        row.is_soft_assigned,
        "is_reclassified":         row.is_reclassified,
        "umap_x":                  row.umap_x,
        "umap_y":                  row.umap_y,
    }


def get_archetype_players(
    engine,
    archetype_name : str,
    season         : Optional[str] = None,
    min_conf       : float = 0.15,
    include_secondary : bool = False,
) -> pd.DataFrame:
    """
    Retourne tous les joueurs d'un archétype donné.

    include_secondary=True inclut aussi les joueurs qui ont cet archétype
    en label secondaire — utile pour C6 quand il cherche des comparables
    à la frontière de deux profils.
    """
    if not include_secondary:
        query = """
            SELECT pa.player_id, p.player_name, pa.saison,
                   pa.archetype, pa.archetype_conf,
                   pa.archetype_secondary, pa.archetype_secondary_conf,
                   pa.is_soft_assigned, pa.is_reclassified,
                   pa.umap_x, pa.umap_y
            FROM player_archetypes pa
            JOIN players p ON p.player_id = pa.player_id
            WHERE pa.archetype = :archetype
            AND   pa.archetype_conf >= :min_conf
        """
        params: dict = {"archetype": archetype_name, "min_conf": min_conf}
    else:
        query = """
            SELECT pa.player_id, p.player_name, pa.saison,
                   pa.archetype, pa.archetype_conf,
                   pa.archetype_secondary, pa.archetype_secondary_conf,
                   pa.is_soft_assigned, pa.is_reclassified,
                   pa.umap_x, pa.umap_y
            FROM player_archetypes pa
            JOIN players p ON p.player_id = pa.player_id
            WHERE (
                pa.archetype = :archetype AND pa.archetype_conf >= :min_conf
            ) OR (
                pa.archetype_secondary = :archetype
                AND pa.archetype_secondary_conf >= :sec_min_conf
            )
        """
        params = {
            "archetype":     archetype_name,
            "min_conf":      min_conf,
            "sec_min_conf":  SOFT_SECONDARY_THRESHOLD,
        }

    if season:
        query += " AND pa.saison = :saison"
        params["saison"] = season

    query += " ORDER BY pa.archetype_conf DESC"

    with engine.connect() as conn:
        return pd.read_sql(text(query), conn, params=params)


def list_archetypes() -> dict:
    """Retourne le mapping complet {label -> nom}."""
    return ARCHETYPE_NAMES.copy()


def get_archetype_stats(engine, season: Optional[str] = None) -> pd.DataFrame:
    """
    Retourne un résumé statistique par archétype — utile pour C6.
    Nombre de joueurs, conf moyenne, % soft assignés.
    """
    query = """
        SELECT
            pa.archetype,
            COUNT(*)                          AS n_joueurs,
            ROUND(AVG(pa.archetype_conf), 3)  AS conf_moyenne,
            ROUND(AVG(CASE WHEN pa.is_soft_assigned THEN 1.0 ELSE 0.0 END), 3) AS pct_soft,
            ROUND(AVG(CASE WHEN pa.is_reclassified  THEN 1.0 ELSE 0.0 END), 3) AS pct_reclassified
        FROM player_archetypes pa
    """
    params: dict = {}
    if season:
        query += " WHERE pa.saison = :saison"
        params["saison"] = season

    query += " GROUP BY pa.archetype ORDER BY n_joueurs DESC"

    with engine.connect() as conn:
        return pd.read_sql(text(query), conn, params=params)