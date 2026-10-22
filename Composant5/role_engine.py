"""Découvre et nomme dynamiquement les archétypes de joueurs via UMAP + HDBSCAN + Centroïdes sémantiques."""

import pickle
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from sqlalchemy import text
from sklearn.preprocessing import RobustScaler
from sklearn.metrics.pairwise import cosine_distances
from umap import UMAP
import hdbscan
from hdbscan import all_points_membership_vectors

# ──────────────────────────────────────────────────────────────────
# Configuration des features et hyperparamètres
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
    "n_components": 2,
    "n_neighbors": 20,
    "min_dist": 0.1,
    "metric": "cosine",
    "random_state": 42,
}

HDBSCAN_PARAMS = {
    "min_cluster_size": 40,
    "min_samples": 3,
    "metric": "euclidean",
    "cluster_selection_method": "eom",
    "prediction_data": True,
}

# Seuils pour soft-assignment
SOFT_PRIMARY_THRESHOLD       = 0.10
SOFT_SECONDARY_THRESHOLD     = 0.06
ASSIGNED_CONF_MIN            = 0.15
ASSIGNED_SECONDARY_THRESHOLD = 0.15

OUTLIER_LABEL     = -1
OUTLIER_ARCHETYPE = "unclassified"

UMAP_PATH      = Path("models/c5_umap.pkl")
CLUSTERER_PATH = Path("models/c5_clusterer.pkl")
SCALER_PATH    = Path("models/c5_role_scaler.pkl")
NAMES_PATH     = Path("models/c5_archetype_names.pkl")
CENTROIDS_PATH = Path("models/c5_cluster_centroids.pkl")


# ──────────────────────────────────────────────────────────────────
# Persistance des modèles
# ──────────────────────────────────────────────────────────────────

def _save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _load(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"[role_engine] Fichier introuvable : {path}\n"
            "Exécutez build_and_save_roles(engine) au préalable."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


# ──────────────────────────────────────────────────────────────────
# Nommage sémantique dynamique des clusters
# ──────────────────────────────────────────────────────────────────

def _classify_cluster_profile(sub: pd.DataFrame) -> str:
    """
    Attribue un nom d'archétype basketball explicite, distinct et réaliste à un cluster
    en analysant les caractéristiques statistiques moyennes de ses membres.
    """
    pts = float(sub["pts_per100_shrunk"].mean())
    ast = float(sub["ast_per100_shrunk"].mean())
    blk = float(sub["blk_per100_shrunk"].mean())
    stl = float(sub["stl_per100_shrunk"].mean())
    thp = float(sub["three_rate"].mean())
    ftr = float(sub["free_throw_rate"].mean())
    usg = float(sub["usg_pct"].mean() * 100)
    ts  = float(sub["ts_pct"].mean() * 100)
    oreb = float(sub["oreb_pct"].mean() * 100)
    pie = float(sub["pie"].mean() * 100)

    # 1. Créateurs d'attaque majeurs / Superstars (PTS >= 27)
    if pts >= 27.0:
        if usg >= 26.5 and pie >= 13.0:
            return "mvp_offensive_engine"
        elif pts >= 29.8:
            return "elite_shotmaker_scorer"
        else:
            return "primary_playmaker_creator"

    # 2. Protecteurs d'arceau & intérieurs dominants
    if blk >= 1.5 and thp <= 0.25:
        if pts >= 22.5:
            return "two_way_unicorn_big"
        else:
            return "defensive_anchor_rim_protector"

    # 3. Intérieur d'impact au rebond offensif
    if oreb >= 10.0 and thp <= 0.20:
        return "offensive_glass_interior_big"

    # 4. Spécialistes purs du tir extérieur (Catch & Shoot)
    if thp >= 0.60:
        return "spot_up_spacer"

    # 5. Meneur défensif & voleur de ballon
    if stl >= 2.0 and ast >= 5.5:
        return "ball_hawk_playmaker"

    # 6. Attaquant agressif vers le cercle (provoque des lancers)
    if ftr >= 0.50:
        return "slashing_foul_drawer"

    # 7. Ailiers scoreurs extérieurs
    if thp >= 0.48 and pts >= 22.0:
        if usg >= 20.0:
            return "microwave_volume_shooter"
        else:
            return "perimeter_scoring_wing"

    # 8. Ailier connecteur & animateur
    if ast >= 5.0 and pie >= 10.0:
        return "connector_playmaking_forward"

    # 9. Rotation faible usage
    if usg <= 13.5:
        return "low_usage_replacement_wing"

    # 10. Ailier classique 3-and-D
    if ts >= 50.0 and thp >= 0.50:
        return "three_and_d_wing"

    # 11. Meneur organisateur du banc
    if ast >= 4.5 and thp <= 0.45:
        return "secondary_bench_handler"

    return "developmental_perimeter_wing"


def _generate_archetype_names(df: pd.DataFrame, cluster_col: str = "cluster") -> dict[int, str]:
    """
    Génère un dictionnaire {cluster_id -> nom_archetype} unique et sémantique
    pour tous les clusters découverts par HDBSCAN sans aucun suffixe numérique.
    """
    names: dict[int, str] = {OUTLIER_LABEL: OUTLIER_ARCHETYPE}
    unique_clusters = sorted([c for c in df[cluster_col].unique() if c >= 0])
    for c in unique_clusters:
        sub = df[df[cluster_col] == c]
        names[c] = _classify_cluster_profile(sub)

    return names


# ──────────────────────────────────────────────────────────────────
# Centroïdes et Fallback Cosinus
# ──────────────────────────────────────────────────────────────────

def _compute_cluster_centroids(matrix: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    """Calcule le centroïde 14D de chaque cluster valide (label >= 0)."""
    centroids = {}
    for l in set(labels):
        if l >= 0:
            centroids[l] = matrix[labels == l].mean(axis=0)
    return centroids


def _compute_soft_labels(
    labels: np.ndarray,
    soft_vectors: np.ndarray,
    matrix: np.ndarray,
    centroids: dict[int, np.ndarray],
    archetype_names: dict[int, str],
) -> list[dict]:
    """
    Calcule les labels d'archétypes pour tous les points :
    - Point assigné par HDBSCAN : garde ou reclassifie si conf trop basse.
    - Outlier HDBSCAN (-1) avec densité suffisante : rattrapé par soft membership vector.
    - Outlier résiduel (ex: superstar atypique) : rattrapé par proximité cosinus avec le
      centroïde d'archétype le plus proche.
    """
    results = []
    centroid_keys = sorted(centroids.keys())
    centroid_mat = np.array([centroids[k] for k in centroid_keys]) if centroid_keys else None

    for i, label in enumerate(labels):
        probs = soft_vectors[i]
        is_outlier = (label == OUTLIER_LABEL)
        ranked = sorted(enumerate(probs), key=lambda x: -x[1])
        top1_idx, top1_prob = ranked[0]
        top2_idx, top2_prob = ranked[1] if len(ranked) > 1 else (None, 0.0)

        if not is_outlier:
            hdbscan_prob = float(probs[label])

            if hdbscan_prob >= ASSIGNED_CONF_MIN:
                primary_idx = label
                primary_prob = hdbscan_prob
                is_reclassified = False
            else:
                primary_idx = top1_idx
                primary_prob = float(top1_prob)
                is_reclassified = True

            primary_name = archetype_names.get(primary_idx, f"cluster_{primary_idx}")

            # Label secondaire éventuel
            others = [(j, p) for j, p in ranked if j != primary_idx]
            sec_name, sec_prob = None, None
            if others and others[0][1] >= ASSIGNED_SECONDARY_THRESHOLD:
                sec_name = archetype_names.get(others[0][0], f"cluster_{others[0][0]}")
                sec_prob = float(others[0][1])

        else:
            is_reclassified = False

            if top1_prob >= SOFT_PRIMARY_THRESHOLD:
                # Rattrapage standard via soft membership
                primary_idx = top1_idx
                primary_name = archetype_names.get(primary_idx, f"cluster_{primary_idx}")
                primary_prob = float(top1_prob)

                sec_name, sec_prob = None, None
                if top2_prob >= SOFT_SECONDARY_THRESHOLD and top2_idx != top1_idx:
                    sec_name = archetype_names.get(top2_idx, f"cluster_{top2_idx}")
                    sec_prob = float(top2_prob)
            else:
                # Fallback centroïde pour les profils exceptionnels/isolés
                if centroid_mat is not None and len(centroid_mat) > 0:
                    dists = cosine_distances(matrix[i : i + 1], centroid_mat)[0]
                    best_k_idx = int(np.argmin(dists))
                    best_cluster_id = centroid_keys[best_k_idx]
                    best_dist = float(dists[best_k_idx])

                    primary_idx = best_cluster_id
                    primary_name = archetype_names.get(best_cluster_id, f"cluster_{best_cluster_id}")
                    # Confiance relative basée sur la proximité cosinus
                    primary_prob = max(0.15, min(0.95, round(1.0 - best_dist / 2.0, 4)))
                    sec_name, sec_prob = None, None
                else:
                    primary_name = OUTLIER_ARCHETYPE
                    primary_prob = float(top1_prob)
                    sec_name, sec_prob = None, None

        results.append({
            "archetype": primary_name,
            "archetype_conf": round(primary_prob, 4),
            "archetype_secondary": sec_name,
            "archetype_secondary_conf": round(sec_prob, 4) if sec_prob else None,
            "is_soft_assigned": bool(is_outlier),
            "is_reclassified": bool(is_reclassified),
            "original_label": int(label),
        })

    return results


# ──────────────────────────────────────────────────────────────────
# Écriture en base PostgreSQL
# ──────────────────────────────────────────────────────────────────

def _save_assignments_to_db(
    engine,
    df: pd.DataFrame,
    soft_labels: list[dict],
    umap_coords: np.ndarray,
) -> None:
    """Persiste les affectations d'archétypes dans la table player_archetypes."""
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS player_archetypes (
                player_id                INTEGER,
                saison                   VARCHAR,
                cluster_label            INTEGER,
                archetype                VARCHAR,
                archetype_conf           FLOAT,
                archetype_secondary      VARCHAR,
                archetype_secondary_conf FLOAT,
                is_soft_assigned         BOOLEAN,
                is_reclassified          BOOLEAN,
                umap_x                   FLOAT,
                umap_y                   FLOAT,
                PRIMARY KEY (player_id, saison)
            )
        """))

        rows = []
        for i, (_, row) in enumerate(df.iterrows()):
            sl = soft_labels[i]
            rows.append({
                "player_id": int(row["player_id"]),
                "saison": str(row["saison"]),
                "cluster_label": int(sl["original_label"]),
                "archetype": str(sl["archetype"]),
                "archetype_conf": float(sl["archetype_conf"]),
                "archetype_secondary": sl["archetype_secondary"],
                "archetype_secondary_conf": sl["archetype_secondary_conf"],
                "is_soft_assigned": bool(sl["is_soft_assigned"]),
                "is_reclassified": bool(sl["is_reclassified"]),
                "umap_x": round(float(umap_coords[i, 0]), 4),
                "umap_y": round(float(umap_coords[i, 1]), 4),
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

    print(f"[role_engine] [OK] {len(rows)} affectations écrites dans player_archetypes.")


# ──────────────────────────────────────────────────────────────────
# Interface publique d'entraînement et d'inférence
# ──────────────────────────────────────────────────────────────────

def build_and_save_roles(engine) -> dict[int, str]:
    """
    Entraîne le modèle topologique d'archétypes et persiste les résultats :
    1. Charge les features fiables et replacement-level depuis player_features.
    2. Standardise via RobustScaler.
    3. Projette en 2D via UMAP pour la préservation géométrique et les coordonnées visuelles.
    4. Segmente par HDBSCAN.
    5. Nomme dynamiquement les clusters d'après leurs statistiques réelles.
    6. Applique le soft assignment avec fallback centroïde pour éliminer les non-classifiés.
    7. Sauvegarde les modèles et écrit dans player_archetypes.
    """
    print("[role_engine] Chargement des features...")
    cols_sql = ", ".join([f"pf.{f}" for f in SIMILARITY_FEATURES])
    query = f"""
        SELECT pf.player_id, pf.saison, pf.valuation_tier,
               pf.age_at_date, {cols_sql}
        FROM player_features pf
        WHERE pf.valuation_tier IN ('reliable', 'replacement_level')
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)

    df = df.drop_duplicates(subset=["player_id", "saison"]).reset_index(drop=True)
    print(f"[role_engine] {len(df)} profils joueur-saison chargés.")

    # 1. Standardisation
    scaler = RobustScaler()
    matrix = scaler.fit_transform(df[SIMILARITY_FEATURES].fillna(0))
    _save(scaler, SCALER_PATH)

    # 2. UMAP
    print("[role_engine] Calcul projection UMAP...")
    umap_model = UMAP(**UMAP_PARAMS)
    coords = umap_model.fit_transform(matrix)
    _save(umap_model, UMAP_PATH)

    # 3. HDBSCAN
    print("[role_engine] Clustering HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(**HDBSCAN_PARAMS)
    clusterer.fit(coords)
    _save(clusterer, CLUSTERER_PATH)

    labels = clusterer.labels_
    df["cluster"] = labels

    # 4. Nommage sémantique dynamique des clusters
    archetype_names = _generate_archetype_names(df, cluster_col="cluster")
    _save(archetype_names, NAMES_PATH)
    print(f"[role_engine] {len(archetype_names) - 1} archétypes sémantiques identifiés.")

    # 5. Centroïdes et rattrapage
    centroids = _compute_cluster_centroids(matrix, labels)
    _save(centroids, CENTROIDS_PATH)

    print("[role_engine] Calcul soft membership & rattrapage centroïdes...")
    soft_vectors = all_points_membership_vectors(clusterer)
    soft_labels = _compute_soft_labels(labels, soft_vectors, matrix, centroids, archetype_names)

    n_soft_assigned = sum(1 for s in soft_labels if s["is_soft_assigned"])
    n_reclassified = sum(1 for s in soft_labels if s["is_reclassified"])
    n_unclassified = sum(1 for s in soft_labels if s["archetype"] == OUTLIER_ARCHETYPE)

    print(f"[role_engine] Soft-assigned (outliers rattrapés) : {n_soft_assigned}")
    print(f"[role_engine] Reclassifiés (faible densité)     : {n_reclassified}")
    print(f"[role_engine] Non-classifiés résiduels           : {n_unclassified} (0 attendu)")

    # 6. Sauvegarde en DB
    _save_assignments_to_db(engine, df, soft_labels, coords)

    return archetype_names


def predict_archetype(
    vector: np.ndarray,
    umap_model: Optional[UMAP] = None,
    clusterer: Optional[hdbscan.HDBSCAN] = None,
    scaler: Optional[RobustScaler] = None,
    archetype_names: Optional[dict[int, str]] = None,
    centroids: Optional[dict[int, np.ndarray]] = None,
) -> dict:
    """
    Prédit l'archétype d'un nouveau profil statistique (joueur hors DB).
    Utilise approximate_predict avec fallback centroïde cosinus si besoin.
    """
    if scaler is None:
        scaler = _load(SCALER_PATH)
    if umap_model is None:
        umap_model = _load(UMAP_PATH)
    if clusterer is None:
        clusterer = _load(CLUSTERER_PATH)
    if archetype_names is None:
        archetype_names = _load(NAMES_PATH)
    if centroids is None:
        centroids = _load(CENTROIDS_PATH)

    scaled = scaler.transform(vector.reshape(1, -1))
    coords_2d = umap_model.transform(scaled)

    pred_labels, pred_probs = hdbscan.approximate_predict(clusterer, coords_2d)
    label = int(pred_labels[0])
    prob = float(pred_probs[0])

    if label >= 0 and prob >= ASSIGNED_CONF_MIN:
        archetype = archetype_names.get(label, OUTLIER_ARCHETYPE)
    else:
        # Fallback centroïde
        centroid_keys = sorted(centroids.keys())
        centroid_mat = np.array([centroids[k] for k in centroid_keys])
        dists = cosine_distances(scaled, centroid_mat)[0]
        best_k = int(np.argmin(dists))
        label = centroid_keys[best_k]
        prob = max(0.15, min(0.95, round(1.0 - float(dists[best_k]) / 2.0, 4)))
        archetype = archetype_names.get(label, OUTLIER_ARCHETYPE)

    return {
        "cluster_label": label,
        "archetype": archetype,
        "archetype_conf": round(prob, 4),
        "umap_x": round(float(coords_2d[0, 0]), 4),
        "umap_y": round(float(coords_2d[0, 1]), 4),
    }


def get_player_archetype(
    engine,
    player_id: int,
    season: str,
) -> Optional[dict]:
    """Récupère l'archétype d'un joueur-saison depuis la base de données."""
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
        "cluster_label": row.cluster_label,
        "archetype": row.archetype,
        "archetype_conf": row.archetype_conf,
        "archetype_secondary": row.archetype_secondary,
        "archetype_secondary_conf": row.archetype_secondary_conf,
        "is_soft_assigned": row.is_soft_assigned,
        "is_reclassified": row.is_reclassified,
        "umap_x": row.umap_x,
        "umap_y": row.umap_y,
    }


def get_archetype_players(
    engine,
    archetype_name: str,
    season: Optional[str] = None,
    min_conf: float = 0.15,
    include_secondary: bool = False,
) -> pd.DataFrame:
    """Retourne la liste des joueurs appartenant à un archétype donné."""
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
            "archetype": archetype_name,
            "min_conf": min_conf,
            "sec_min_conf": SOFT_SECONDARY_THRESHOLD,
        }

    if season:
        query += " AND pa.saison = :saison"
        params["saison"] = season

    query += " ORDER BY pa.archetype_conf DESC"

    with engine.connect() as conn:
        return pd.read_sql(text(query), conn, params=params)


def list_archetypes() -> dict[int, str]:
    """Retourne la liste des archétypes sémantiques entraînés."""
    return _load(NAMES_PATH)


def get_archetype_stats(engine, season: Optional[str] = None) -> pd.DataFrame:
    """Retourne un résumé statistique par archétype (volume, confiance moyenne, etc.)."""
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