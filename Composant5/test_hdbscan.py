import pandas as pd
import numpy as np
import hdbscan
from sqlalchemy import create_engine
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances

from config import DB_URL

engine = create_engine(DB_URL)

SIMILARITY_FEATURES = [
    "pts_per100_shrunk", "ast_per100_shrunk", "reb_per100_shrunk",
    "stl_per100_shrunk", "blk_per100_shrunk", "ts_pct", "efg_pct",
    "usg_pct", "three_rate", "free_throw_rate", "ast_pct",
    "oreb_pct", "dreb_pct", "trend_pts", "trend_pie", "pie", "net_rating",
]

query = f"""
    SELECT pf.player_id, pf.saison, pf.valuation_tier,
           pf.age_at_date, p.player_name, p.position,
           {', '.join(['pf.' + f for f in SIMILARITY_FEATURES])}
    FROM player_features pf
    JOIN players p ON p.player_id = pf.player_id
    WHERE pf.valuation_tier IN ('reliable', 'replacement_level')
"""

df = pd.read_sql(query, engine)
print(f"Pool : {len(df)} joueurs-saisons\n")

scaler = RobustScaler()
matrix = scaler.fit_transform(df[SIMILARITY_FEATURES].fillna(0))

configs = [
    {"min_cluster_size": 20, "min_samples": 2},
    {"min_cluster_size": 30, "min_samples": 3},
    {"min_cluster_size": 40, "min_samples": 5},
    {"min_cluster_size": 15, "min_samples": 2},
]

# ------------------------------------------------------------------
# Test 1 : euclidienne, algorithme generic
# ------------------------------------------------------------------

print("=== Test euclidienne + generic ===")
for cfg in configs:
    c = hdbscan.HDBSCAN(
        **cfg,
        metric="euclidean",
        algorithm="generic",
        cluster_selection_method="eom",
    )
    c.fit(matrix)
    n_clusters = len(set(c.labels_)) - (1 if -1 in c.labels_ else 0)
    n_outliers = (c.labels_ == -1).sum()
    print(f"  {cfg} → {n_clusters} clusters, {n_outliers} outliers ({round(n_outliers/len(df)*100,1)}%)")

# ------------------------------------------------------------------
# Test 2 : PCA 6 composantes + euclidienne
# ------------------------------------------------------------------

print("\n=== Test PCA (6 composantes) + euclidienne ===")
pca = PCA(n_components=6)
matrix_pca = pca.fit_transform(matrix)
print(f"  variance expliquée : {round(pca.explained_variance_ratio_.sum() * 100, 1)}%")

for cfg in configs:
    c = hdbscan.HDBSCAN(
        **cfg,
        metric="euclidean",
        algorithm="generic",
        cluster_selection_method="eom",
    )
    c.fit(matrix_pca)
    n_clusters = len(set(c.labels_)) - (1 if -1 in c.labels_ else 0)
    n_outliers = (c.labels_ == -1).sum()
    print(f"  {cfg} → {n_clusters} clusters, {n_outliers} outliers ({round(n_outliers/len(df)*100,1)}%)")

# ------------------------------------------------------------------
# Test 3 : cosinus via precomputed
# ------------------------------------------------------------------

print("\n=== Test cosinus (precomputed) ===")
dist_matrix = cosine_distances(matrix)

for cfg in configs:
    c = hdbscan.HDBSCAN(
        **cfg,
        metric="precomputed",
        algorithm="generic",
        cluster_selection_method="eom",
    )
    c.fit(dist_matrix)
    n_clusters = len(set(c.labels_)) - (1 if -1 in c.labels_ else 0)
    n_outliers = (c.labels_ == -1).sum()
    print(f"  {cfg} → {n_clusters} clusters, {n_outliers} outliers ({round(n_outliers/len(df)*100,1)}%)")