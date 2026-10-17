import pandas as pd
import numpy as np
import hdbscan
from hdbscan import all_points_membership_vectors
from hdbscan.prediction import PredictionData
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sqlalchemy import create_engine
from sklearn.preprocessing import RobustScaler
from umap import UMAP
from config import DB_URL
import sys
from io import StringIO

engine = create_engine(DB_URL)

SIMILARITY_FEATURES = [
    "pts_per100_shrunk", "ast_per100_shrunk", "stl_per100_shrunk",
    "blk_per100_shrunk", "ts_pct", "usg_pct", "three_rate",
    "free_throw_rate", "oreb_pct", "dreb_pct",
    "trend_pts", "trend_pie", "pie", "net_rating",
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

def simplify_position(pos):
    if pd.isna(pos): return "Unknown"
    pos = str(pos)
    if "Center" in pos and "Forward" not in pos: return "Center"
    if "Forward" in pos and "Guard" not in pos: return "Forward"
    if "Guard" in pos and "Forward" not in pos: return "Guard"
    return "Hybrid"

df["position_simple"] = df["position"].apply(simplify_position)

scaler = RobustScaler()
matrix = scaler.fit_transform(df[SIMILARITY_FEATURES].fillna(0))

known_players = [
    "LeBron James", "Stephen Curry", "Nikola Jokic",
    "Giannis Antetokounmpo", "Kevin Durant", "Joel Embiid",
    "Luka Doncic", "Jayson Tatum", "Anthony Davis",
    "Ja Morant", "Damian Lillard", "Bam Adebayo",
    "Rudy Gobert", "Draymond Green", "Klay Thompson",
    "Nikola Vucevic", "Karl-Anthony Towns", "Devin Booker",
    "Trae Young", "Tyrese Haliburton", "Paolo Banchero",
    "Victor Wembanyama", "Chet Holmgren", "Franz Wagner",
]

CLUSTER_NAMES = {
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

SOFT_PRIMARY_THRESHOLD       = 0.10
SOFT_SECONDARY_THRESHOLD     = 0.06
ASSIGNED_SECONDARY_THRESHOLD = 0.15

# ════════════════════════════════════════════════════════════════════
# Tee — tout ce qui est print() va aussi dans le fichier texte
# ════════════════════════════════════════════════════════════════════
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

log_file = open("results_3d.txt", "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, log_file)

# ════════════════════════════════════════════════════════════════════
# Boucle sur les deux configs : 2D (référence) et 3D (test)
# ════════════════════════════════════════════════════════════════════
configs = [
    {"n_components": 2, "label": "2D"},
    {
    "n_components": 3,
    "label": "3D",
    "umap_min_dist": 0.1,   # même que 2D
    "mcs": 40,               # même que 2D
    "ms": 3,                 # même que 2D
    # metric="precomputed" via cosine_distances(coords)
},
]

all_results = {}

for cfg in configs:
    dim   = cfg["n_components"]
    label = cfg["label"]

    print(f"\n{'='*70}")
    print(f"CONFIG {label} — UMAP n_components={dim}")
    print(f"{'='*70}")
    print(f"Pool : {len(df)} joueurs-saisons")

    # ── UMAP ────────────────────────────────────────────────────────
    print(f"Calcul UMAP {label}...")
    umap_model = UMAP(
        n_components=dim,
        n_neighbors=20,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    coords = umap_model.fit_transform(matrix)

    # Stocker les coords pour la viz
    if dim == 2:
        df["umap_x"] = coords[:, 0]
        df["umap_y"] = coords[:, 1]
    else:
        df["umap3_x"] = coords[:, 0]
        df["umap3_y"] = coords[:, 1]
        df["umap3_z"] = coords[:, 2]

    # ── HDBSCAN ─────────────────────────────────────────────────────
    if dim == 3:
        print(f"Clustering HDBSCAN {label} (precomputed cosine, min_cluster_size=40, min_samples=3)...")
        dist_matrix = cosine_distances(coords).astype(np.float64)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=40,
            min_samples=3,
            metric="precomputed",
            cluster_selection_method="eom",
        )
        clusterer.fit(dist_matrix)
        clusterer._prediction_data = PredictionData(coords, clusterer.condensed_tree_, min_samples=3)
    else:
        print(f"Clustering HDBSCAN {label} (min_cluster_size=15, min_samples=2)...")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=40,
            min_samples=3,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )
        clusterer.fit(coords)

    n_clusters = len(set(clusterer.labels_)) - (1 if -1 in clusterer.labels_ else 0)
    n_outliers = (clusterer.labels_ == -1).sum()
    pct        = round(n_outliers / len(df) * 100, 1)
    print(f"→ {n_clusters} clusters, {n_outliers} outliers ({pct}%)")

    # ── Soft membership vectors ──────────────────────────────────────
    print(f"Calcul soft membership vectors {label}...")
    soft_vectors = all_points_membership_vectors(clusterer)

    # ── Soft assignment ──────────────────────────────────────────────
    soft_rows = []
    for i, lbl in enumerate(clusterer.labels_):
        probs      = soft_vectors[i]
        is_outlier = (lbl == -1)
        ranked     = sorted(enumerate(probs), key=lambda x: -x[1])
        top1_idx, top1_prob = ranked[0]
        top2_idx, top2_prob = ranked[1] if len(ranked) > 1 else (None, 0.0)

        if not is_outlier:
            primary_name = CLUSTER_NAMES.get(lbl, f"cluster_{lbl}")
            primary_prob = float(probs[lbl])
            others = [(j, p) for j, p in ranked if j != lbl]
            sec_name, sec_prob = None, None
            if others and others[0][1] >= ASSIGNED_SECONDARY_THRESHOLD:
                sec_name = CLUSTER_NAMES.get(others[0][0], f"cluster_{others[0][0]}")
                sec_prob = float(others[0][1])
        else:
            if top1_prob >= SOFT_PRIMARY_THRESHOLD:
                primary_name = CLUSTER_NAMES.get(top1_idx, f"cluster_{top1_idx}")
                primary_prob = float(top1_prob)
                sec_name, sec_prob = None, None
                if top2_prob >= SOFT_SECONDARY_THRESHOLD and top2_idx != top1_idx:
                    sec_name = CLUSTER_NAMES.get(top2_idx, f"cluster_{top2_idx}")
                    sec_prob = float(top2_prob)
            else:
                primary_name = "unclassified"
                primary_prob = float(top1_prob)
                sec_name, sec_prob = None, None

        row = {
            "player_name":              df.iloc[i]["player_name"],
            "saison":                   df.iloc[i]["saison"],
            "position_simple":          df.iloc[i]["position_simple"],
            "pie":                      df.iloc[i]["pie"],
            "original_label":           int(lbl),
            "membership_prob":          float(clusterer.probabilities_[i]),
            "is_soft_assigned":         is_outlier,
            "archetype":                primary_name,
            "archetype_conf":           round(primary_prob, 4),
            "archetype_secondary":      sec_name,
            "archetype_secondary_conf": round(sec_prob, 4) if sec_prob else None,
        }
        if dim == 2:
            row["umap_x"] = df.iloc[i]["umap_x"]
            row["umap_y"] = df.iloc[i]["umap_y"]
        else:
            row["umap3_x"] = df.iloc[i]["umap3_x"]
            row["umap3_y"] = df.iloc[i]["umap3_y"]
            row["umap3_z"] = df.iloc[i]["umap3_z"]

        soft_rows.append(row)

    soft_df = pd.DataFrame(soft_rows)
    soft_df["cluster_label"] = clusterer.labels_

    # ── Stats résumé ─────────────────────────────────────────────────
    all_outliers  = soft_df[soft_df["original_label"] == -1]
    n_soft_ok     = (all_outliers["archetype"] != "unclassified").sum()
    n_unclassified= (all_outliers["archetype"] == "unclassified").sum()
    double        = soft_df[
        (soft_df["original_label"] != -1) &
        (soft_df["archetype_secondary"].notna())
    ]
    avg_conf_assigned = soft_df[~soft_df["is_soft_assigned"]]["archetype_conf"].mean()
    avg_conf_soft     = soft_df[soft_df["is_soft_assigned"] & (soft_df["archetype"] != "unclassified")]["archetype_conf"].mean()

    print(f"\n── MÉTRIQUES {label} ──────────────────────────────────────────")
    print(f"Clusters                 : {n_clusters}")
    print(f"Outliers HDBSCAN         : {n_outliers} ({pct}%)")
    print(f"Soft assignés (≥0.10)    : {n_soft_ok}")
    print(f"Vrais unclassified       : {n_unclassified}")
    print(f"Double label             : {len(double)} / {(~soft_df['is_soft_assigned']).sum()}")
    print(f"Conf moy. assignés       : {avg_conf_assigned:.4f}")
    print(f"Conf moy. soft           : {avg_conf_soft:.4f}")

    print(f"\n── OUTLIERS STARS {label} ─────────────────────────────────────")
    outlier_stars = soft_df[
        (soft_df["original_label"] == -1) &
        (soft_df["player_name"].isin(known_players))
    ].sort_values("archetype_conf", ascending=False)
    print(outlier_stars[[
        "player_name", "saison",
        "archetype", "archetype_conf",
        "archetype_secondary", "archetype_secondary_conf",
    ]].to_string(index=False))

    print(f"\n── DISTRIBUTION OUTLIERS {label} ──────────────────────────────")
    print(all_outliers["archetype"].value_counts().to_string())

    print(f"\n── DOUBLE LABEL STARS {label} ─────────────────────────────────")
    stars_double = double[double["player_name"].isin(known_players)].sort_values(
        "archetype_secondary_conf", ascending=False
    ).head(20)
    print(stars_double[[
        "player_name", "saison",
        "archetype", "archetype_conf",
        "archetype_secondary", "archetype_secondary_conf",
    ]].to_string(index=False))

    print(f"\n── MEMBRES DISTANTS PAR ARCHETYPE {label} (conf < 0.30) ───────")
    archetypes_to_check = [a for a in soft_df["archetype"].unique() if a != "unclassified"]
    total_distants = 0
    for arch in sorted(archetypes_to_check):
        distants = soft_df[
            (soft_df["archetype"] == arch) &
            (~soft_df["is_soft_assigned"]) &
            (soft_df["archetype_conf"] < 0.30)
        ].sort_values("archetype_conf")
        total_distants += len(distants)
        if len(distants) > 0:
            print(f"\n  {arch} — {len(distants)} distants :")
            print(distants[["player_name", "saison", "archetype_conf"]].to_string(index=False))
    print(f"\nTotal distants toutes catégories : {total_distants}")

    print(f"\n── CLUSTERS DÉTAIL {label} ─────────────────────────────────────")
    for cl in sorted(soft_df["cluster_label"].unique()):
        grp  = soft_df[soft_df["cluster_label"] == cl]
        name = CLUSTER_NAMES.get(cl, f"cluster_{cl}")
        stars_in = grp[grp["player_name"].isin(known_players)]["player_name"].unique().tolist()
        print(f"  {'OUTLIERS' if cl == -1 else f'C{cl}':<4} {name:<22} "
              f"n={len(grp):<5} "
              f"conf_moy={grp['archetype_conf'].mean():.3f}  "
              f"stars={stars_in[:4]}")

    # Export CSV
    csv_name = f"soft_assignment_{label.lower()}.csv"
    soft_df.to_csv(csv_name, index=False, encoding="utf-8-sig")
    print(f"\n→ {csv_name} exporté")

    all_results[label] = {
        "soft_df":    soft_df,
        "clusterer":  clusterer,
        "coords":     coords,
        "n_clusters": n_clusters,
        "n_outliers": n_outliers,
        "pct":        pct,
        "n_soft_ok":  n_soft_ok,
        "n_unclassified": n_unclassified,
        "total_distants": total_distants,
        "avg_conf_assigned": avg_conf_assigned,
        "avg_conf_soft": avg_conf_soft,
        "double_label": len(double),
    }

# ════════════════════════════════════════════════════════════════════
# COMPARATIF FINAL 2D vs 3D
# ════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("COMPARATIF FINAL 2D vs 3D")
print(f"{'='*70}")
print(f"{'Métrique':<35} {'2D':>12} {'3D':>12}")
print("-"*60)
metrics = [
    ("Clusters",             "n_clusters"),
    ("Outliers HDBSCAN",     "n_outliers"),
    ("% outliers",           "pct"),
    ("Soft assignés",        "n_soft_ok"),
    ("Vrais unclassified",   "n_unclassified"),
    ("Total distants",       "total_distants"),
    ("Conf moy. assignés",   "avg_conf_assigned"),
    ("Conf moy. soft",       "avg_conf_soft"),
    ("Double label",         "double_label"),
]
for name, key in metrics:
    v2 = all_results["2D"][key]
    v3 = all_results["3D"][key]
    flag = "✅" if (
        (key in ("n_clusters",) and abs(v3-v2) <= 3) or
        (key in ("n_outliers","pct","n_unclassified","total_distants") and v3 < v2) or
        (key in ("n_soft_ok","double_label") and v3 > v2) or
        (key in ("avg_conf_assigned","avg_conf_soft") and v3 > v2)
    ) else "→"
    if isinstance(v2, float):
        print(f"  {name:<33} {v2:>12.4f} {v3:>12.4f}  {flag}")
    else:
        print(f"  {name:<33} {v2:>12} {v3:>12}  {flag}")

# ════════════════════════════════════════════════════════════════════
# VISUALISATIONS
# ════════════════════════════════════════════════════════════════════

palette       = plt.cm.tab20(np.linspace(0, 1, 20))
colors_pos    = {"Guard": "blue", "Forward": "green",
                 "Center": "red", "Hybrid": "orange", "Unknown": "gray"}

# ── VIZ A : 2D classique (référence) ────────────────────────────────
soft_2d   = all_results["2D"]["soft_df"]
clust_2d  = all_results["2D"]["clusterer"]

unique_labels_2d = sorted(clust_2d.labels_)
unique_labels_2d = sorted(set(clust_2d.labels_))
color_map_2d = {
    lbl: "lightgray" if lbl == -1 else palette[i % 20]
    for i, lbl in enumerate(unique_labels_2d)
}

fig, axes = plt.subplots(1, 3, figsize=(22, 8))
for lbl in unique_labels_2d:
    mask  = soft_2d["cluster_label"] == lbl
    name  = f"outliers ({mask.sum()})" if lbl == -1 else f"C{lbl} ({mask.sum()})"
    alpha = 0.15 if lbl == -1 else 0.7
    sz    = 10   if lbl == -1 else 18
    axes[0].scatter(soft_2d.loc[mask,"umap_x"], soft_2d.loc[mask,"umap_y"],
                    c=[color_map_2d[lbl]], label=name, alpha=alpha, s=sz)
for _, row in soft_2d[soft_2d["player_name"].isin(known_players)].iterrows():
    axes[0].annotate(row["player_name"], (row["umap_x"], row["umap_y"]),
                     fontsize=5, alpha=0.9, xytext=(3,3), textcoords="offset points")
axes[0].set_title(f"2D — {all_results['2D']['n_clusters']} clusters")
axes[0].legend(markerscale=1.2, fontsize=5, ncol=2, loc="upper left")

sc = axes[1].scatter(soft_2d["umap_x"], soft_2d["umap_y"],
                     c=soft_2d["membership_prob"], cmap="RdYlGn",
                     alpha=0.6, s=15, vmin=0, vmax=1)
plt.colorbar(sc, ax=axes[1], label="membership_prob")
axes[1].set_title("2D — Certitude HDBSCAN")

sc2 = axes[2].scatter(soft_2d["umap_x"], soft_2d["umap_y"],
                      c=soft_2d["archetype_conf"], cmap="RdYlGn",
                      alpha=0.6, s=15, vmin=0, vmax=1)
plt.colorbar(sc2, ax=axes[2], label="archetype_conf")
axes[2].set_title("2D — Confiance soft assignment")

plt.suptitle("Référence 2D", fontsize=13)
plt.tight_layout()
plt.savefig("viz_A_2d_reference.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n→ viz_A_2d_reference.png")

# ── VIZ B : 3D — 3 projections planes (XY, XZ, YZ) ─────────────────
soft_3d  = all_results["3D"]["soft_df"]
clust_3d = all_results["3D"]["clusterer"]

unique_labels_3d = sorted(set(clust_3d.labels_))
color_map_3d = {
    lbl: "lightgray" if lbl == -1 else palette[i % 20]
    for i, lbl in enumerate(unique_labels_3d)
}

projections = [("umap3_x","umap3_y","XY"), ("umap3_x","umap3_z","XZ"), ("umap3_y","umap3_z","YZ")]
fig, axes = plt.subplots(1, 3, figsize=(22, 8))
for ax, (px, py, ptitle) in zip(axes, projections):
    for lbl in unique_labels_3d:
        mask  = soft_3d["cluster_label"] == lbl
        alpha = 0.12 if lbl == -1 else 0.65
        sz    = 8    if lbl == -1 else 16
        ax.scatter(soft_3d.loc[mask, px], soft_3d.loc[mask, py],
                   c=[color_map_3d[lbl]], alpha=alpha, s=sz)
    for _, row in soft_3d[soft_3d["player_name"].isin(known_players)].iterrows():
        ax.annotate(row["player_name"], (row[px], row[py]),
                    fontsize=4.5, alpha=0.85, xytext=(2,2), textcoords="offset points")
    ax.set_title(f"3D projection {ptitle} — {all_results['3D']['n_clusters']} clusters")
plt.suptitle("3D — Projections planes (XY / XZ / YZ)", fontsize=13)
plt.tight_layout()
plt.savefig("viz_B_3d_projections.png", dpi=150, bbox_inches="tight")
plt.close()
print("→ viz_B_3d_projections.png")

# ── VIZ C : 3D scatter interactif matplotlib (4 angles) ─────────────
fig = plt.figure(figsize=(20, 16))
angles = [(20, 45), (20, 135), (60, 45), (5, 90)]
for idx, (elev, azim) in enumerate(angles):
    ax = fig.add_subplot(2, 2, idx+1, projection="3d")
    for lbl in unique_labels_3d:
        mask  = soft_3d["cluster_label"] == lbl
        alpha = 0.10 if lbl == -1 else 0.60
        sz    = 8    if lbl == -1 else 18
        ax.scatter(
            soft_3d.loc[mask,"umap3_x"],
            soft_3d.loc[mask,"umap3_y"],
            soft_3d.loc[mask,"umap3_z"],
            c=[color_map_3d[lbl]], alpha=alpha, s=sz,
        )
    # Annoter seulement les stars pour lisibilité
    for _, row in soft_3d[soft_3d["player_name"].isin(known_players)].iterrows():
        ax.text(row["umap3_x"], row["umap3_y"], row["umap3_z"],
                row["player_name"], fontsize=4, alpha=0.8)
    ax.set_title(f"élév={elev}° azim={azim}°", fontsize=9)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("X", fontsize=7)
    ax.set_ylabel("Y", fontsize=7)
    ax.set_zlabel("Z", fontsize=7)

plt.suptitle(f"3D scatter — {all_results['3D']['n_clusters']} clusters HDBSCAN", fontsize=13)
plt.tight_layout()
plt.savefig("viz_C_3d_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print("→ viz_C_3d_scatter.png")

# ── VIZ D : Confiance 2D vs 3D côte à côte ──────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# 2D membership_prob
sc = axes[0][0].scatter(soft_2d["umap_x"], soft_2d["umap_y"],
                         c=soft_2d["membership_prob"], cmap="RdYlGn",
                         alpha=0.7, s=14, vmin=0, vmax=1)
plt.colorbar(sc, ax=axes[0][0])
axes[0][0].set_title(f"2D — membership_prob HDBSCAN\nmoy={soft_2d['membership_prob'].mean():.3f}")

# 2D archetype_conf
sc = axes[0][1].scatter(soft_2d["umap_x"], soft_2d["umap_y"],
                         c=soft_2d["archetype_conf"], cmap="RdYlGn",
                         alpha=0.7, s=14, vmin=0, vmax=1)
plt.colorbar(sc, ax=axes[0][1])
axes[0][1].set_title(f"2D — archetype_conf soft\nmoy={soft_2d['archetype_conf'].mean():.3f}")

# 3D projection XY membership_prob
sc = axes[1][0].scatter(soft_3d["umap3_x"], soft_3d["umap3_y"],
                         c=soft_3d["membership_prob"], cmap="RdYlGn",
                         alpha=0.7, s=14, vmin=0, vmax=1)
plt.colorbar(sc, ax=axes[1][0])
axes[1][0].set_title(f"3D (proj XY) — membership_prob HDBSCAN\nmoy={soft_3d['membership_prob'].mean():.3f}")

# 3D projection XY archetype_conf
sc = axes[1][1].scatter(soft_3d["umap3_x"], soft_3d["umap3_y"],
                         c=soft_3d["archetype_conf"], cmap="RdYlGn",
                         alpha=0.7, s=14, vmin=0, vmax=1)
plt.colorbar(sc, ax=axes[1][1])
axes[1][1].set_title(f"3D (proj XY) — archetype_conf soft\nmoy={soft_3d['archetype_conf'].mean():.3f}")

plt.suptitle("Comparaison confiance 2D vs 3D", fontsize=13)
plt.tight_layout()
plt.savefig("viz_D_confidence_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("→ viz_D_confidence_comparison.png")

# ── VIZ E : Par archetype 3D (projection XY) ────────────────────────
archetype_list_3d   = sorted(soft_3d["archetype"].unique())
archetypes_to_plot  = [a for a in archetype_list_3d if a != "unclassified"]
n_arch = len(archetypes_to_plot)
ncols  = 3
nrows  = (n_arch + ncols - 1) // ncols

fig, axes_grid = plt.subplots(nrows, ncols, figsize=(7*ncols, 6*nrows))
axes_flat = axes_grid.flatten()

for idx, arch in enumerate(archetypes_to_plot):
    ax = axes_flat[idx]
    mask_hard      = (soft_3d["archetype"] == arch) & (~soft_3d["is_soft_assigned"])
    mask_soft_asgn = (soft_3d["archetype"] == arch) & (soft_3d["is_soft_assigned"])
    mask_low       = mask_hard & (soft_3d["archetype_conf"] < 0.30)
    mask_good      = mask_hard & (soft_3d["archetype_conf"] >= 0.30)
    mask_rest      = ~(soft_3d["archetype"] == arch)

    ax.scatter(soft_3d.loc[mask_rest,"umap3_x"], soft_3d.loc[mask_rest,"umap3_y"],
               c="lightgray", alpha=0.12, s=7)
    ax.scatter(soft_3d.loc[mask_good,"umap3_x"], soft_3d.loc[mask_good,"umap3_y"],
               c="steelblue", alpha=0.8, s=20,
               label=f"conf≥0.30 ({mask_good.sum()})")
    ax.scatter(soft_3d.loc[mask_low,"umap3_x"], soft_3d.loc[mask_low,"umap3_y"],
               c="red", alpha=0.9, s=28, marker="X",
               label=f"⚠ conf<0.30 ({mask_low.sum()})")
    ax.scatter(soft_3d.loc[mask_soft_asgn,"umap3_x"], soft_3d.loc[mask_soft_asgn,"umap3_y"],
               c="darkorange", alpha=0.9, s=26, marker="^",
               label=f"soft ({mask_soft_asgn.sum()})")

    for _, row in soft_3d[(soft_3d["archetype"]==arch) & soft_3d["player_name"].isin(known_players)].iterrows():
        ax.annotate(f"{row['player_name']}\n({row['archetype_conf']:.2f})",
                    (row["umap3_x"], row["umap3_y"]),
                    fontsize=5, color="darkblue", xytext=(4,4), textcoords="offset points")
    for _, row in soft_3d[mask_low].sort_values("archetype_conf").head(6).iterrows():
        if row["player_name"] not in known_players:
            ax.annotate(f"{row['player_name']}\n({row['archetype_conf']:.2f})",
                        (row["umap3_x"], row["umap3_y"]),
                        fontsize=4.5, color="darkred", xytext=(4,4), textcoords="offset points")

    total = mask_hard.sum() + mask_soft_asgn.sum()
    ax.set_title(f"{arch}\n({total} total | {mask_low.sum()} distants)", fontsize=9)
    ax.legend(fontsize=6, loc="upper left")

for idx in range(n_arch, len(axes_flat)):
    axes_flat[idx].set_visible(False)

plt.suptitle("3D (proj XY) — Par archetype : assignés / soft / distants", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("viz_E_3d_per_archetype.png", dpi=150, bbox_inches="tight")
plt.close()
print("→ viz_E_3d_per_archetype.png")

# ── Fermeture log ────────────────────────────────────────────────────
sys.stdout = sys.__stdout__
log_file.close()
print("→ results_3d.txt exporté")
print("\nFichiers générés :")
print("  results_3d.txt")
print("  soft_assignment_2d.csv")
print("  soft_assignment_3d.csv")
print("  viz_A_2d_reference.png")
print("  viz_B_3d_projections.png")
print("  viz_C_3d_scatter.png")
print("  viz_D_confidence_comparison.png")
print("  viz_E_3d_per_archetype.png")