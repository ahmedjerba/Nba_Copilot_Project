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

# Reconfigurer stdout en UTF-8 AVANT le Tee (Windows cp1252 → crash sur →, ✅, etc.)
try:
    sys.__stdout__.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            try:
                f.write(obj)
                f.flush()
            except Exception:
                pass
    def flush(self):
        for f in self.files:
            try:
                f.flush()
            except Exception:
                pass

log_file = open("results_3d.txt", "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, log_file)

# ════════════════════════════════════════════════════════════════════
# On teste uniquement les configs pertinentes (supprimer 2D/3D/6D-eom-mcs40 trop lents)
configs = [
    # ── Référence : 2D original (production actuelle)
    {"n_components": 2, "label": "2D",
     "mcs": 40, "ms": 3, "csm": "eom"},
    # ── Test 1 : 6D + mcs recalibré + leaf (recommandation principale)
    {"n_components": 6, "label": "6D_mcs15_leaf",
     "mcs": 15, "ms": 2, "csm": "leaf"},
    # ── Test 2 : 4D compromis + leaf
    {"n_components": 4, "label": "4D_mcs20_leaf",
     "mcs": 20, "ms": 2, "csm": "leaf"},
    # ── Test 3 : 6D + eom (mcs recalibré mais sans leaf pour comparer)
    {"n_components": 6, "label": "6D_mcs15_eom",
     "mcs": 15, "ms": 2, "csm": "eom"},
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
    elif dim == 3:
        df["umap3_x"] = coords[:, 0]
        df["umap3_y"] = coords[:, 1]
        df["umap3_z"] = coords[:, 2]
    else:  # 6D et plus
        for _d in range(dim):
            df[f"umap{dim}d_{_d}"] = coords[:, _d]

    # ── HDBSCAN (paramètres depuis cfg) ───────────────────────────────────
    mcs = cfg.get("mcs", 40)
    ms  = cfg.get("ms",  3)
    csm = cfg.get("csm", "eom")

    if dim > 2:
        # Cosine precomputed : seule métrique cohérente sur coords UMAP ≥ 3D
        print(f"Clustering HDBSCAN {label} (precomputed cosine, mcs={mcs}, ms={ms}, csm={csm})...")
        dist_matrix = cosine_distances(coords).astype(np.float64)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=ms,
            metric="precomputed",
            cluster_selection_method=csm,
        )
        clusterer.fit(dist_matrix)
        clusterer._prediction_data = PredictionData(coords, clusterer.condensed_tree_, min_samples=ms)
    else:
        print(f"Clustering HDBSCAN {label} (euclidean, mcs={mcs}, ms={ms}, csm={csm})...")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=ms,
            metric="euclidean",
            cluster_selection_method=csm,
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
        elif dim == 3:
            row["umap3_x"] = df.iloc[i]["umap3_x"]
            row["umap3_y"] = df.iloc[i]["umap3_y"]
            row["umap3_z"] = df.iloc[i]["umap3_z"]
        else:  # 6D+
            for _d in range(dim):
                row[f"umap{dim}d_{_d}"] = df.iloc[i][f"umap{dim}d_{_d}"]

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
# COMPARATIF FINAL : 2D vs 3D vs 6D
# ════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("COMPARATIF FINAL : 2D  vs  3D  vs  6D")
print(f"{'='*80}")

available_labels = [cfg["label"] for cfg in configs]
col_w = 12
header = f"{'Métrique':<35}" + "".join(f"{lbl:>{col_w}}" for lbl in available_labels)
print(header)
print("-" * (35 + col_w * len(available_labels)))

metrics = [
    ("Clusters",           "n_clusters"),
    ("Outliers HDBSCAN",   "n_outliers"),
    ("% outliers",         "pct"),
    ("Soft assignés",      "n_soft_ok"),
    ("Vrais unclassified", "n_unclassified"),
    ("Total distants",     "total_distants"),
    ("Conf moy. assignés", "avg_conf_assigned"),
    ("Conf moy. soft",     "avg_conf_soft"),
    ("Double label",       "double_label"),
]

ref_label = "2D"  # référence pour les flèches comparatives

for name, key in metrics:
    vals = {lbl: all_results[lbl][key] for lbl in available_labels if lbl in all_results}
    ref  = vals.get(ref_label)
    row  = f"  {name:<33}"
    for lbl in available_labels:
        v = vals.get(lbl)
        if v is None:
            row += f"{'N/A':>{col_w}}"
            continue
        # Meilleur que la référence 2D ?
        better = (
            (key == "n_clusters"          and ref is not None and abs(v - ref) <= 3) or
            (key in ("n_outliers", "pct", "n_unclassified", "total_distants") and ref is not None and v < ref) or
            (key in ("n_soft_ok", "double_label")                              and ref is not None and v > ref) or
            (key in ("avg_conf_assigned", "avg_conf_soft")                     and ref is not None and v > ref)
        ) if lbl != ref_label else None
        flag = "" if lbl == ref_label else (" ✅" if better else " →")
        if isinstance(v, float):
            row += f"{v:>{col_w}.4f}{flag}"
        else:
            row += f"{v:>{col_w}}{flag}"
    print(row)

# ── Focus superstars : taux d'outliers sur les known_players ────────
print(f"\n── OUTLIERS STARS par config ───────────────────────────────────")
for lbl in available_labels:
    if lbl not in all_results:
        continue
    sdf = all_results[lbl]["soft_df"]
    n_stars_outlier = ((sdf["player_name"].isin(known_players)) & (sdf["original_label"] == -1)).sum()
    n_stars_total   = sdf["player_name"].isin(known_players).sum()
    pct_s = n_stars_outlier / n_stars_total * 100 if n_stars_total else 0
    print(f"  {lbl} : {n_stars_outlier}/{n_stars_total} stars en outlier ({pct_s:.1f}%)")

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

# ── VIZ B-F : Un overview par config (générique) ────────────────────
for lbl in available_labels:
    if lbl == "2D" or lbl not in all_results:
        continue  # 2D déjà fait en VIZ A

    res    = all_results[lbl]
    sdf    = res["soft_df"]
    clst   = res["clusterer"]
    dim    = next(c["n_components"] for c in configs if c["label"] == lbl)

    # Axes de projection : toujours les 2 premiers axes UMAP du config
    if dim == 2:
        px, py = "umap_x", "umap_y"
    elif dim == 3:
        px, py = "umap3_x", "umap3_y"
    else:
        px, py = f"umap{dim}d_0", f"umap{dim}d_1"

    if px not in sdf.columns:
        print(f"[VIZ] colonnes manquantes pour {lbl}, skip")
        continue

    unique_lbls = sorted(set(clst.labels_))
    cmap_lbl = {
        l: "lightgray" if l == -1 else palette[i % 20]
        for i, l in enumerate(unique_lbls)
    }

    fig, axes = plt.subplots(1, 3, figsize=(22, 8))

    # Subplot 1 : clusters
    for l in unique_lbls:
        mask  = sdf["cluster_label"] == l
        name  = f"outliers ({mask.sum()})" if l == -1 else f"C{l} ({mask.sum()})"
        alpha = 0.15 if l == -1 else 0.7
        sz    = 10   if l == -1 else 18
        axes[0].scatter(sdf.loc[mask, px], sdf.loc[mask, py],
                        c=[cmap_lbl[l]], label=name, alpha=alpha, s=sz)
    for _, row in sdf[sdf["player_name"].isin(known_players)].iterrows():
        axes[0].annotate(row["player_name"], (row[px], row[py]),
                         fontsize=5, alpha=0.9, xytext=(3, 3), textcoords="offset points")
    axes[0].set_title(f"{lbl} — {res['n_clusters']} clusters")
    axes[0].legend(markerscale=1.2, fontsize=4, ncol=2, loc="upper left")

    # Subplot 2 : membership_prob
    sc = axes[1].scatter(sdf[px], sdf[py], c=sdf["membership_prob"],
                         cmap="RdYlGn", alpha=0.6, s=15, vmin=0, vmax=1)
    plt.colorbar(sc, ax=axes[1], label="membership_prob")
    axes[1].set_title(f"{lbl} — membership_prob\nmoy={sdf['membership_prob'].mean():.3f}")

    # Subplot 3 : archetype_conf
    sc2 = axes[2].scatter(sdf[px], sdf[py], c=sdf["archetype_conf"],
                          cmap="RdYlGn", alpha=0.6, s=15, vmin=0, vmax=1)
    plt.colorbar(sc2, ax=axes[2], label="archetype_conf")
    axes[2].set_title(f"{lbl} — archetype_conf\nmoy={sdf['archetype_conf'].mean():.3f}")

    plt.suptitle(f"UMAP {dim}D + HDBSCAN — {lbl}", fontsize=13)
    plt.tight_layout()
    fname = f"viz_{lbl.lower()}_overview.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"-> {fname}")

    # Export CSV
    csv_name = f"soft_assignment_{lbl.lower()}.csv"
    sdf.to_csv(csv_name, index=False, encoding="utf-8-sig")
    print(f"-> {csv_name} exporte")

# ── Fermeture log ────────────────────────────────────────────────────
sys.stdout = sys.__stdout__
log_file.close()
print("-> results_3d.txt exporte")
print("\nFini.")

