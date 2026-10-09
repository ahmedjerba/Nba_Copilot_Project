# ─────────────────────────────────────────────
# Composant3 / feature_engineering.py
# Transformations FE appliquées sur le dataset pairs
# ─────────────────────────────────────────────
#
# ÉTAT DES FEATURES (résultats backtest in-sample) :
#
#   ✅ Rareté par poste         : -41k$  (6 features)
#   ✅ Interactions pos×skill   : -58k$  (6 features)  ← plus gros gain
#   ✅ Index de polyvalence      : -22k$  (2 features)
#   ❌ Ratios efficacité/volume : +6k$   (dégrade légèrement, désactivé par défaut)
#   ❌ PCA rebond               : +5k$   (n'apporte rien, désactivé par défaut)
#
#   🔴 À TESTER (features restantes) :
#      - age_peak_interaction   : jeune loin du pic = potentiel
#      - scoring_efficiency     : pts × ts_pct
#      - clean_playmaking       : ast / tov
#      - bankability_score      : dispo × trend × expérience
#      - archetype KMeans       : profil joueur (6 clusters)
# ─────────────────────────────────────────────

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ══════════════════════════════════════════════════════════════════════════════
# Définition des groupes de features de base (sans PIE)
# ══════════════════════════════════════════════════════════════════════════════

# Remplacer les groupes et NUM_BASE par ceci :

SCORING    = ["pts_per100_shrunk", "fg3m_per100_shrunk", "ts_pct", "usg_pct"]

REBOUNDING = ["reb_per100_shrunk", "oreb_pct", "dreb_pct"]

PLAYMAKING = ["ast_per100_shrunk", "ast_pct", "tov_per100_shrunk"]

DEFENSE    = ["stl_per100_shrunk", "blk_per100_shrunk", "def_rating"]

IMPACT     = ["net_rating", "plus_minus_per100_shrunk"]

DISPO      = ["poss", "gp"]

PROFIL     = ["age_at_date", "bmi", "saisons_experience"]
# ↑ supprimé : peak_distance (= linéaire de age), height_inches, weight (→ bmi)

TENDANCE   = ["trend_pts", "trend_pie"]

TIRS       = ["ft_pct", "free_throw_rate"]

NUM_BASE = (
    SCORING + REBOUNDING + PLAYMAKING +
    DEFENSE + IMPACT + DISPO + PROFIL +
    TENDANCE + TIRS
)
# → 25 features, ratio 24 obs/feature ✅

seen = set()
NUM_BASE = [f for f in NUM_BASE if not (f in seen or seen.add(f))]

CAT_FEATURES = ["position", "valuation_tier"]

# ══════════════════════════════════════════════════════════════════════════════
# FE VALIDÉES
# ══════════════════════════════════════════════════════════════════════════════

def add_rarete_par_poste(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Percentile rank de chaque skill au sein de la même position.
    Capture la rareté du profil sur le marché : un 3pt-shooter parmi les Centers
    vaut plus qu'un 3pt-shooter ordinaire parmi les Guards.
    """
    RARETE_FEATURES = [
        "pts_per100_shrunk", "fg3m_per100_shrunk", "ast_per100_shrunk",
        "stl_per100_shrunk", "blk_per100_shrunk", "reb_per100_shrunk"
    ]
    df = df.copy()
    cols = []
    for feat in RARETE_FEATURES:
        col = f"rarete_{feat}"
        df[col] = df.groupby("position")[feat].rank(pct=True)
        cols.append(col)
    return df, cols


def add_interactions_position_skill(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Interactions skill × poste (flag binaire).
    Le marché NBA paie différemment le scoring d'un Guard et d'un Center :
    ces features capturent cette non-linéarité sans exploser le nombre de variables.
    """
    df = df.copy()
    df["is_guard"]   = (df["position"] == "Guard").astype(int)
    df["is_center"]  = (df["position"] == "Center").astype(int)
    df["is_forward"] = (df["position"] == "Forward").astype(int)

    df["scoring_guard"]    = df["pts_per100_shrunk"]   * df["is_guard"]
    df["shooting_guard"]   = df["fg3m_per100_shrunk"]  * df["is_guard"]
    df["playmaking_guard"] = df["ast_per100_shrunk"]   * df["is_guard"]
    df["rim_center"]       = df["blk_per100_shrunk"]   * df["is_center"]
    df["reb_center"]       = df["reb_per100_shrunk"]   * df["is_center"]
    df["versatile_fwd"]    = df["fg3m_per100_shrunk"]  * df["is_forward"]

    cols = [
        "scoring_guard", "shooting_guard", "playmaking_guard",
        "rim_center", "reb_center", "versatile_fwd"
    ]
    return df, cols


def add_polyvalence(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Index de polyvalence et niveau global.
    - polyvalence  : std des 5 dimensions → mesure la versatilité
    - niveau_global: mean des 5 dimensions → niveau agrégé sans PIE
    """
    POLY_FEATURES = [
        "pts_per100_shrunk", "reb_per100_shrunk", "ast_per100_shrunk",
        "stl_per100_shrunk", "blk_per100_shrunk"
    ]
    df = df.copy()
    df["polyvalence"]    = df[POLY_FEATURES].std(axis=1)
    df["niveau_global"]  = df[POLY_FEATURES].mean(axis=1)
    return df, ["polyvalence", "niveau_global"]


# ══════════════════════════════════════════════════════════════════════════════
# FE À TESTER 🔴
# ══════════════════════════════════════════════════════════════════════════════

def add_age_peak_interaction(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    age_peak_interaction = peak_distance × (1 - age_at_date / 38)
    Interprétation : un joueur jeune ET loin de son pic a du potentiel.
    Le marché NBA anticipe-t-il la progression ? → à valider en backtest.
    peak_distance > 0 : encore avant le pic  /  < 0 : après le pic
    """
    df = df.copy()
    df["age_peak_interaction"] = (
        df["peak_distance"] * (1 - df["age_at_date"] / 38)
    )
    return df, ["age_peak_interaction"]


def add_scoring_efficiency(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    scoring_efficiency = pts_per100_shrunk × ts_pct
    Combine volume et efficacité de scoring.
    Hypothèse : le marché sur-paie le scoring pur ; l'efficacité module ce prix.
    """
    df = df.copy()
    df["scoring_efficiency"] = df["pts_per100_shrunk"] * df["ts_pct"]
    return df, ["scoring_efficiency"]


def add_clean_playmaking(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    clean_playmaking = ast_per100_shrunk / tov_per100_shrunk.clip(0.5)
    Ratio passes décisives / pertes de balle.
    Distingue le meneur propre du meneur à risque.
    """
    df = df.copy()
    df["clean_playmaking"] = (
        df["ast_per100_shrunk"] / df["tov_per100_shrunk"].clip(lower=0.5)
    )
    return df, ["clean_playmaking"]


def add_bankability_score(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    bankability_score = (gp/82) × (1 + trend_pie.clip(-0.03, 0.03)) × log1p(saisons_experience)
    Composite : disponibilité × tendance × ancienneté (= fiabilité perçue par les GMs).
    """
    df = df.copy()
    dispo     = (df["gp"] / 82).clip(0, 1.05)
    tendance  = 1 + df["trend_pie"].clip(-0.03, 0.03)
    experience = np.log1p(df["saisons_experience"])
    df["bankability_score"] = dispo * tendance * experience
    return df, ["bankability_score"]


def add_archetype_kmeans(
    df: pd.DataFrame,
    n_clusters: int = 6,
    random_state: int = 42
) -> tuple[pd.DataFrame, list[str]]:
    """
    Clustering KMeans (6 clusters) sur les dimensions scoring/rebond/passes/défense.
    Produit une variable catégorielle encodée en OHE → archetype_0 … archetype_5.
    Hypothèse : le marché paie différemment selon l'archétype (scorer pur, 3&D, etc.)
    Note : le clustering est fit sur le dataset courant (transductive).
    Pour la prod, il faudra sauvegarder le modèle KMeans et le réappliquer.
    """
    ARCHETYPE_FEATURES = [
        "pts_per100_shrunk", "reb_per100_shrunk", "ast_per100_shrunk",
        "stl_per100_shrunk", "blk_per100_shrunk", "fg3m_per100_shrunk",
        "usg_pct", "def_rating"
    ]
    df = df.copy()

    imp  = SimpleImputer(strategy="median")
    scal = StandardScaler()
    X    = scal.fit_transform(imp.fit_transform(df[ARCHETYPE_FEATURES]))

    km   = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)

    df["archetype"] = labels
    # OHE → archetype_0, archetype_1, …
    ohe_cols = pd.get_dummies(df["archetype"], prefix="archetype").astype(int)
    df = pd.concat([df, ohe_cols], axis=1)

    cols = [f"archetype_{i}" for i in range(n_clusters)]
    return df, cols


# ══════════════════════════════════════════════════════════════════════════════
# FE DÉSACTIVÉES (dégradent le modèle)
# ══════════════════════════════════════════════════════════════════════════════

def add_ratios_efficacite(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    ❌ Désactivé : dégrade de ~6k$ en backtest.
    Laissé ici pour traçabilité.
    """
    df = df.copy()
    df["efficacite_scoring"] = df["pts_per100_shrunk"] / df["usg_pct"].clip(0.1)
    df["efficacite_shoot3"]  = (
        df["fg3m_per100_shrunk"] / df["fg3a_per100_shrunk"].clip(0.1)
    )
    df["efficacite_passe"]   = (
        df["ast_per100_shrunk"] / df["tov_per100_shrunk"].clip(0.1)
    )
    return df, ["efficacite_scoring", "efficacite_shoot3", "efficacite_passe"]


def add_pca_rebond(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    ❌ Désactivé : n'apporte rien en backtest.
    Laissé ici pour traçabilité.
    """
    REB_FEATURES = [
        "reb_per100_shrunk", "oreb_per100_shrunk", "dreb_per100_shrunk",
        "oreb_pct", "dreb_pct", "reb_pct"
    ]
    df = df.copy()
    imp = SimpleImputer(strategy="median")
    X   = imp.fit_transform(df[REB_FEATURES])
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    df["pca_reb_1"] = X_pca[:, 0]
    df["pca_reb_2"] = X_pca[:, 1]
    return df, ["pca_reb_1", "pca_reb_2"]


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE COMPLET
# ══════════════════════════════════════════════════════════════════════════════

def build_features(
    df: pd.DataFrame,
    # FE validées — activées par défaut
    use_rarete:       bool = True,
    use_interactions: bool = True,
    use_polyvalence:  bool = True,
    # FE à tester — désactivées par défaut
    use_age_peak:     bool = True,
    use_scoring_eff:  bool = True,
    use_clean_pm:     bool = True,
    use_bankability:  bool = True,
    use_archetype:    bool = True,
    # FE dégradantes — désactivées
    use_ratios:       bool = False,
    use_pca_rebond:   bool = False,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Applique toutes les FE activées et retourne :
        (df_enrichi, num_features_list, cat_features_list)

    num_features_list : toutes les colonnes numériques à passer au modèle
    cat_features_list : colonnes catégorielles (position, valuation_tier)
    """
    extra_cols = []

    if use_rarete:
        df, cols = add_rarete_par_poste(df)
        extra_cols += cols

    if use_interactions:
        df, cols = add_interactions_position_skill(df)
        extra_cols += cols

    if use_polyvalence:
        df, cols = add_polyvalence(df)
        extra_cols += cols

    if use_age_peak:
        df, cols = add_age_peak_interaction(df)
        extra_cols += cols

    if use_scoring_eff:
        df, cols = add_scoring_efficiency(df)
        extra_cols += cols

    if use_clean_pm:
        df, cols = add_clean_playmaking(df)
        extra_cols += cols

    if use_bankability:
        df, cols = add_bankability_score(df)
        extra_cols += cols

    if use_archetype:
        df, cols = add_archetype_kmeans(df)
        extra_cols += cols

    if use_ratios:
        df, cols = add_ratios_efficacite(df)
        extra_cols += cols

    if use_pca_rebond:
        df, cols = add_pca_rebond(df)
        extra_cols += cols

    num_features = NUM_BASE + extra_cols
    # Garder uniquement les colonnes présentes dans df
    num_features = [c for c in num_features if c in df.columns]

    print(f"[feature_engineering] {len(num_features)} features numériques  "
          f"+ {len(CAT_FEATURES)} catégorielles")

    return df, num_features, CAT_FEATURES