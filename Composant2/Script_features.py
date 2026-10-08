import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
import yaml
from datetime import date

# --- Connexion ---
conn = psycopg2.connect(
    host="localhost", port=5432,
    dbname="nba_copilot", user="nba_user", password="nba_pass"
)

# --- Chargement config ---
with open('config_features.yaml', 'r') as f:
    config = yaml.safe_load(f)

K_MAX = config['shrinkage']['k_max']
POSS_SEUIL = config['shrinkage']['poss_seuil']
FEATURE_VERSION = 'v1.0'

# --- Chargement données ---
df = pd.read_sql("""
    SELECT 
        ps.player_id, ps.saison, ps.team_id, ps.poss, ps.gp,
        ps.valuation_tier, ps.age,
        ps.pts_per100, ps.reb_per100, ps.ast_per100, ps.stl_per100,
        ps.blk_per100, ps.blka_per100, ps.tov_per100, ps.pf_per100,
        ps.fgm_per100, ps.fga_per100, ps.fg3m_per100, ps.fg3a_per100,
        ps.ftm_per100, ps.fta_per100, ps.oreb_per100, ps.dreb_per100,
        ps.plus_minus_per100,
        ps.fg_pct, ps.fg3_pct, ps.ft_pct, ps.efg_pct, ps.ts_pct,
        ps.off_rating, ps.def_rating, ps.net_rating,
        ps.usg_pct, ps.pie, ps.ast_pct, ps.oreb_pct, ps.dreb_pct, ps.reb_pct,
        p.birthdate, p.height, p.weight, p.position
    FROM player_stats ps
    JOIN players p ON ps.player_id = p.player_id
    WHERE ps.poss > 0
""", conn)

print(f"Lignes chargées : {len(df)}")

# ============================================================
# SHRINKAGE CONTINU PAR POSS
# ============================================================
def k_joueur(poss):
    if poss >= POSS_SEUIL:
        return 0
    return K_MAX * (1 - poss / POSS_SEUIL)

def shrink(stat, poss, moyenne):
    k = k_joueur(poss)
    if k == 0:
        return stat
    if pd.isna(stat):
        return moyenne
    return (poss * stat + k * moyenne) / (poss + k)

metriques = [
    'pts_per100', 'reb_per100', 'ast_per100', 'stl_per100',
    'blk_per100', 'blka_per100', 'tov_per100', 'pf_per100',
    'fgm_per100', 'fga_per100', 'fg3m_per100', 'fg3a_per100',
    'ftm_per100', 'fta_per100', 'oreb_per100', 'dreb_per100',
    'plus_minus_per100'
]

# Moyenne globale de référence (sur les fiables uniquement)
fiables = df[df['valuation_tier'] == 'reliable']
moyennes = {col: fiables[col].mean() for col in metriques}

print("Moyennes de référence (fiables) :")
for col, moy in moyennes.items():
    print(f"  {col:<25} : {moy:.2f}")

# Appliquer le shrinkage
for col in metriques:
    df[f'{col}_shrunk'] = df.apply(
        lambda r: shrink(r[col], r['poss'], moyennes[col]), axis=1
    )

# ============================================================
# PROFIL PHYSIQUE
# ============================================================
def height_to_inches(h):
    """'6-8' ou '6-08' → 80.0"""
    try:
        parts = str(h).replace('"','').split('-')
        return int(parts[0]) * 12 + int(parts[1])
    except:
        return None

df['height_inches'] = df['height'].apply(height_to_inches)
df['bmi'] = df.apply(
    lambda r: (r['weight'] / (r['height_inches'] ** 2)) * 703
    if pd.notna(r['weight']) and pd.notna(r['height_inches']) and r['height_inches'] > 0
    else None, axis=1
)

# ============================================================
# AGE A LA DATE DE REFERENCE
# ============================================================
# Date de référence = 1er janvier de l'année de fin de saison
# Ex: saison '2021-22' → date_ref = 2022-01-01
def date_reference(saison):
    annee_fin = int(saison.split('-')[0]) + 1
    return date(annee_fin, 1, 1)

def age_at_date(birthdate, saison):
    if pd.isna(birthdate):
        return None
    try:
        ref = date_reference(saison)
        bd = pd.to_datetime(birthdate).date()
        age = (ref - bd).days / 365.25
        return round(age, 2)
    except:
        return None

df['age_at_date'] = df.apply(
    lambda r: age_at_date(r['birthdate'], r['saison']), axis=1
)
df['peak_distance'] = df['age_at_date'].apply(
    lambda a: round(a - 27, 2) if pd.notna(a) else None
)

# ============================================================
# SAISONS EXPERIENCE
# ============================================================
# Nombre de saisons jouées avant la saison courante
saisons_order = ['2021-22','2022-23','2023-24','2024-25','2025-26']
saison_rank = {s: i for i, s in enumerate(saisons_order)}

df['saison_rank'] = df['saison'].map(saison_rank)
df = df.sort_values(['player_id','saison_rank'])

df['saisons_experience'] = df.groupby('player_id').cumcount()

# ============================================================
# FEATURES DERIVEES
# ============================================================
def safe_div(a, b):
    try:
        if pd.isna(a) or pd.isna(b) or b == 0:
            return None
        return a / b
    except:
        return None

df['three_rate']      = df.apply(lambda r: safe_div(r['fg3a_per100'], r['fga_per100']), axis=1)
df['free_throw_rate'] = df.apply(lambda r: safe_div(r['fta_per100'], r['fga_per100']), axis=1)
df['assist_rate']     = df.apply(lambda r: safe_div(r['ast_per100'], r['fga_per100']), axis=1)
df['turnover_rate']   = df.apply(
    lambda r: safe_div(
        r['tov_per100'],
        r['fga_per100'] + 0.44 * r['fta_per100'] + r['ast_per100']
    ), axis=1
)

# ============================================================
# FEATURES TEMPORELLES (trend N vs N-1)
# ============================================================
df = df.sort_values(['player_id','saison_rank'])

df['trend_pts'] = df.groupby('player_id')['pts_per100_shrunk'].diff()
df['trend_pie'] = df.groupby('player_id')['pie'].diff()

# ============================================================
# INSERTION EN BASE
# ============================================================
print(f"\nInsertion de {len(df)} lignes dans player_features...")

def v(x):
    if x is None:
        return None
    try:
        if np.isnan(float(x)):
            return None
    except (TypeError, ValueError):
        pass
    return x

rows = [(
    int(r.player_id), r.saison, int(r.team_id), FEATURE_VERSION,
    v(r.poss), v(r.gp), r.valuation_tier,
    v(r.height_inches), v(r.weight), v(r.bmi),
    v(r.age_at_date), v(r.peak_distance), int(r.saisons_experience),
    v(r.pts_per100_shrunk), v(r.reb_per100_shrunk), v(r.ast_per100_shrunk),
    v(r.stl_per100_shrunk), v(r.blk_per100_shrunk), v(r.tov_per100_shrunk),
    v(r.fgm_per100_shrunk), v(r.fga_per100_shrunk), v(r.fg3m_per100_shrunk),
    v(r.fg3a_per100_shrunk), v(r.ftm_per100_shrunk), v(r.fta_per100_shrunk),
    v(r.oreb_per100_shrunk), v(r.dreb_per100_shrunk), v(r.plus_minus_per100_shrunk),
    v(r.blka_per100_shrunk), v(r.pf_per100_shrunk),
    v(r.fg_pct), v(r.fg3_pct), v(r.ft_pct), v(r.efg_pct), v(r.ts_pct),
    v(r.off_rating), v(r.def_rating), v(r.net_rating),
    v(r.usg_pct), v(r.pie), v(r.ast_pct), v(r.oreb_pct), v(r.dreb_pct), v(r.reb_pct),
    v(r.three_rate), v(r.free_throw_rate), v(r.turnover_rate), v(r.assist_rate),
    v(r.trend_pts), v(r.trend_pie)
) for r in df.itertuples()]

# Dédoublonnage
seen = set()
rows_clean = []
for row in rows:
    key = (row[0], row[1], row[2], row[3])
    if key not in seen:
        seen.add(key)
        rows_clean.append(row)

execute_values(conn.cursor(), """
    INSERT INTO player_features (
        player_id, saison, team_id, feature_version,
        poss, gp, valuation_tier,
        height_inches, weight, bmi,
        age_at_date, peak_distance, saisons_experience,
        pts_per100_shrunk, reb_per100_shrunk, ast_per100_shrunk,
        stl_per100_shrunk, blk_per100_shrunk, tov_per100_shrunk,
        fgm_per100_shrunk, fga_per100_shrunk, fg3m_per100_shrunk,
        fg3a_per100_shrunk, ftm_per100_shrunk, fta_per100_shrunk,
        oreb_per100_shrunk, dreb_per100_shrunk, plus_minus_per100_shrunk,
        blka_per100_shrunk, pf_per100_shrunk,
        fg_pct, fg3_pct, ft_pct, efg_pct, ts_pct,
        off_rating, def_rating, net_rating,
        usg_pct, pie, ast_pct, oreb_pct, dreb_pct, reb_pct,
        three_rate, free_throw_rate, turnover_rate, assist_rate,
        trend_pts, trend_pie
    ) VALUES %s
    ON CONFLICT (player_id, saison, team_id, feature_version) DO UPDATE SET
        valuation_tier   = EXCLUDED.valuation_tier,
        pts_per100_shrunk = EXCLUDED.pts_per100_shrunk,
        trend_pts        = EXCLUDED.trend_pts
""", rows_clean)

conn.commit()
conn.close()
print(f"Terminé — {len(rows_clean)} lignes insérées dans player_features")