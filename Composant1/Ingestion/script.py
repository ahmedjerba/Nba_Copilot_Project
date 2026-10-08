import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import math

# --- Connexion ---
conn = psycopg2.connect(
    host="localhost",
    port=5432,
    dbname="nba_copilot",
    user="nba_user",
    password="nba_pass"
)
cur = conn.cursor()

# --- Chargement ---
df = pd.read_csv("player_stats_contracts.csv")
df['BIRTHDATE'] = pd.to_datetime(df['BIRTHDATE'], errors='coerce').dt.date

def v(x):
    if x is None:
        return None
    if str(x) in ('NaT', 'nan', 'None', ''):
        return None
    try:
        if math.isnan(float(x)):
            return None
    except (TypeError, ValueError):
        pass
    return x

def to_int(x):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None

def dedup(rows, key_indices):
    seen = set()
    clean = []
    for row in rows:
        key = tuple(row[i] for i in key_indices)
        if key not in seen:
            seen.add(key)
            clean.append(row)
    return clean

# ============================================================
# 1. TABLE players
# ============================================================
print("Insertion players...")
players = df.sort_values('POSS', ascending=False)\
            .drop_duplicates('PLAYER_ID', keep='first')[
    ['PLAYER_ID','PLAYER_NAME','BIRTHDATE','POSITION',
     'HEIGHT','WEIGHT','COUNTRY','DRAFT_YEAR','DRAFT_ROUND',
     'DRAFT_NUMBER','TEAM_ID']
].copy()

rows = [(
    int(r.PLAYER_ID), r.PLAYER_NAME,
    None if str(v(r.BIRTHDATE)) in ('NaT','None','nan','') else v(r.BIRTHDATE),
    v(r.POSITION), v(r.HEIGHT), v(r.WEIGHT), v(r.COUNTRY),
    to_int(r.DRAFT_YEAR), to_int(r.DRAFT_ROUND), to_int(r.DRAFT_NUMBER),
    int(r.TEAM_ID), None
) for r in players.itertuples()]

rows = dedup(rows, [0])
execute_values(cur, """
    INSERT INTO players
        (player_id, player_name, birthdate, position, height, weight,
         country, draft_year, draft_round, draft_number, team_id, team_name)
    VALUES %s
    ON CONFLICT (player_id) DO UPDATE SET
        player_name  = EXCLUDED.player_name,
        birthdate    = EXCLUDED.birthdate,
        position     = EXCLUDED.position,
        team_id      = EXCLUDED.team_id
""", rows)
print(f"  {len(rows)} joueurs insérés")

# ============================================================
# 2. TABLE player_stats
# ============================================================
print("Insertion player_stats...")
df_stats = df.sort_values('POSS', ascending=False)\
             .drop_duplicates(subset=['PLAYER_ID','SAISON','TEAM_ID'], keep='first')\
             .copy()

rows = [(
    int(r.PLAYER_ID), r.SAISON, int(r.TEAM_ID),
    v(r.TEAM_ABBREVIATION), v(r.TEAM_COUNT), v(r.AGE),
    v(r.GP), v(r.W_PCT), v(r.POSS), r.valuation_tier,
    v(r.PTS_PER100), v(r.REB_PER100), v(r.AST_PER100),
    v(r.STL_PER100), v(r.BLK_PER100), v(r.BLKA_PER100),
    v(r.TOV_PER100), v(r.PF_PER100),
    v(r.FGM_PER100), v(r.FGA_PER100),
    v(r.FG3M_PER100), v(r.FG3A_PER100),
    v(r.FTM_PER100), v(r.FTA_PER100),
    v(r.OREB_PER100), v(r.DREB_PER100), v(r.PLUS_MINUS_PER100),
    v(r.FG_PCT), v(r.FG3_PCT), v(r.FT_PCT), v(r.EFG_PCT),
    v(r.TS_PCT), v(r.OFF_RATING), v(r.DEF_RATING), v(r.NET_RATING),
    v(r.PACE), v(r.USG_PCT), v(r.PIE),
    v(r.AST_PCT), v(r.AST_TO), v(r.AST_RATIO),
    v(r.OREB_PCT), v(r.DREB_PCT), v(r.REB_PCT),
    v(r.TM_TOV_PCT), v(r.FGM_ADV), v(r.FGA_ADV), v(r.FG_PCT_ADV),
    v(r.MIN_PG), v(r.PTS_PG), v(r.REB_PG), v(r.AST_PG),
    v(r.STL_PG), v(r.BLK_PG), v(r.TOV_PG),
    v(r.FGM_PG), v(r.FGA_PG), v(r.FG3M_PG), v(r.FG3A_PG),
    v(r.FTM_PG), v(r.FTA_PG), v(r.OREB_PG), v(r.DREB_PG),
    v(r.PLUS_MINUS_PG)
) for r in df_stats.itertuples()]

rows = dedup(rows, [0, 1, 2])
print(f"  {len(rows)} lignes après dédoublonnage")

execute_values(cur, """
    INSERT INTO player_stats (
        player_id, saison, team_id, team_abbreviation, team_count,
        age, gp, w_pct, poss, valuation_tier,
        pts_per100, reb_per100, ast_per100, stl_per100, blk_per100,
        blka_per100, tov_per100, pf_per100,
        fgm_per100, fga_per100, fg3m_per100, fg3a_per100,
        ftm_per100, fta_per100, oreb_per100, dreb_per100, plus_minus_per100,
        fg_pct, fg3_pct, ft_pct, efg_pct, ts_pct,
        off_rating, def_rating, net_rating, pace, usg_pct, pie,
        ast_pct, ast_to, ast_ratio, oreb_pct, dreb_pct, reb_pct,
        tm_tov_pct, fgm_adv, fga_adv, fg_pct_adv,
        min_pg, pts_pg, reb_pg, ast_pg, stl_pg, blk_pg, tov_pg,
        fgm_pg, fga_pg, fg3m_pg, fg3a_pg, ftm_pg, fta_pg,
        oreb_pg, dreb_pg, plus_minus_pg
    ) VALUES %s
    ON CONFLICT (player_id, saison, team_id) DO UPDATE SET
        valuation_tier = EXCLUDED.valuation_tier,
        pts_per100     = EXCLUDED.pts_per100
""", rows)
print(f"  {len(rows)} lignes insérées")

# ============================================================
# 3. TABLE contracts
# ============================================================
print("Insertion contracts...")
contracts = df[df['base_salary'].notna()]\
    .sort_values('cap_hit', ascending=False)\
    .drop_duplicates(subset=['PLAYER_ID','SAISON'], keep='first')\
    .copy()

rows = [(
    int(r.PLAYER_ID), r.SAISON, None,
    int(r.base_salary), int(r.cap_hit),
    v(r.years_remaining),
    int(r.contract_value) if v(r.contract_value) else None
) for r in contracts.itertuples()]

rows = dedup(rows, [0, 1])
print(f"  {len(rows)} lignes après dédoublonnage")

execute_values(cur, """
    INSERT INTO contracts
        (player_id, saison, spotrac_player_id,
         base_salary, cap_hit, years_remaining, contract_value)
    VALUES %s
    ON CONFLICT (player_id, saison) DO UPDATE SET
        base_salary     = EXCLUDED.base_salary,
        cap_hit         = EXCLUDED.cap_hit,
        years_remaining = EXCLUDED.years_remaining,
        contract_value  = EXCLUDED.contract_value
""", rows)
print(f"  {len(rows)} contrats insérés")

# ============================================================
# 4. TABLE player_id_map
# ============================================================
print("Insertion player_id_map...")
id_map = pd.read_csv("player_id_map.csv")
id_map = id_map[id_map['PLAYER_ID'].notna()]

rows = [(
    int(r.PLAYER_ID), v(r.spotrac_player_id),
    r.player_name_stats, v(r.player_name_spotrac), r.match_type
) for r in id_map.itertuples()]

rows = dedup(rows, [0])

execute_values(cur, """
    INSERT INTO player_id_map
        (player_id, spotrac_player_id, player_name_stats,
         player_name_spotrac, match_type)
    VALUES %s
    ON CONFLICT (player_id) DO NOTHING
""", rows)
print(f"  {len(rows)} mappings insérés")

# ============================================================
conn.commit()
cur.close()
conn.close()
print("\nTerminé — toutes les tables sont remplies.")