import os
import sys
from pathlib import Path
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# --- Chargement des variables d'environnement ---
# Recherche du fichier .env à la racine du projet ou dans le dossier actuel
env_path = Path(__file__).resolve().parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
DB_NAME = os.getenv("POSTGRES_DB", "nba_copilot")
DB_USER = os.getenv("POSTGRES_USER", "nba_user")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "nba_pass")

print(f"Connexion à la base de données {DB_NAME} sur {DB_HOST}:{DB_PORT}...")
conn = psycopg2.connect(
    host=DB_HOST,
    port=DB_PORT,
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASS
)
cur = conn.cursor()

# ============================================================
# 1. NETTOYAGE DES DONNÉES DE TEST FICTIVES
# ============================================================
# Supprime l'équipe fictive avec ID=1 créée manuellement lors de tests
cur.execute("DELETE FROM team_payrolls WHERE team_id = 1;")
cur.execute("DELETE FROM teams WHERE team_id = 1;")
conn.commit()

# ============================================================
# 2. POPULATION DE LA TABLE TEAMS (30 FRANCHISES NBA OFFICIELLES)
# ============================================================
print("\n[1/2] Insertion des 30 équipes NBA officielles dans la table 'teams'...")

NBA_TEAMS = [
    (1610612737, "Atlanta Hawks", "ATL", "East", "Southeast"),
    (1610612738, "Boston Celtics", "BOS", "East", "Atlantic"),
    (1610612739, "Cleveland Cavaliers", "CLE", "East", "Central"),
    (1610612740, "New Orleans Pelicans", "NOP", "West", "Southwest"),
    (1610612741, "Chicago Bulls", "CHI", "East", "Central"),
    (1610612742, "Dallas Mavericks", "DAL", "West", "Southwest"),
    (1610612743, "Denver Nuggets", "DEN", "West", "Northwest"),
    (1610612744, "Golden State Warriors", "GSW", "West", "Pacific"),
    (1610612745, "Houston Rockets", "HOU", "West", "Southwest"),
    (1610612746, "LA Clippers", "LAC", "West", "Pacific"),
    (1610612747, "Los Angeles Lakers", "LAL", "West", "Pacific"),
    (1610612748, "Miami Heat", "MIA", "East", "Southeast"),
    (1610612749, "Milwaukee Bucks", "MIL", "East", "Central"),
    (1610612750, "Minnesota Timberwolves", "MIN", "West", "Northwest"),
    (1610612751, "Brooklyn Nets", "BKN", "East", "Atlantic"),
    (1610612752, "New York Knicks", "NYK", "East", "Atlantic"),
    (1610612753, "Orlando Magic", "ORL", "East", "Southeast"),
    (1610612754, "Indiana Pacers", "IND", "East", "Central"),
    (1610612755, "Philadelphia 76ers", "PHI", "East", "Atlantic"),
    (1610612756, "Phoenix Suns", "PHX", "West", "Pacific"),
    (1610612757, "Portland Trail Blazers", "POR", "West", "Northwest"),
    (1610612758, "Sacramento Kings", "SAC", "West", "Pacific"),
    (1610612759, "San Antonio Spurs", "SAS", "West", "Southwest"),
    (1610612760, "Oklahoma City Thunder", "OKC", "West", "Northwest"),
    (1610612761, "Toronto Raptors", "TOR", "East", "Atlantic"),
    (1610612762, "Utah Jazz", "UTA", "West", "Northwest"),
    (1610612763, "Memphis Grizzlies", "MEM", "West", "Southwest"),
    (1610612764, "Washington Wizards", "WAS", "East", "Southeast"),
    (1610612765, "Detroit Pistons", "DET", "East", "Central"),
    (1610612766, "Charlotte Hornets", "CHA", "East", "Southeast"),
]

execute_values(cur, """
    INSERT INTO teams (team_id, team_name, abbreviation, conference, division)
    VALUES %s
    ON CONFLICT (team_id) DO UPDATE SET
        team_name    = EXCLUDED.team_name,
        abbreviation = EXCLUDED.abbreviation,
        conference   = EXCLUDED.conference,
        division     = EXCLUDED.division;
""", NBA_TEAMS)

conn.commit()
print(f"  -> {len(NBA_TEAMS)} franchises NBA insérées/mises à jour dans 'teams'.")

# ============================================================
# 3. POPULATION DE LA TABLE TEAM_PAYROLLS PAR SAISON
# ============================================================
print("\n[2/2] Calcul et insertion des masses salariales dans 'team_payrolls'...")

# Identifier toutes les saisons présentes dans la table contracts
cur.execute("SELECT DISTINCT saison FROM contracts ORDER BY saison;")
saisons = [row[0] for row in cur.fetchall()]
print(f"  Saisons détectées dans 'contracts' : {', '.join(saisons)}")

# Calcul rigoureux des masses salariales par équipe et par saison :
# 1. On croise chaque équipe (30 franchises) avec chaque saison détectée
# 2. On agrège les contrats des joueurs appartenant à cette équipe pour cette saison
# 3. On enregistre le total_cap_hit, total_base_salary et nb_joueurs
sql_payrolls = """
    WITH all_team_seasons AS (
        SELECT t.team_id, s.saison
        FROM teams t
        CROSS JOIN (SELECT DISTINCT saison FROM contracts) s
    ),
    contract_teams AS (
        SELECT
            c.player_id,
            c.saison,
            COALESCE(ps.team_id, p.team_id) AS team_id,
            c.cap_hit,
            c.base_salary
        FROM contracts c
        JOIN players p ON p.player_id = c.player_id
        LEFT JOIN player_stats ps ON ps.player_id = c.player_id AND ps.saison = c.saison
    ),
    computed_payrolls AS (
        SELECT
            team_id,
            saison,
            COALESCE(SUM(cap_hit), 0)         AS total_cap_hit,
            COALESCE(SUM(base_salary), 0)     AS total_base_salary,
            COUNT(DISTINCT player_id)         AS nb_joueurs
        FROM contract_teams
        WHERE team_id IS NOT NULL
        GROUP BY team_id, saison
    )
    INSERT INTO team_payrolls (team_id, saison, total_cap_hit, total_base_salary, nb_joueurs)
    SELECT
        ats.team_id,
        ats.saison,
        COALESCE(cp.total_cap_hit, 0)     AS total_cap_hit,
        COALESCE(cp.total_base_salary, 0) AS total_base_salary,
        COALESCE(cp.nb_joueurs, 0)        AS nb_joueurs
    FROM all_team_seasons ats
    LEFT JOIN computed_payrolls cp
        ON ats.team_id = cp.team_id AND ats.saison = cp.saison
    ON CONFLICT (team_id, saison) DO UPDATE SET
        total_cap_hit     = EXCLUDED.total_cap_hit,
        total_base_salary = EXCLUDED.total_base_salary,
        nb_joueurs        = EXCLUDED.nb_joueurs;
"""

cur.execute(sql_payrolls)
conn.commit()

# Affichage du bilan
cur.execute("SELECT COUNT(*) FROM team_payrolls;")
total_payrolls = cur.fetchone()[0]
print(f"  -> {total_payrolls} entrées insérées/mises à jour dans 'team_payrolls' ({len(NBA_TEAMS)} équipes x {len(saisons)} saisons).")

# Aperçu pour la saison 2025-26
print("\n=== APERÇU DES PAYROLLS POUR LA SAISON 2025-26 (Top 5) ===")
cur.execute("""
    SELECT 
        tp.team_id,
        t.team_name,
        tp.saison,
        tp.nb_joueurs,
        tp.total_cap_hit,
        tp.total_base_salary
    FROM team_payrolls tp
    JOIN teams t ON t.team_id = tp.team_id
    WHERE tp.saison = '2025-26'
    ORDER BY tp.total_cap_hit DESC
    LIMIT 5;
""")
for row in cur.fetchall():
    team_id, team_name, saison, nb, cap_hit, base_sal = row
    print(f"  • {team_name} (ID: {team_id}) | {saison} | {nb} joueurs | Cap hit: {cap_hit:,}$ | Base: {base_sal:,}$")

cur.close()
conn.close()
print("\n[OK] Terminé avec succès ! Les tables 'teams' et 'team_payrolls' sont maintenant correctement peuplées.")
