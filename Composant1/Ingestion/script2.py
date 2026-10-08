import psycopg2

conn = psycopg2.connect(
    host="localhost", port=5432,
    dbname="nba_copilot", user="nba_user", password="nba_pass"
)
cur = conn.cursor()

# Récupérer team_name depuis player_stats via TEAM_ABBREVIATION
cur.execute("""
    UPDATE players p
    SET team_name = ps.team_abbreviation
    FROM (
        SELECT DISTINCT ON (player_id) player_id, team_abbreviation
        FROM player_stats
        ORDER BY player_id, saison DESC
    ) ps
    WHERE p.player_id = ps.player_id
    AND p.team_name IS NULL
""")

updated = cur.rowcount
conn.commit()
cur.close()
conn.close()
print(f"{updated} joueurs mis à jour avec team_name")