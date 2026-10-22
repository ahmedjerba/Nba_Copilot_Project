import psycopg2

conn = psycopg2.connect(
    host="localhost", port=5432,
    dbname="nba_copilot", user="nba_user", password="nba_pass"
)
cur = conn.cursor()

# Récupérer team_id et team_name depuis player_stats via la saison la plus récente
cur.execute("""
    UPDATE players p
    SET team_name = ps.team_abbreviation,
        team_id = ps.team_id
    FROM (
        SELECT DISTINCT ON (player_id) player_id, team_id, team_abbreviation
        FROM player_stats
        ORDER BY player_id, saison DESC
    ) ps
    WHERE p.player_id = ps.player_id
""")

updated = cur.rowcount
conn.commit()
cur.close()
conn.close()
print(f"{updated} joueurs mis à jour avec team_id et team_name récents")