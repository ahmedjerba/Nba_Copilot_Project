from nba_api.stats.endpoints import commonplayerinfo
from nba_api.stats.static import players
import pandas as pd
import time
from pathlib import Path

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "ingestion/data/raw"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Récupérer tous les joueurs actifs
    all_players = players.get_active_players()

    results = []

    for player in all_players:
        try:
            info = commonplayerinfo.CommonPlayerInfo(player_id=player['id'])
            df = info.get_data_frames()[0]
            results.append(df[[
                'PERSON_ID',
                'DISPLAY_FIRST_LAST',
                'BIRTHDATE',
                'POSITION',
                'HEIGHT',
                'WEIGHT',
                'COUNTRY',
                'DRAFT_YEAR',
                'DRAFT_ROUND',
                'DRAFT_NUMBER',
                'TEAM_ID',
                'TEAM_NAME'
            ]].iloc[0])
            time.sleep(0.6)  # délai pour éviter le rate limit
        except Exception as e:
            print(f"Erreur {player['full_name']} ({player['id']}) : {e}")
            continue

    final = pd.DataFrame(results)
    final.to_csv(DATA_DIR / 'common_player_info.csv', index=False)
    print(f"Terminé — {len(final)} joueurs extraits")
