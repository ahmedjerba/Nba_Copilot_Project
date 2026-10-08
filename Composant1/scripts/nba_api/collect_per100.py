import json, time, hashlib
from pathlib import Path
import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
    CACHE = PROJECT_ROOT / "ingestion/cache/cache_nba_api"
    CACHE.mkdir(parents=True, exist_ok=True)
    DATA_DIR = PROJECT_ROOT / "ingestion/data/raw"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    

    CACHE = CACHE
    CACHE.mkdir(exist_ok=True)
    DELAI = 1.5
    TIMEOUT = 30
    SAISONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]

    # ===== Colonnes décidées ensemble =====
    COLONNES_PER100 = [
        "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "AGE",
        "GP", "W", "L", "W_PCT", "MIN",
        "PTS", "REB", "AST", "STL", "BLK", "BLKA", "TOV", "PF", "PFD",
        "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA", "FT_PCT",
        "OREB", "DREB", "PLUS_MINUS", "TEAM_COUNT",
    ]
    COLONNES_ADVANCED = [
        "PLAYER_ID", "TEAM_COUNT",
        "OFF_RATING", "DEF_RATING", "NET_RATING", "PACE", "USG_PCT", "TM_TOV_PCT",
        "AST_PCT", "AST_TO", "AST_RATIO", "OREB_PCT", "DREB_PCT", "REB_PCT",
        "EFG_PCT", "TS_PCT", "PIE", "POSS",
        "FGM", "FGA", "FG_PCT",   # gardées seulement pour croiser/vérifier avec per100
    ]

    def appel(nom, saison, per_mode, measure_type):
        """Appelle LeagueDashPlayerStats avec cache disque, retourne un DataFrame brut."""
        cle = hashlib.md5(f"{nom}_{saison}_{per_mode}_{measure_type}".encode()).hexdigest()[:10]
        fichier = CACHE / f"{nom}_{saison}_{cle}.json"
        if fichier.exists():
            data = json.loads(fichier.read_text())
        else:
            time.sleep(DELAI)
            try:
                data = leaguedashplayerstats.LeagueDashPlayerStats(
                    timeout=TIMEOUT, season=saison,
                    per_mode_detailed=per_mode,
                    measure_type_detailed_defense=measure_type,
                ).get_dict()
            except Exception as e:
                print(f"  ERREUR {nom} {saison}: {type(e).__name__}: {str(e)[:120]}")
                return None
            fichier.write_text(json.dumps(data))
        rs = data["resultSets"][0]
        return pd.DataFrame(rs["rowSet"], columns=rs["headers"])

    # ===== Collecte per100 =====
    morceaux_per100 = []
    for saison in SAISONS:
        print(f"per100 {saison}...")
        df = appel("per100", saison, "Per100Possessions", "Base")
        if df is None:
            continue
        df = df[COLONNES_PER100].copy()
        df.insert(0, "SAISON", saison)
        morceaux_per100.append(df)
    table_per100 = pd.concat(morceaux_per100, ignore_index=True)

    # ===== Collecte advanced =====
    morceaux_advanced = []
    for saison in SAISONS:
        print(f"advanced {saison}...")
        df = appel("advanced", saison, "PerGame", "Advanced")
        if df is None:
            continue
        df = df[COLONNES_ADVANCED].copy()
        df.insert(0, "SAISON", saison)
        morceaux_advanced.append(df)
    table_advanced = pd.concat(morceaux_advanced, ignore_index=True)

    # ===== Fusion en une seule table =====
    # On fusionne sur (SAISON, PLAYER_ID). Les colonnes FGM/FGA/FG_PCT d'advanced
    # sont renommées avec un suffixe _ADV pour vérification, sans écraser celles de per100.
    table_advanced_pour_fusion = table_advanced.rename(columns={
        "FGM": "FGM_ADV", "FGA": "FGA_ADV", "FG_PCT": "FG_PCT_ADV",
    })

    table_finale = table_per100.merge(
        table_advanced_pour_fusion,
        on=["SAISON", "PLAYER_ID", "TEAM_COUNT"],
        how="left",   # on garde toutes les lignes per100, même si advanced manque pour l'une d'elles
    )

    table_finale = table_finale.sort_values(["SAISON", "PLAYER_NAME"]).reset_index(drop=True)

    print(f"\nTable finale : {len(table_finale)} lignes, {len(table_finale.columns)} colonnes")
    print("Saisons présentes :", sorted(table_finale['SAISON'].unique()))

    # Contrôle de cohérence : les tirs per100 vs advanced doivent être proches (pas identiques,
    # unités différentes : per100 vs total saison), juste pour repérer une vraie anomalie.
    manquants_advanced = table_finale["OFF_RATING"].isna().sum()
    print("Lignes sans données advanced (à vérifier si > 0) :", manquants_advanced)

    table_finale.to_csv(DATA_DIR / "nba_api_complet.csv", index=False)
    print("\nEnregistré : nba_api_complet.csv")
