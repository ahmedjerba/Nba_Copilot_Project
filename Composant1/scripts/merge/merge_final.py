import pandas as pd, unicodedata, re
from difflib import get_close_matches
from pathlib import Path

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "ingestion/data/raw"
    FINAL_DIR = PROJECT_ROOT / "ingestion/data/final"
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    

    # --- Chargement ---
    stats     = pd.read_csv(DATA_DIR / "player_stats_merged.csv")
    contracts = pd.read_csv(DATA_DIR / "contracts_raw_final.csv")
    id_map    = pd.read_csv(DATA_DIR / "player_id_map.csv")

    # Exclure saison 2026-27
    contracts = contracts[contracts['saison_nba'] != '2026-27'].copy()

    # --- Ajouter PLAYER_ID aux contrats via la map ---
    contracts = contracts.merge(
        id_map[['spotrac_player_id','PLAYER_ID']],
        on='spotrac_player_id',
        how='left'
    )

    # --- Merger stats + contrats sur PLAYER_ID + saison ---
    df = stats.merge(
        contracts[['PLAYER_ID','saison_nba','base_salary','cap_hit',
                   'years_remaining','contract_value']],
        left_on=['PLAYER_ID','SAISON'],
        right_on=['PLAYER_ID','saison_nba'],
        how='left'
    )

    # --- Stats ---
    print(f"Shape finale : {df.shape}")
    print(f"Avec contrat : {df['base_salary'].notna().sum()} lignes")
    print(f"Sans contrat : {df['base_salary'].isna().sum()} lignes")
    print(f"NaN contrats par colonne :")
    print(df[['base_salary','cap_hit','years_remaining','contract_value']].isna().sum())
    print()
    print(df[['PLAYER_ID','PLAYER_NAME','SAISON','PTS_PER100',
              'valuation_tier','base_salary','cap_hit',
              'years_remaining','contract_value']].head(10).to_string())

    df.to_csv(FINAL_DIR / "player_stats_contracts.csv", index=False)
    print("\nFichier généré : player_stats_contracts.csv")
