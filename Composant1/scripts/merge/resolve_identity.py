import pandas as pd, unicodedata, re
from difflib import get_close_matches
from pathlib import Path

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "ingestion/data/raw"
    

    stats = pd.read_csv(DATA_DIR / "player_stats_merged.csv")
    contracts = pd.read_csv(DATA_DIR / "contracts_raw_final.csv")
    contracts = contracts[contracts['saison_nba'] != '2026-27'].copy()

    def normalize(name):
        name = str(name).lower().strip()
        name = unicodedata.normalize('NFD', name)
        name = ''.join(c for c in name if unicodedata.category(c) != 'Mn')
        name = re.sub(r'\b(jr|sr|ii|iii|iv)\b\.?', '', name)
        name = re.sub(r'[^a-z ]', '', name)
        name = re.sub(r'\s+', ' ', name).strip()
        return name

    stats['name_norm'] = stats['PLAYER_NAME'].apply(normalize)
    contracts['name_norm'] = contracts['player_name'].apply(normalize)

    stats_map = stats[['PLAYER_ID','PLAYER_NAME','name_norm']].drop_duplicates('name_norm').set_index('name_norm')
    contracts_map = contracts[['spotrac_player_id','player_name','name_norm']].drop_duplicates('name_norm').set_index('name_norm')

    stats_names = set(stats_map.index)
    contracts_names = set(contracts_map.index)

    # --- Corrections manuelles connues ---
    manual_fixes = {
        'cameron christie':    'cam christie',
        'cameron thomas':      'cam thomas',
        'herb jones':          'herbert jones',
        'ishmail wainright':   'ish wainright',
        'louis williams':      'lou williams',
        'nicolas claxton':     'nic claxton',
        'rj nembhard':         'ryan nembhard',
        'ron holland':         'ronald holland',
        'vincent williams':    'vince williams',
        # noms trop différents pour le fuzzy — correspondances manuelles
        'dj stewart':          'dj stewart',        # absent des stats = pas de minutes
        'ishmael smith':       'ishmael smith',     # idem
        'jeenathan williams':  'jeenathan williams',
        'kenyon martin':       'kenyon martin',
        'luca vildoza':        'luca vildoza',
        'mohamed bamba':       'mo bamba',
        'nahshon hyland':      'bones hyland',
        'sviatoslav mykhailiuk': 'svi mykhailiuk',
        'thomas sorber':       'thomas sorber',
    }

    # --- Construire la map ---
    rows = []

    # Match direct
    for name in stats_names & contracts_names:
        rows.append({
            'name_norm':         name,
            'player_name_stats': stats_map.loc[name, 'PLAYER_NAME'],
            'player_name_spotrac': contracts_map.loc[name, 'player_name'],
            'PLAYER_ID':         stats_map.loc[name, 'PLAYER_ID'],
            'spotrac_player_id': contracts_map.loc[name, 'spotrac_player_id'],
            'match_type':        'direct',
        })

    # Corrections manuelles/fuzzy
    for spotrac_name, stats_name in manual_fixes.items():
        if spotrac_name not in contracts_map.index:
            continue
        if stats_name not in stats_map.index:
            # Pas dans les stats = joueur sans minutes, on garde quand même le lien
            rows.append({
                'name_norm':           spotrac_name,
                'player_name_stats':   stats_name,
                'player_name_spotrac': contracts_map.loc[spotrac_name, 'player_name'],
                'PLAYER_ID':           None,
                'spotrac_player_id':   contracts_map.loc[spotrac_name, 'spotrac_player_id'],
                'match_type':          'no_stats',
            })
            continue
        rows.append({
            'name_norm':           spotrac_name,
            'player_name_stats':   stats_map.loc[stats_name, 'PLAYER_NAME'],
            'player_name_spotrac': contracts_map.loc[spotrac_name, 'player_name'],
            'PLAYER_ID':           stats_map.loc[stats_name, 'PLAYER_ID'],
            'spotrac_player_id':   contracts_map.loc[spotrac_name, 'spotrac_player_id'],
            'match_type':          'manual',
        })

    id_map = pd.DataFrame(rows)

    print(f"Total mappings : {len(id_map)}")
    print(f"Match types :\n{id_map['match_type'].value_counts()}")
    print(f"Sans PLAYER_ID (pas de stats) : {id_map['PLAYER_ID'].isna().sum()}")

    id_map.to_csv(DATA_DIR / "player_id_map.csv", index=False)
    print("\nFichier généré : player_id_map.csv")
