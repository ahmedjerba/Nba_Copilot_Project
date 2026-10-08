# Ordre d'exécution des scripts d'ingestion

1. **Spotrac (Contrats & Salaires)**
   - `python scripts/spotrac/collect_salaries.py`
   - `python scripts/spotrac/collect_contract_length.py`
   - `python scripts/spotrac/collect_contract_value.py`
   - `python scripts/spotrac/collect_2021.py`

2. **NBA API (Stats)**
   - `python scripts/nba_api/collect_per100.py`
   - `python scripts/nba_api/collect_pergame.py`
   - `python scripts/nba_api/collect_common_player_info.py`

3. **Merge**
   - `python scripts/merge/merge_contracts.py` (Génère `contracts_raw_final.csv`)
   - `python scripts/merge/merge_stats.py` (Génère `player_stats_merged.csv`)
   - `python scripts/merge/resolve_identity.py` (Génère `player_id_map.csv`)
   - `python scripts/merge/merge_final.py` (Génère `player_stats_contracts.csv`)
