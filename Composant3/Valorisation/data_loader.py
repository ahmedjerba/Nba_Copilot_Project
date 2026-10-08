# ─────────────────────────────────────────────
# Composant3 / data_loader.py
# Charge et assemble le dataset d'entraînement
# ─────────────────────────────────────────────

import pandas as pd
from sqlalchemy import create_engine, text

from Composant3.Valorisation.config import DB_URL, FEATURE_VERSION, SALARY_CAP, SEASON_WEIGHTS


def load_pairs(segment: str = "middle_class") -> pd.DataFrame:
    """
    Retourne le dataset joueur-saison avec cap_hit_pct comme cible.

    segment : "middle_class" (0.05-0.25), "all", ou "max" (>0.25)

    Chaque ligne = features saison N  +  cap_hit saison N
    Join direct : on valorise un joueur avec ses stats de la même saison
    (le contrat signé l'été N reflète les stats de la saison N).
    """
    engine = create_engine(DB_URL)

    with engine.connect() as conn:
        df_features = pd.read_sql(text("""
            SELECT pf.*, p.position, p.draft_year, p.draft_round, p.draft_number
            FROM player_features pf
            LEFT JOIN players p ON pf.player_id = p.player_id
            WHERE pf.feature_version = :v
        """), conn, params={"v": FEATURE_VERSION})

        df_contracts = pd.read_sql(text("SELECT * FROM contracts"), conn)

    # ── Normaliser la saison du contrat ────────────────────────────────────
    # contracts.saison est au format "2022-23" (saison en cours du contrat)
    # player_features.saison est au format "2021-22" (saison jouée)
    # Le contrat signé à l'été 2022 porte sur la saison 2022-23
    # → on veut l'apparier aux stats de 2021-22 (la saison précédente)
    # saison_key = "2021-22" quand contrat.saison = "2022-23"
    df_contracts["saison_key"] = df_contracts["saison"].apply(
        lambda s: f"{int(s[:4]) - 1}-{s[:4][2:]}"
    )
    df_contracts["salary_cap"] = df_contracts["saison_key"].map(SALARY_CAP)
    df_contracts["cap_hit_pct"] = df_contracts["cap_hit"] / df_contracts["salary_cap"]

    # ── Filtre de segment ──────────────────────────────────────────────────
    if segment == "middle_class":
        mask = (
            (df_contracts["cap_hit_pct"] >= 0.05) &
            (df_contracts["cap_hit_pct"] <= 0.25) &
            (df_contracts["salary_cap"].notna())
        )
    elif segment == "max":
        mask = (
            (df_contracts["cap_hit_pct"] > 0.25) &
            (df_contracts["salary_cap"].notna())
        )
    else:  # "all"
        mask = df_contracts["salary_cap"].notna()

    contracts_filtered = df_contracts[mask].copy()

    # ── Jointure features ↔ contrat ───────────────────────────────────────
    pairs = df_features.merge(
        contracts_filtered[[
            "player_id", "saison_key", "cap_hit_pct", "cap_hit",
            "salary_cap", "years_remaining"
        ]],
        left_on=["player_id", "saison"],
        right_on=["player_id", "saison_key"],
        how="inner"
    )

    # ── Pondération temporelle ─────────────────────────────────────────────
    pairs["sample_weight"] = pairs["saison"].map(SEASON_WEIGHTS).fillna(1.0)

    # ── draft_position : rang absolu dans la draft ─────────────────────────
    # draft_number est dans players ; on le récupère directement
    # (draft_round et draft_number sont déjà dans le SELECT ci-dessus)
    pairs["draft_position"] = pairs.apply(
        lambda r: (r["draft_round"] - 1) * 30 + r["draft_number"]
        if pd.notna(r["draft_round"]) and pd.notna(r["draft_number"])
        else None,
        axis=1
    )

    pairs = pairs.drop(columns=["saison_key"])

    print(f"[data_loader] segment={segment!r}  →  {len(pairs)} paires")
    print(f"  Par saison :\n{pairs['saison'].value_counts().sort_index().to_string()}")
    print(f"  cap_hit_pct  mean={pairs['cap_hit_pct'].mean():.4f}  "
          f"std={pairs['cap_hit_pct'].std():.4f}  "
          f"min={pairs['cap_hit_pct'].min():.4f}  "
          f"max={pairs['cap_hit_pct'].max():.4f}")

    return pairs