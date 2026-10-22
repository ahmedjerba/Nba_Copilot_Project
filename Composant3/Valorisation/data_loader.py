# ─────────────────────────────────────────────
# Composant3 / data_loader.py
# Charge et assemble le dataset d'entraînement
# ─────────────────────────────────────────────

import pandas as pd
from sqlalchemy import create_engine, text

from Composant3.Valorisation.config import DB_URL, FEATURE_VERSION, SALARY_CAP, SEASON_WEIGHTS


def load_pairs(segment: str = "market") -> pd.DataFrame:
    """
    Retourne le dataset joueur-saison avec cap_hit_pct comme cible.

    segment :
        "market"       (0.001-0.30) ← défaut — vétérans + role players + middle + stars
        "middle_class" (0.05-0.25)
        "max"          (>0.30)      ← non modélisable, usage analytique seulement
        "all"          (tout)

    Exclusions systématiques (tous segments) :
        - Rookies    : saisons_experience <= 1
        - Two-way    : cap_hit_pct < 0.001
    """
    engine = create_engine(DB_URL)

    with engine.connect() as conn:
        df_features = pd.read_sql(text("""
            SELECT pf.*, p.player_name, p.position, p.draft_year, p.draft_round, p.draft_number, p.team_name
            FROM player_features pf
            LEFT JOIN players p ON pf.player_id = p.player_id
            WHERE pf.feature_version = :v
        """), conn, params={"v": FEATURE_VERSION})

        df_contracts = pd.read_sql(text("SELECT * FROM contracts"), conn)

    # ── Normaliser la saison du contrat ────────────────────────────────────
    # CORRECTION C3-1 : mapping direct sur la saison d'exécution du contrat
    # (suppression du décalage -1 an qui détruisait 2021-22 et faussait tous les cap_hit_pct)
    df_contracts["salary_cap"] = df_contracts["saison"].map(SALARY_CAP)
    df_contracts["cap_hit_pct"] = df_contracts["cap_hit"] / df_contracts["salary_cap"]

    # ── Filtre de segment ──────────────────────────────────────────────────
    if segment == "market":
        mask = (
            (df_contracts["cap_hit_pct"] >= 0.001) &
            (df_contracts["cap_hit_pct"] <= 0.30) &
            (df_contracts["salary_cap"].notna())
        )
    elif segment == "middle_class":
        mask = (
            (df_contracts["cap_hit_pct"] >= 0.05) &
            (df_contracts["cap_hit_pct"] <= 0.25) &
            (df_contracts["salary_cap"].notna())
        )
    elif segment == "max":
        mask = (
            (df_contracts["cap_hit_pct"] > 0.30) &
            (df_contracts["salary_cap"].notna())
        )
    else:  # "all"
        mask = df_contracts["salary_cap"].notna()

    contracts_filtered = df_contracts[mask].copy()

    # ── Jointure features ↔ contrat ───────────────────────────────────────
    pairs = df_features.merge(
        contracts_filtered[[
            "player_id", "saison", "cap_hit_pct", "cap_hit",
            "salary_cap", "years_remaining"
        ]],
        on=["player_id", "saison"],
        how="inner"
    )

    # ── Pondération temporelle ─────────────────────────────────────────────
    pairs["sample_weight"] = pairs["saison"].map(SEASON_WEIGHTS).fillna(1.0)

    # ── draft_position et expérience réelle ────────────────────────────────
    pairs["draft_position"] = pairs.apply(
        lambda r: (r["draft_round"] - 1) * 30 + r["draft_number"]
        if pd.notna(r["draft_round"]) and pd.notna(r["draft_number"])
        else None,
        axis=1
    )

    # CORRECTION : calcul de l'expérience réelle à la volée (sans toucher la BDD)
    # Évite le bug du cumcount qui mettait saisons_experience = 0 pour tout le monde en 2021-22
    season_start = pairs["saison"].str[:4].astype(int)
    real_exp = season_start - pairs["draft_year"]
    # Fallback pour joueurs non draftés : estimation via l'âge (entrée vers 20 ans)
    pairs["saisons_experience"] = real_exp.fillna(
        (pairs["age_at_date"] - 20).clip(lower=0)
    ).clip(lower=0)

    # ── Exclusions systématiques pour l'entraînement ───────────────────────
    # Si segment == 'all' (inférence globale), on conserve tous les profils (superstars, rookies, minimums)
    n_avant = len(pairs)
    n_twoway = 0
    n_rookie = 0

    if segment != "all":
        # Two-way contracts (< 0.1% du cap)
        mask_twoway = pairs["cap_hit_pct"] < 0.001
        pairs = pairs[~mask_twoway].copy()
        n_twoway = int(mask_twoway.sum())

        # Vrais Rookies (uniquement ceux draftés l'année même ou jeunes < 21 ans à exp 0)
        mask_rookie = (pairs["draft_year"] == season_start) | (
            pairs["draft_year"].isna() & (pairs["age_at_date"] < 21) & (pairs["saisons_experience"] == 0)
        )
        n_rookie = int(mask_rookie.sum())
        pairs = pairs[~mask_rookie].copy()

    print(f"[data_loader] segment={segment!r}  ->  {len(pairs)} paires")
    print(f"  Exclusions : {n_rookie} rookies  |  {n_twoway} two-way  "
          f"(retire : {n_avant - len(pairs)})")
    print(f"  Par saison :\n{pairs['saison'].value_counts().sort_index().to_string()}")
    print(f"  cap_hit_pct  mean={pairs['cap_hit_pct'].mean():.4f}  "
          f"std={pairs['cap_hit_pct'].std():.4f}  "
          f"min={pairs['cap_hit_pct'].min():.4f}  "
          f"max={pairs['cap_hit_pct'].max():.4f}")

    return pairs