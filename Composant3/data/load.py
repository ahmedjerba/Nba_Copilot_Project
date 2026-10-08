import pandas as pd
from sqlalchemy import create_engine, text
from Composant3.config import DB_URL, FEATURE_VERSION

def load_training_pairs() -> pd.DataFrame:
    engine = create_engine(DB_URL)
    query = text("""
        SELECT *
        FROM training_pairs
        WHERE feature_version = :version
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"version": FEATURE_VERSION})

    df["season_year"] = df["saison"].str[:4].astype(int)
    return df