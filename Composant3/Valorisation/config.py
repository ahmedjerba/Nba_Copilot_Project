# ─────────────────────────────────────────────
# Composant3 / config.py
# Constantes partagées par tous les modules
# ─────────────────────────────────────────────

from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

DB_URL = (
    f"postgresql://{os.getenv('POSTGRES_USER')}"
    f":{os.getenv('POSTGRES_PASSWORD')}"
    f"@{os.getenv('POSTGRES_HOST')}"
    f":{os.getenv('POSTGRES_PORT')}"
    f"/{os.getenv('POSTGRES_DB')}"
)

FEATURE_VERSION = "v1.0"


SALARY_CAP = {
    "2021-22": 112_414_000,
    "2022-23": 123_655_000,
    "2023-24": 136_021_000,
    "2024-25": 140_588_000,
}

SALARY_CAP_REF = 140_588_000   # cap de référence pour affichage en $

# Segments de valorisation
SEGMENTS = {
    "rookie_minimum": (0.000, 0.050),
    "middle_class":   (0.050, 0.250),
    "max":            (0.250, 1.000),
}

# Pondération temporelle des saisons d'entraînement
SEASON_WEIGHTS = {
    "2021-22": 1.0,
    "2022-23": 1.5,
    "2023-24": 2.0,
    "2024-25": 2.5,
}