# ─────────────────────────────────────────────
# Composant3 / config.py
# Constantes partagées par tous les modules
# ─────────────────────────────────────────────

from dotenv import load_dotenv
from pathlib import Path
import os

# Load .env by traversing up to project root
for p in Path(__file__).resolve().parents:
    if (p / ".env").exists():
        load_dotenv(dotenv_path=p / ".env")
        break
else:
    load_dotenv()

_user = os.getenv('POSTGRES_USER', 'nba_user')
_pass = os.getenv('POSTGRES_PASSWORD', 'nba_pass')
_host = os.getenv('POSTGRES_HOST', 'localhost')
_port = os.getenv('POSTGRES_PORT', '5432')
_db   = os.getenv('POSTGRES_DB', 'nba_copilot')

DB_URL = f"postgresql://{_user}:{_pass}@{_host}:{_port}/{_db}"

FEATURE_VERSION = "v1.0"


SALARY_CAP = {
    "2021-22": 112_414_000,
    "2022-23": 123_655_000,
    "2023-24": 136_021_000,
    "2024-25": 140_588_000,
    "2025-26": 154_647_000,
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