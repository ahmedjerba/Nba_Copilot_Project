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
MIN_TARGET_POSS = 300