# __init__.py
"""
C3 — Player Valuation & Surplus Engine
Expose les fonctions publiques et outils à C6 (Agent LLM) et C4 (CBA Engine).
"""

import sys
from pathlib import Path

# Assure la présence du dossier racine dans sys.path si exécuté localement
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── Valuation Service (point d'entrée principal & outils Agent) ────
try:
    from .valuation_service import (
        get_player_value,
        get_bulk_valuations,
        CAP_PCT_MAX,
        CAP_PCT_LOW,
        CAP_PCT_TWOWAY,
    )
except (ImportError, ValueError):
    from valuation_service import (
        get_player_value,
        get_bulk_valuations,
        CAP_PCT_MAX,
        CAP_PCT_LOW,
        CAP_PCT_TWOWAY,
    )

# ── Schemas & Contrats d'échange ───────────────────────────────────
try:
    from .schemas import (
        PlayerValueRequest,
        PlayerValueResponse,
        FacteurExplicatif,
    )
except (ImportError, ValueError):
    from schemas import (
        PlayerValueRequest,
        PlayerValueResponse,
        FacteurExplicatif,
    )

# ── Modélisation & Registry (Valorisation & Modèles ML) ─────────────
try:
    from .Valorisation.model import BetaValuationModel
    from .Valorisation.model_registry import (
        load_model,
        save_model,
        list_models,
    )
    from .Valorisation.feature_engineering import (
        build_features,
        NUM_BASE,
        SCORING,
        REBOUNDING,
        PLAYMAKING,
        DEFENSE,
        IMPACT,
        DISPO,
        PROFIL,
        TENDANCE,
        TIRS,
    )
except (ImportError, ValueError):
    from Valorisation.model import BetaValuationModel
    from Valorisation.model_registry import (
        load_model,
        save_model,
        list_models,
    )
    from Valorisation.feature_engineering import (
        build_features,
        NUM_BASE,
        SCORING,
        REBOUNDING,
        PLAYMAKING,
        DEFENSE,
        IMPACT,
        DISPO,
        PROFIL,
        TENDANCE,
        TIRS,
    )

# ── Configuration & Constantes CBA/Valorisation ───────────────────
try:
    from .Valorisation.config import (
        SALARY_CAP,
        SALARY_CAP_REF,
        FEATURE_VERSION,
        SEGMENTS,
        SEASON_WEIGHTS,
        DB_URL,
    )
except (ImportError, ValueError):
    from Valorisation.config import (
        SALARY_CAP,
        SALARY_CAP_REF,
        FEATURE_VERSION,
        SEGMENTS,
        SEASON_WEIGHTS,
        DB_URL,
    )

# ── Évaluation & Backtest ──────────────────────────────────────────
try:
    from .evaluation.metrics import (
        by_age_bucket,
        by_tier,
        coverage,
    )
    from .backtest.walk_forward import run as run_walk_forward
except (ImportError, ValueError):
    from evaluation.metrics import (
        by_age_bucket,
        by_tier,
        coverage,
    )
    from backtest.walk_forward import run as run_walk_forward

__all__ = [
    # valuation_service
    "get_player_value",
    "get_bulk_valuations",
    "CAP_PCT_MAX",
    "CAP_PCT_LOW",
    "CAP_PCT_TWOWAY",
    # schemas
    "PlayerValueRequest",
    "PlayerValueResponse",
    "FacteurExplicatif",
    # model & registry
    "BetaValuationModel",
    "load_model",
    "save_model",
    "list_models",
    "build_features",
    # feature groups & config
    "NUM_BASE",
    "SCORING",
    "REBOUNDING",
    "PLAYMAKING",
    "DEFENSE",
    "IMPACT",
    "DISPO",
    "PROFIL",
    "TENDANCE",
    "TIRS",
    "SALARY_CAP",
    "SALARY_CAP_REF",
    "FEATURE_VERSION",
    "SEGMENTS",
    "SEASON_WEIGHTS",
    "DB_URL",
    # evaluation & backtest
    "by_age_bucket",
    "by_tier",
    "coverage",
    "run_walk_forward",
]
