# __init__.py
"""
C5 — Comparable Engine
Expose les fonctions publiques à C6 (Agent LLM).
"""

# ── Comparable Service (point d'entrée principal) ──────────────────
from comparable_service import (
    find,
    ComparableRequest,
    ComparableResponse,
    ComparableItem,
    ComparableMode,
    ARCHETYPE_CONFIDENCE,
    CONFIDENCE_WARNINGS,
)

# ── Role Engine ────────────────────────────────────────────────────
from role_engine import (
    build_and_save_roles,
    predict_archetype,
    get_player_archetype,
    get_archetype_players,
    get_archetype_stats,
    list_archetypes,
)

# ── Similarity ────────────────────────────────────────────────────
from similarity import (
    find_similar_players,
    find_historical_comparables,
    ComparableResult,
)

# ── Validator ─────────────────────────────────────────────────────
from validator import (
    run_backtest,
    run_backtest_by_archetype,
    run_backtest_soft_vs_hard,
)

__all__ = [
    # comparable_service
    "find",
    "ComparableRequest",
    "ComparableResponse",
    "ComparableItem",
    "ComparableMode",
    "ARCHETYPE_CONFIDENCE",
    "CONFIDENCE_WARNINGS",
    # role_engine
    "build_and_save_roles",
    "predict_archetype",
    "get_player_archetype",
    "get_archetype_players",
    "get_archetype_stats",
    "list_archetypes",
    # similarity
    "find_similar_players",
    "find_historical_comparables",
    "ComparableResult",
    # validator
    "run_backtest",
    "run_backtest_by_archetype",
    "run_backtest_soft_vs_hard",
]
