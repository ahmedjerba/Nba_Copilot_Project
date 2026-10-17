# __init__.py
"""
C4 — CBA Engine & Trade Optimizer
Expose les fonctions publiques et outils à C6 (Agent LLM).
"""

import sys
from pathlib import Path

# Assure la présence du dossier racine dans sys.path si exécuté localement
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── CBA Service (Outils agent LLM & dispatch) ──────────────────────
try:
    from .cba_service import (
        tool_propose_moves,
        tool_check_move,
        tool_get_team_state,
        dispatch,
        CBA_TOOLS,
        ProposeMovesInput,
        ProposeMovesOutput,
        CheckMoveInput,
        CheckMoveOutput,
        GetTeamStateInput,
        TeamStateOutput,
        MoveOutput,
    )
except (ImportError, ValueError):
    from cba_service import (
        tool_propose_moves,
        tool_check_move,
        tool_get_team_state,
        dispatch,
        CBA_TOOLS,
        ProposeMovesInput,
        ProposeMovesOutput,
        CheckMoveInput,
        CheckMoveOutput,
        GetTeamStateInput,
        TeamStateOutput,
        MoveOutput,
    )

# ── Engine (Génération de moves & Modes avancés) ───────────────────
try:
    from .engine import (
        propose_moves,
        propose_trade_for_target,
        propose_trades_with_team,
        EngineResult,
        RosterPlayer,
        PICK_VALUES,
    )
except (ImportError, ValueError):
    from engine import (
        propose_moves,
        propose_trade_for_target,
        propose_trades_with_team,
        EngineResult,
        RosterPlayer,
        PICK_VALUES,
    )

# ── Optimizer (Solveur CP-SAT OR-Tools) ────────────────────────────
try:
    from .optimizer import optimize_trades
except (ImportError, ValueError):
    from optimizer import optimize_trades

# ── Rules & Validation (Règles CBA NBA 2025-26) ─────────────────────
try:
    from .ruleees import (
        is_legal,
        check_salary_cap,
        check_trade_matching,
        check_luxury_tax,
        check_exceptions,
        check_roster_size,
    )
except (ImportError, ValueError):
    from ruleees import (
        is_legal,
        check_salary_cap,
        check_trade_matching,
        check_luxury_tax,
        check_exceptions,
        check_roster_size,
    )

# ── CBA Loader (Règles & seuils financiers YAML) ───────────────────
try:
    from .cba_loader import (
        load_cba_rules,
        clear_cache,
        list_available_saisons,
        RULES_DIR,
    )
except (ImportError, ValueError):
    from cba_loader import (
        load_cba_rules,
        clear_cache,
        list_available_saisons,
        RULES_DIR,
    )

# ── Schemas & Modèles CBA ──────────────────────────────────────────
try:
    from .schemas import (
        MoveType,
        ExceptionType,
        MoveStatus,
        PickRound,
        Contract,
        DraftPick,
        TeamPayroll,
        Move,
        RuleResult,
        MoveResult,
    )
except (ImportError, ValueError):
    from schemas import (
        MoveType,
        ExceptionType,
        MoveStatus,
        PickRound,
        Contract,
        DraftPick,
        TeamPayroll,
        Move,
        RuleResult,
        MoveResult,
    )

__all__ = [
    # cba_service
    "tool_propose_moves",
    "tool_check_move",
    "tool_get_team_state",
    "dispatch",
    "CBA_TOOLS",
    "ProposeMovesInput",
    "ProposeMovesOutput",
    "CheckMoveInput",
    "CheckMoveOutput",
    "GetTeamStateInput",
    "TeamStateOutput",
    "MoveOutput",
    # engine
    "propose_moves",
    "propose_trade_for_target",
    "propose_trades_with_team",
    "EngineResult",
    "RosterPlayer",
    "PICK_VALUES",
    # optimizer
    "optimize_trades",
    # rules
    "is_legal",
    "check_salary_cap",
    "check_trade_matching",
    "check_luxury_tax",
    "check_exceptions",
    "check_roster_size",
    # cba_loader
    "load_cba_rules",
    "clear_cache",
    "list_available_saisons",
    "RULES_DIR",
    # schemas
    "MoveType",
    "ExceptionType",
    "MoveStatus",
    "PickRound",
    "Contract",
    "DraftPick",
    "TeamPayroll",
    "Move",
    "RuleResult",
    "MoveResult",
]
