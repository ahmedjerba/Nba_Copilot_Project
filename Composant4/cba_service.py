# Composant4/cba_service.py
"""
NBA Front-Office Copilot — Composant 4
Service CBA exposé à l'agent LLM.

Responsabilités :
- Wrapper propre autour de engine.py et rules.py
- Validation des inputs (Pydantic)
- Formatage des outputs pour l'agent
- Gestion centralisée des erreurs
- Logging structuré
"""

from __future__ import annotations

import logging
import traceback
from typing import Any, Optional

from pydantic import BaseModel, Field, validator

from Composant4.engine import EngineResult, propose_moves
from Composant4.ruleees import is_legal
from Composant4.schemas import (
    Contract,
    ExceptionType,
    Move,
    MoveResult,
    MoveStatus,
    MoveType,
    TeamPayroll,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Input schemas (ce que l'agent envoie)
# ---------------------------------------------------------------------------

class ProposeMovesInput(BaseModel):
    """Input de l'outil propose_moves."""

    team_id: int = Field(
        ...,
        description="ID de l'équipe NBA dans la DB (ex: 1610612738 pour Boston)",
        gt=0,
    )
    saison: str = Field(
        default="2025-26",
        description="Saison au format 'YYYY-YY' (ex: '2025-26')",
        pattern=r"^\d{4}-\d{2}$",
    )
    top_n: int = Field(
        default=10,
        description="Nombre de moves à retourner (max 25)",
        ge=1,
        le=25,
    )
    include_trades: bool = Field(default=True,  description="Inclure les trades")
    include_signs:  bool = Field(default=True,  description="Inclure les signatures")
    include_waives: bool = Field(default=True,  description="Inclure les libérations")

    @validator("saison")
    def validate_saison(cls, v: str) -> str:
        parts = v.split("-")
        if len(parts) != 2:
            raise ValueError("Format saison invalide. Attendu: 'YYYY-YY'")
        year_start = int(parts[0])
        year_end   = int(parts[1])
        if year_end != (year_start + 1) % 100:
            raise ValueError(
                f"Saison incohérente: {v}. "
                f"L'année de fin doit être l'année de début + 1."
            )
        return v


class CheckMoveInput(BaseModel):
    """Input de l'outil check_move (vérification d'un move spécifique)."""

    team_id: int = Field(..., description="ID de l'équipe", gt=0)
    saison:  str = Field(default="2025-26", pattern=r"^\d{4}-\d{2}$")

    # State de l'équipe passé explicitement par l'agent
    total_cap_hit:     int   = Field(..., description="Masse salariale actuelle en $", ge=0)
    total_base_salary: int   = Field(..., description="Salaire de base total en $",   ge=0)
    nb_joueurs:        int   = Field(..., description="Nombre de joueurs au roster",   ge=0, le=20)
    mle_used:          bool  = Field(default=False)
    bi_annual_used:    bool  = Field(default=False)

    # Move à évaluer
    move_type:    str           = Field(..., description="TRADE | SIGN | WAIVE")
    salary_out:   int           = Field(default=0, ge=0)
    salary_in:    int           = Field(default=0, ge=0)
    salary_offer: Optional[int] = Field(default=None, ge=0)
    exception:    str           = Field(
        default="NONE",
        description="NONE | MLE_NON_TAXPAYER | MLE_TAXPAYER | BI_ANNUAL | ROOM_MLE",
    )
    description:  str = Field(default="Move custom", max_length=500)

    # Joueurs impliqués (simplifiés)
    players_out: list[dict[str, Any]] = Field(default_factory=list)
    players_in:  list[dict[str, Any]] = Field(default_factory=list)

    # Picks impliqués
    picks_out: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Picks envoyés (ex: [{'round': 1, 'protection': None, 'years_out': 1}])",
    )
    picks_in: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Picks reçus",
    )
    value_delta: int = Field(
        default=0,
        description="Delta de valeur estimé en $ (positif = gain)",
    )

    @validator("move_type")
    def validate_move_type(cls, v: str) -> str:
        valid = {m.value for m in MoveType}
        if v.upper() not in valid:
            raise ValueError(f"move_type invalide: '{v}'. Valeurs valides: {valid}")
        return v.upper()

    @validator("exception")
    def validate_exception(cls, v: str) -> str:
        valid = {e.value for e in ExceptionType}
        if v.upper() not in valid:
            raise ValueError(f"exception invalide: '{v}'. Valeurs valides: {valid}")
        return v.upper()


class GetTeamStateInput(BaseModel):
    """Input pour récupérer l'état financier d'une équipe."""
    team_id: int = Field(..., gt=0)
    saison:  str = Field(default="2025-26", pattern=r"^\d{4}-\d{2}$")


# ---------------------------------------------------------------------------
# Output schemas (ce que l'agent reçoit)
# ---------------------------------------------------------------------------

class MoveOutput(BaseModel):
    """Représentation d'un move pour l'agent."""
    rank:            int
    status:          str
    description:     str
    cap_hit_after:   int
    cap_space_after: int
    value_delta:     int
    value_signal:    str
    blocking_rule:   Optional[str]
    blocking_reason: Optional[str]
    rule_results:    list[dict[str, Any]]

    @classmethod
    def from_result(cls, result: MoveResult, rank: int) -> "MoveOutput":
        return cls(
            rank            = rank,
            status          = result.status.value,
            description     = result.move.description,
            cap_hit_after   = result.cap_hit_after,
            cap_space_after = result.cap_space_after,
            value_delta     = result.value_delta,
            value_signal    = result.value_signal,
            blocking_rule   = result.blocking_rule,
            blocking_reason = result.blocking_reason,
            rule_results    = [
                {
                    "rule":   r.rule,
                    "passed": r.passed,
                    "reason": r.reason,
                }
                for r in result.rule_results
            ],
        )


class ProposeMovesOutput(BaseModel):
    """Output de propose_moves pour l'agent."""
    success:          bool
    team_id:          int
    saison:           str
    total_cap_hit:    int
    cap_space:        int
    nb_joueurs:       int
    total_candidates: int
    legal_count:      int
    illegal_count:    int
    top_moves:        list[MoveOutput]
    warnings:         list[str]
    error:            Optional[str] = None

    def to_agent_text(self) -> str:
        """Résumé lisible pour l'agent LLM."""
        if not self.success:
            return f"❌ Erreur : {self.error}"

        lines = [
            f"📊 Analyse CBA — Team {self.team_id} | {self.saison}",
            f"   Cap hit actuel : {self.total_cap_hit:,}$",
            f"   Cap space      : {self.cap_space:,}$",
            f"   Joueurs roster : {self.nb_joueurs}",
            f"   Candidats      : {self.total_candidates} "
            f"({self.legal_count} légaux / {self.illegal_count} illégaux)",
            "",
            f"🏆 Top {len(self.top_moves)} moves recommandés :",
        ]

        for m in self.top_moves:
            lines.append(
                f"  {m.rank:>2}. [{m.value_signal}] {m.description}"
            )
            lines.append(
                f"      Cap après : {m.cap_hit_after:,}$ "
                f"| Space : {m.cap_space_after:,}$"
            )

        if self.warnings:
            lines.append("\n⚠️  Avertissements :")
            lines.extend(f"   - {w}" for w in self.warnings)

        return "\n".join(lines)


class CheckMoveOutput(BaseModel):
    """Output de check_move pour l'agent."""
    success:         bool
    status:          str
    description:     str
    cap_hit_after:   int
    cap_space_after: int
    value_delta:     int
    value_signal:    str
    blocking_rule:   Optional[str]
    blocking_reason: Optional[str]
    rule_results:    list[dict[str, Any]]
    error:           Optional[str] = None

    def to_agent_text(self) -> str:
        if not self.success:
            return f"❌ Erreur : {self.error}"

        icon   = "✅" if self.status == "LEGAL" else "❌"
        lines  = [
            f"{icon} Move : {self.description}",
            f"   Statut         : {self.status}",
            f"   Cap après      : {self.cap_hit_after:,}$",
            f"   Space après    : {self.cap_space_after:,}$",
            f"   Valeur delta   : {self.value_signal}",
        ]

        if self.blocking_reason:
            lines.append(f"   ⛔ Blocage      : {self.blocking_reason}")

        lines.append("\n   Détail des règles :")
        for r in self.rule_results:
            icon_r = "✅" if r["passed"] else "❌"
            lines.append(f"   {icon_r} {r['rule']} — {r['reason']}")

        return "\n".join(lines)


class TeamStateOutput(BaseModel):
    """Output de get_team_state pour l'agent."""
    success:           bool
    team_id:           int
    saison:            str
    total_cap_hit:     int
    total_base_salary: int
    nb_joueurs:        int
    cap_space:         int
    tax_space:         int
    apron1_space:      int
    apron2_space:      int
    status:            str   # "under_cap" | "over_cap" | "taxpayer" | "apron1" | "apron2"
    mle_available:     str
    error:             Optional[str] = None

    def to_agent_text(self) -> str:
        if not self.success:
            return f"❌ Erreur : {self.error}"

        lines = [
            f"💰 État financier — Team {self.team_id} | {self.saison}",
            f"   Statut         : {self.status.upper()}",
            f"   Cap hit total  : {self.total_cap_hit:,}$",
            f"   Cap space      : {self.cap_space:,}$",
            f"   Tax space      : {self.tax_space:,}$",
            f"   Apron 1 space  : {self.apron1_space:,}$",
            f"   Apron 2 space  : {self.apron2_space:,}$",
            f"   Joueurs roster : {self.nb_joueurs}",
            f"   MLE dispo      : {self.mle_available}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Service functions (outils exposés à l'agent)
# ---------------------------------------------------------------------------

def tool_propose_moves(raw_input: dict[str, Any]) -> ProposeMovesOutput:
    """
    Outil agent — propose les meilleurs moves CBA pour une équipe.

    Usage agent :
        tool_propose_moves({
            "team_id": 1610612738,
            "saison": "2025-26",
            "top_n": 5,
        })
    """
    try:
        inp = ProposeMovesInput(**raw_input)
    except Exception as e:
        return ProposeMovesOutput(
            success=False, team_id=raw_input.get("team_id", 0),
            saison=raw_input.get("saison", "?"),
            total_cap_hit=0, cap_space=0, nb_joueurs=0,
            total_candidates=0, legal_count=0, illegal_count=0,
            top_moves=[], warnings=[], error=f"Input invalide : {e}",
        )

    try:
        result: EngineResult = propose_moves(
            team_id        = inp.team_id,
            saison         = inp.saison,
            top_n          = inp.top_n,
            include_trades = inp.include_trades,
            include_signs  = inp.include_signs,
            include_waives = inp.include_waives,
        )

        from Composant4.cba_loader import load_cba_rules
        rules     = load_cba_rules(inp.saison)
        cap       = rules["thresholds"]["salary_cap"]
        cap_space = cap - result.team_state.total_cap_hit

        top_moves = [
            MoveOutput.from_result(m, rank=i + 1)
            for i, m in enumerate(result.best_moves)
        ]

        return ProposeMovesOutput(
            success          = True,
            team_id          = inp.team_id,
            saison           = inp.saison,
            total_cap_hit    = result.team_state.total_cap_hit,
            cap_space        = cap_space,
            nb_joueurs       = result.team_state.nb_joueurs,
            total_candidates = result.total_candidates,
            legal_count      = len(result.legal_moves),
            illegal_count    = len(result.illegal_moves),
            top_moves        = top_moves,
            warnings         = result.warnings,
        )

    except ValueError as e:
        logger.warning("tool_propose_moves ValueError: %s", e)
        return ProposeMovesOutput(
            success=False, team_id=inp.team_id, saison=inp.saison,
            total_cap_hit=0, cap_space=0, nb_joueurs=0,
            total_candidates=0, legal_count=0, illegal_count=0,
            top_moves=[], warnings=[], error=str(e),
        )
    except Exception as e:
        logger.error("tool_propose_moves erreur inattendue: %s", traceback.format_exc())
        return ProposeMovesOutput(
            success=False, team_id=inp.team_id, saison=inp.saison,
            total_cap_hit=0, cap_space=0, nb_joueurs=0,
            total_candidates=0, legal_count=0, illegal_count=0,
            top_moves=[], warnings=[], error=f"Erreur interne : {e}",
        )


def tool_check_move(raw_input: dict[str, Any]) -> CheckMoveOutput:
    """
    Outil agent — vérifie si un move spécifique est légal selon le CBA.

    Usage agent :
        tool_check_move({
            "team_id": 1610612738,
            "saison": "2025-26",
            "total_cap_hit": 145_000_000,
            "total_base_salary": 140_000_000,
            "nb_joueurs": 14,
            "move_type": "SIGN",
            "salary_in": 14_104_000,
            "exception": "MLE_NON_TAXPAYER",
            "description": "Signer via MLE non-taxpayer",
        })
    """
    try:
        inp = CheckMoveInput(**raw_input)
    except Exception as e:
        return CheckMoveOutput(
            success=False, status="ERROR", description="?",
            cap_hit_after=0, cap_space_after=0,
            value_delta=0, value_signal="?",
            blocking_rule=None, blocking_reason=None,
            rule_results=[], error=f"Input invalide : {e}",
        )

    try:
        # Reconstituer TeamPayroll
        state = TeamPayroll(
            team_id           = inp.team_id,
            saison            = inp.saison,
            total_cap_hit     = inp.total_cap_hit,
            total_base_salary = inp.total_base_salary,
            nb_joueurs        = inp.nb_joueurs,
            mle_used          = inp.mle_used,
            bi_annual_used    = inp.bi_annual_used,
        )

        # Reconstituer les contrats
        def _to_contract(d: dict) -> Contract:
            return Contract(
                player_id       = d.get("player_id", 0),
                player_name     = d.get("player_name", "Inconnu"),
                saison          = inp.saison,
                cap_hit         = d.get("cap_hit", 0),
                base_salary     = d.get("base_salary", 0),
                years_remaining = d.get("years_remaining", 1),
                contract_value  = d.get("contract_value", 0),
            )

        # Reconstituer le Move
        move = Move(
            move_type    = MoveType(inp.move_type),
            players_out  = [_to_contract(p) for p in inp.players_out],
            players_in   = [_to_contract(p) for p in inp.players_in],
            salary_out   = inp.salary_out,
            salary_in    = inp.salary_in,
            salary_delta = inp.salary_in - inp.salary_out,
            salary_offer = inp.salary_offer,
            exception    = ExceptionType(inp.exception),
            value_delta  = inp.value_delta,
            description  = inp.description,
            picks_in     = inp.picks_in,
            picks_out    = inp.picks_out,
        )

        result: MoveResult = is_legal(move, state, inp.saison)

        from Composant4.cba_loader import load_cba_rules
        rules     = load_cba_rules(inp.saison)
        cap       = rules["thresholds"]["salary_cap"]
        cap_space = cap - result.cap_hit_after

        return CheckMoveOutput(
            success         = True,
            status          = result.status.value,
            description     = result.move.description,
            cap_hit_after   = result.cap_hit_after,
            cap_space_after = cap_space,
            value_delta     = result.value_delta,
            value_signal    = result.value_signal,
            blocking_rule   = result.blocking_rule,
            blocking_reason = result.blocking_reason,
            rule_results    = [
                {
                    "rule":   r.rule,
                    "passed": r.passed,
                    "reason": r.reason,
                }
                for r in result.rule_results
            ],
        )

    except Exception as e:
        logger.error("tool_check_move erreur: %s", traceback.format_exc())
        return CheckMoveOutput(
            success=False, status="ERROR", description=inp.description,
            cap_hit_after=0, cap_space_after=0,
            value_delta=0, value_signal="?",
            blocking_rule=None, blocking_reason=None,
            rule_results=[], error=f"Erreur interne : {e}",
        )


def tool_get_team_state(raw_input: dict[str, Any]) -> TeamStateOutput:
    """
    Outil agent — retourne l'état financier complet d'une équipe.

    Usage agent :
        tool_get_team_state({"team_id": 1610612738, "saison": "2025-26"})
    """
    try:
        inp = GetTeamStateInput(**raw_input)
    except Exception as e:
        return TeamStateOutput(
            success=False, team_id=raw_input.get("team_id", 0),
            saison=raw_input.get("saison", "?"),
            total_cap_hit=0, total_base_salary=0, nb_joueurs=0,
            cap_space=0, tax_space=0, apron1_space=0, apron2_space=0,
            status="error", mle_available="N/A",
            error=f"Input invalide : {e}",
        )

    try:
        from Composant4.engine import _load_team_state
        from Composant4.cba_loader import load_cba_rules

        state = _load_team_state(inp.team_id, inp.saison)
        rules = load_cba_rules(inp.saison)
        thres = rules["thresholds"]

        cap    = thres["salary_cap"]
        tax    = thres["luxury_tax"]
        apron1 = thres.get("first_apron", 0)
        apron2 = thres.get("second_apron", 0)
        hit    = state.total_cap_hit

        cap_space    = cap    - hit
        tax_space    = tax    - hit
        apron1_space = apron1 - hit
        apron2_space = apron2 - hit

        # Statut de l'équipe
        if hit > apron2:
            status = "apron2"
        elif hit > apron1:
            status = "apron1"
        elif hit > tax:
            status = "taxpayer"
        elif hit > cap:
            status = "over_cap"
        else:
            status = "under_cap"

        # MLE disponible
        if hit > apron1:
            mle_available = "Aucun (apron 1 dépassé)"
        elif hit >= tax:
            mle_available = f"MLE taxpayer ({rules['exceptions']['mle_taxpayer']:,}$)"
        else:
            mle_available = (
                f"MLE non-taxpayer ({rules['exceptions']['mle_non_taxpayer']:,}$) "
                f"+ BAE ({rules['exceptions']['bi_annual']:,}$)"
            )

        return TeamStateOutput(
            success           = True,
            team_id           = inp.team_id,
            saison            = inp.saison,
            total_cap_hit     = hit,
            total_base_salary = state.total_base_salary,
            nb_joueurs        = state.nb_joueurs,
            cap_space         = cap_space,
            tax_space         = tax_space,
            apron1_space      = apron1_space,
            apron2_space      = apron2_space,
            status            = status,
            mle_available     = mle_available,
        )

    except Exception as e:
        logger.error("tool_get_team_state erreur: %s", traceback.format_exc())
        return TeamStateOutput(
            success=False, team_id=inp.team_id, saison=inp.saison,
            total_cap_hit=0, total_base_salary=0, nb_joueurs=0,
            cap_space=0, tax_space=0, apron1_space=0, apron2_space=0,
            status="error", mle_available="N/A",
            error=f"Erreur interne : {e}",
        )


# ---------------------------------------------------------------------------
# Registry — catalogue des outils pour l'agent
# ---------------------------------------------------------------------------

CBA_TOOLS: dict[str, dict[str, Any]] = {
    "propose_moves": {
        "fn":          tool_propose_moves,
        "description": (
            "Propose les meilleurs moves CBA (trades, signatures, libérations) "
            "pour une équipe NBA donnée. Retourne les moves triés par value_delta."
        ),
        "input_schema": ProposeMovesInput.schema(),
    },
    "check_move": {
        "fn":          tool_check_move,
        "description": (
            "Vérifie si un move CBA spécifique est légal selon les règles NBA 2025-26. "
            "Retourne le détail de chaque règle vérifiée."
        ),
        "input_schema": CheckMoveInput.schema(),
    },
    "get_team_state": {
        "fn":          tool_get_team_state,
        "description": (
            "Retourne l'état financier complet d'une équipe : cap hit, cap space, "
            "statut (under cap / taxpayer / apron), MLE disponible."
        ),
        "input_schema": GetTeamStateInput.schema(),
    },
}


def dispatch(tool_name: str, raw_input: dict[str, Any]) -> str:
    """
    Point d'entrée unique pour l'agent — dispatche vers le bon outil
    et retourne toujours du texte lisible.

    Args:
        tool_name : "propose_moves" | "check_move" | "get_team_state"
        raw_input : dict d'arguments bruts depuis l'agent

    Returns:
        Texte formaté prêt à être injecté dans le contexte de l'agent.
    """
    if tool_name not in CBA_TOOLS:
        available = list(CBA_TOOLS.keys())
        return (
            f"❌ Outil '{tool_name}' inconnu. "
            f"Outils disponibles : {available}"
        )

    logger.info("dispatch → %s | input: %s", tool_name, raw_input)
    result = CBA_TOOLS[tool_name]["fn"](raw_input)
    return result.to_agent_text()