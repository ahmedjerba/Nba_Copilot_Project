"""
agent/tools/propose_moves.py
============================
Outil de génération de propositions de mouvements via le solveur OR-Tools (Composant 4).
"""

from __future__ import annotations
from datetime import datetime
import inspect
from typing import Any, Optional

from agent.schemas.inputs import ProposeMovesInput
from agent.schemas.outputs import (
    ProposeMovesOutput,
    MoveProposal,
    NumericWithSource,
)
from agent.tools.registry import register_tool

try:
    from components.composant4 import composant4_propose_moves
except (ImportError, ModuleNotFoundError):
    def composant4_propose_moves(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Composant 4 non connecté : composant4_propose_moves non implémenté.")


def _to_numeric(val: Any, default_source: str, unit: Optional[str] = None) -> NumericWithSource:
    """Helper pour s'assurer que tout nombre est wrappé dans un NumericWithSource."""
    if isinstance(val, NumericWithSource):
        return val
    if isinstance(val, dict):
        return NumericWithSource(
            value=float(val.get("value", 0.0)),
            source=val.get("source", default_source),
            unit=val.get("unit", unit),
        )
    return NumericWithSource(value=float(val if val is not None else 0.0), source=default_source, unit=unit)


@register_tool(
    name="propose_moves",
    description="Génère une liste de mouvements faisables pour atteindre un objectif sous contraintes CBA. Utilise le solveur OR-Tools du composant 4.",
    input_model=ProposeMovesInput,
)
async def propose_moves(inp: ProposeMovesInput) -> ProposeMovesOutput:
    """
    Appelle le solveur d'optimisation sous contraintes (OR-Tools) du composant 4
    afin de proposer des scénarios de mouvements conformes au CBA maximisant la surplus value.
    """
    raw_res = composant4_propose_moves(
        objective=inp.objective,
        team_id=inp.team_id,
        constraints=inp.constraints,
        exclude_player_ids=inp.exclude_player_ids,
        max_proposals=inp.max_proposals,
    )
    if inspect.isawaitable(raw_res):
        raw_res = await raw_res

    if raw_res is None:
        raise ValueError(f"Aucune proposition générée pour team_id={inp.team_id} (composant 4).")

    if isinstance(raw_res, ProposeMovesOutput):
        return raw_res

    data: dict[str, Any] = raw_res if isinstance(raw_res, dict) else vars(raw_res)

    raw_proposals = data.get("proposals", [])
    proposals: list[MoveProposal] = []
    for prop in raw_proposals:
        p_data = prop if isinstance(prop, dict) else vars(prop)
        proposals.append(
            MoveProposal(
                proposal_id=str(p_data.get("proposal_id", "PROP_001")),
                move_type=str(p_data.get("move_type", "trade")),
                description=str(p_data.get("description", "")),
                players_involved=[int(pid) for pid in p_data.get("players_involved", [])],
                estimated_surplus_gain_m=_to_numeric(
                    p_data.get("estimated_surplus_gain_m", 0.0),
                    default_source="composant4.estimated_surplus_gain",
                    unit="M$",
                ),
                cba_pre_checked=bool(p_data.get("cba_pre_checked", True)),
                cba_legal=p_data.get("cba_legal"),
                confidence_score=_to_numeric(
                    p_data.get("confidence_score", 0.85),
                    default_source="composant4.confidence_score",
                    unit="%",
                ),
                rationale=str(p_data.get("rationale", "")),
            )
        )

    return ProposeMovesOutput(
        tool_name="propose_moves",
        team_id=int(data.get("team_id", inp.team_id)),
        objective_echoed=str(data.get("objective_echoed", inp.objective)),
        proposals=proposals,
        cap_space_available_usd=_to_numeric(
            data.get("cap_space_available_usd", 0.0),
            default_source="composant4.cap_space_available",
            unit="USD",
        ),
        timestamp_utc=datetime.utcnow(),
    )
