"""
agent/tools/check_move.py
========================
Outil de vérification de la conformité CBA d'un mouvement (Composant 4).
"""

from __future__ import annotations
from datetime import datetime
import inspect
from typing import Any, Optional

from agent.schemas.inputs import CheckMoveInput
from agent.schemas.outputs import (
    CheckMoveOutput,
    CBAViolation,
    SalaryMatchingDetail,
    NumericWithSource,
)
from agent.tools.registry import register_tool

try:
    from components.composant4 import composant4_check_move
except (ImportError, ModuleNotFoundError):
    def composant4_check_move(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Composant 4 non connecté : composant4_check_move non implémenté.")


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
    name="check_move",
    description="Vérifie la conformité CBA d’un mouvement (trade, signature, waive). Retourne légal/illégal avec motif détaillé.",
    input_model=CheckMoveInput,
)
async def check_move(inp: CheckMoveInput) -> CheckMoveOutput:
    """
    Appelle le moteur de règles CBA du composant 4 pour tester la légalité
    d'un mouvement (trade, signature, waive, extension) et évaluer son impact salary cap / tablier.
    """
    move_type_str = inp.move_type.value if hasattr(inp.move_type, "value") else str(inp.move_type)
    players_out_payload = [
        {"player_id": leg.player_id, "team_id": leg.team_id}
        for leg in inp.players_out
    ]
    players_in_payload = [
        {"player_id": leg.player_id, "team_id": leg.team_id}
        for leg in inp.players_in
    ]

    raw_res = composant4_check_move(
        move_type=move_type_str,
        players_out=players_out_payload,
        players_in=players_in_payload,
        cash_considerations_usd=inp.cash_considerations_usd,
        target_team_id=inp.target_team_id,
        proposed_salary_usd=inp.proposed_salary_usd,
        proposed_years=inp.proposed_years,
    )
    if inspect.isawaitable(raw_res):
        raw_res = await raw_res

    if raw_res is None:
        raise ValueError("Aucune réponse retournée par le vérificateur CBA (composant 4).")

    if isinstance(raw_res, CheckMoveOutput):
        return raw_res

    data: dict[str, Any] = raw_res if isinstance(raw_res, dict) else vars(raw_res)

    # Violations CBA
    raw_violations = data.get("violations", [])
    violations: list[CBAViolation] = []
    for viol in raw_violations:
        v_data = viol if isinstance(viol, dict) else vars(viol)
        violations.append(
            CBAViolation(
                rule_id=str(v_data.get("rule_id", "CBA_RULE")),
                description=str(v_data.get("description", "")),
                severity=str(v_data.get("severity", "blocking")),
                remediation_hint=v_data.get("remediation_hint"),
            )
        )

    # Détails matching salarial
    raw_matching = data.get("salary_matching", [])
    salary_matching: list[SalaryMatchingDetail] = []
    for match in raw_matching:
        m_data = match if isinstance(match, dict) else vars(match)
        salary_matching.append(
            SalaryMatchingDetail(
                team_id=int(m_data.get("team_id", 0)),
                team_abbreviation=str(m_data.get("team_abbreviation", "N/A")),
                salary_out_usd=_to_numeric(
                    m_data.get("salary_out_usd", 0.0),
                    default_source="composant4.salary_out",
                    unit="USD",
                ),
                salary_in_usd=_to_numeric(
                    m_data.get("salary_in_usd", 0.0),
                    default_source="composant4.salary_in",
                    unit="USD",
                ),
                delta_usd=_to_numeric(
                    m_data.get("delta_usd", 0.0),
                    default_source="composant4.salary_delta",
                    unit="USD",
                ),
                compliant=bool(m_data.get("compliant", True)),
            )
        )

    # Impact cap par équipe
    raw_cap_impact = data.get("cap_impact_by_team", {})
    cap_impact_by_team: dict[str, NumericWithSource] = {}
    for team_abbr, impact_val in raw_cap_impact.items():
        cap_impact_by_team[team_abbr] = _to_numeric(
            impact_val,
            default_source="composant4.cap_impact",
            unit="USD",
        )

    # Statut tablier
    apron_status = data.get("apron_status_after")
    apron_status_after: Optional[dict[str, str]] = (
        {str(k): str(v) for k, v in apron_status.items()}
        if apron_status is not None
        else None
    )

    return CheckMoveOutput(
        tool_name="check_move",
        move_type=str(data.get("move_type", move_type_str)),
        is_legal=bool(data.get("is_legal", False)),
        violations=violations,
        salary_matching=salary_matching,
        cap_impact_by_team=cap_impact_by_team,
        apron_status_after=apron_status_after,
        notes=list(data.get("notes", [])),
        timestamp_utc=datetime.utcnow(),
    )
