"""
agent/tools/find_comparables.py
===============================
Outil de recherche de profils de joueurs comparables historiques via KNN (Composant 5).
"""

from __future__ import annotations
from datetime import datetime
import inspect
from typing import Any, Optional

from agent.schemas.inputs import FindComparablesInput
from agent.schemas.outputs import (
    FindComparablesOutput,
    ComparablePlayer,
    NumericWithSource,
)
from agent.tools.registry import register_tool

try:
    from components.composant5 import composant5_find_comparables
except (ImportError, ModuleNotFoundError):
    def composant5_find_comparables(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Composant 5 non connecté : composant5_find_comparables non implémenté.")


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
    name="find_comparables",
    description="Trouve les joueurs historiquement similaires à un joueur donné au même âge. Utilise le KNN du composant 5.",
    input_model=FindComparablesInput,
)
async def find_comparables(inp: FindComparablesInput) -> FindComparablesOutput:
    """
    Appelle le modèle KNN du composant 5 pour identifier les joueurs NBA
    historiquement les plus proches d'un joueur cible au même âge.
    """
    raw_res = composant5_find_comparables(
        player_id=inp.player_id,
        n_comparables=inp.n_comparables,
        same_position_only=inp.same_position_only,
        active_only=inp.active_only,
        season=inp.season,
    )
    if inspect.isawaitable(raw_res):
        raw_res = await raw_res

    if raw_res is None:
        raise ValueError(f"Aucun comparable trouvé pour player_id={inp.player_id} (composant 5).")

    if isinstance(raw_res, FindComparablesOutput):
        return raw_res

    data: dict[str, Any] = raw_res if isinstance(raw_res, dict) else vars(raw_res)

    raw_comparables = data.get("comparables", [])
    comparables: list[ComparablePlayer] = []
    for comp in raw_comparables:
        c_data = comp if isinstance(comp, dict) else vars(comp)

        war_val = c_data.get("war_at_comparison")
        war_numeric = (
            _to_numeric(war_val, default_source="composant3.war", unit="wins")
            if war_val is not None
            else None
        )

        salary_val = c_data.get("salary_at_comparison_usd")
        salary_numeric = (
            _to_numeric(salary_val, default_source="players.salary", unit="USD")
            if salary_val is not None
            else None
        )

        season_ref = str(c_data.get("season_reference", "2023-24"))

        comparables.append(
            ComparablePlayer(
                player_id=int(c_data.get("player_id", 0)),
                full_name=str(c_data.get("full_name", "")),
                similarity_score=_to_numeric(
                    c_data.get("similarity_score", 0.0),
                    default_source="composant5.similarity_score",
                    unit="%",
                ),
                age_at_comparison=_to_numeric(
                    c_data.get("age_at_comparison", 25.0),
                    default_source="players.age",
                    unit="years",
                ),
                position=str(c_data.get("position", "N/A")),
                war_at_comparison=war_numeric,
                salary_at_comparison_usd=salary_numeric,
                season_reference=season_ref,
                key_similarities=list(c_data.get("key_similarities", [])),
                key_differences=list(c_data.get("key_differences", [])),
            )
        )

    return FindComparablesOutput(
        tool_name="find_comparables",
        reference_player_id=int(data.get("reference_player_id", inp.player_id)),
        reference_player_name=str(data.get("reference_player_name", f"Player #{inp.player_id}")),
        comparables=comparables,
        method_used=str(data.get("method_used", "KNN-Composant5")),
        timestamp_utc=datetime.utcnow(),
    )
