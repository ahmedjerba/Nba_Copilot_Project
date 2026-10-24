"""
agent/tools/search_players.py
=============================
Outil de recherche multicritère et de valorisation des joueurs (Composant 3).
"""

from __future__ import annotations
from datetime import datetime
import inspect
from typing import Any, Optional

from agent.schemas.inputs import SearchPlayersInput
from agent.schemas.outputs import (
    SearchPlayersOutput,
    PlayerSummary,
    NumericWithSource,
)
from agent.tools.registry import register_tool

try:
    from Composant3 import composant3_search_players
except (ImportError, ModuleNotFoundError):
    def composant3_search_players(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Composant 3 non connecté : composant3_search_players non implémenté.")


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
    name="search_players",
    description="Recherche des joueurs NBA par filtres (poste, salaire, âge, surplus value). Retourne une liste triée de joueurs avec leur valorisation.",
    input_model=SearchPlayersInput,
)
async def search_players(inp: SearchPlayersInput) -> SearchPlayersOutput:
    """
    Appelle le composant 3 pour filtrer et valoriser une liste de joueurs NBA
    selon les critères (poste, tranche d'âge, salaire, surplus value, recherche textuelle).
    """
    positions_filter = (
        [p.value if hasattr(p, "value") else str(p) for p in inp.positions]
        if inp.positions
        else None
    )
    sort_by_str = inp.sort_by.value if hasattr(inp.sort_by, "value") else str(inp.sort_by)

    raw_res = composant3_search_players(
        query=inp.query,
        team_ids=inp.team_ids,
        positions=positions_filter,
        age_min=inp.age_min,
        age_max=inp.age_max,
        salary_min_usd=inp.salary_min_usd,
        salary_max_usd=inp.salary_max_usd,
        min_surplus_value=inp.min_surplus_value,
        sort_by=sort_by_str,
        limit=inp.limit,
    )
    if inspect.isawaitable(raw_res):
        raw_res = await raw_res

    if raw_res is None:
        raise ValueError("Aucun résultat retourné par le moteur de recherche (composant 3).")

    if isinstance(raw_res, SearchPlayersOutput):
        return raw_res

    data: dict[str, Any] = raw_res if isinstance(raw_res, dict) else vars(raw_res)

    raw_player_list = data.get("results", [])
    results: list[PlayerSummary] = []
    for item in raw_player_list:
        p_data = item if isinstance(item, dict) else vars(item)

        war_val = p_data.get("war")
        war_numeric = (
            _to_numeric(war_val, default_source="composant3.war", unit="wins")
            if war_val is not None
            else None
        )

        results.append(
            PlayerSummary(
                player_id=int(p_data.get("player_id", 0)),
                full_name=str(p_data.get("full_name", "")),
                team_abbreviation=str(p_data.get("team_abbreviation", "N/A")),
                position=str(p_data.get("position", "N/A")),
                age=_to_numeric(p_data.get("age", 25.0), default_source="players.age", unit="years"),
                salary_usd=_to_numeric(
                    p_data.get("salary_usd", 0.0),
                    default_source="players.salary_2024_25",
                    unit="USD",
                ),
                surplus_value_m=_to_numeric(
                    p_data.get("surplus_value_m", 0.0),
                    default_source="composant3.surplus_value",
                    unit="M$",
                ),
                war=war_numeric,
            )
        )

    # Construction de l'interprétation de la requête si non fournie
    query_interpreted = str(
        data.get("query_interpreted")
        or f"Filtres: query='{inp.query}', positions={positions_filter}, sort={sort_by_str}, limit={inp.limit}"
    )

    return SearchPlayersOutput(
        tool_name="search_players",
        results=results,
        total_found=int(data.get("total_found", len(results))),
        query_interpreted=query_interpreted,
        timestamp_utc=datetime.utcnow(),
    )
