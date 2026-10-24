"""
agent/tools/get_player_value.py
================================
Wrapper de l'outil get_player_value autour du composant 3.
Appelle valuation_service.get_player_value et traduit le
PlayerValueResponse en GetPlayerValueOutput (schéma agent).
"""

from __future__ import annotations

from Composant3.valuation_service import get_player_value as composant3_get_player_value

from agent.tools.registry import register_tool
from agent.schemas.inputs import GetPlayerValueInput
from agent.schemas.outputs import (
    GetPlayerValueOutput,
    ContractDetail,
    ValueProjection,
    NumericWithSource,
)


@register_tool(
    name="get_player_value",
    description=(
        "Retourne la fiche complète de valorisation d'un joueur : "
        "contrat, surplus value, projections saison par saison."
    ),
    input_model=GetPlayerValueInput,
)
async def get_player_value(inp: GetPlayerValueInput) -> GetPlayerValueOutput:
    """
    Appelle composant3.valuation_service.get_player_value puis
    traduit le PlayerValueResponse en GetPlayerValueOutput.
    Toute valeur numérique est wrappée dans NumericWithSource.
    Lève une exception si le composant 3 retourne une erreur —
    le registry la convertit en ToolError.
    """

    season = inp.season or "2024-25"

    # --- Appel composant 3 ---
    resp = composant3_get_player_value(
        player_id=inp.player_id,
        saison=season,
    )

    # Erreur métier retournée par le composant 3
    if resp.erreur and resp.valeur_dollars is None:
        raise ValueError(f"Composant3 erreur : {resp.erreur}")

    # --- Contrat ---
    # Le composant 3 expose cap_hit mais pas le détail du contrat.
    # On construit un ContractDetail minimal depuis ce qu'on a.
    contract = ContractDetail(
        salary_usd=NumericWithSource(
            value=float(resp.cap_hit),
            source="composant3.contracts.cap_hit",
            unit="USD",
        ),
        years_remaining=NumericWithSource(
            value=0.0,   # non exposé par valuation_service — à enrichir si besoin
            source="composant3.contracts.years_remaining",
        ),
        contract_type="unknown",  # non exposé — à enrichir depuis contracts_raw
    )

    # --- Surplus value ---
    current_surplus = NumericWithSource(
        value=float(resp.surplus) if resp.surplus is not None else 0.0,
        source="composant3.valuation_service.surplus",
        unit="USD",
    )

    # --- WAR / valeur marché comme proxy ---
    war_current = NumericWithSource(
        value=float(resp.valeur_pct),   # valeur en % du cap = meilleure proxy disponible
        source="composant3.valuation_service.valeur_pct",
        unit="%cap",
    ) if resp.valeur_pct is not None else None

    # --- Intervalle de confiance p10/p90 comme projection unique ---
    # Le composant 3 ne retourne pas de projections multi-saisons ici.
    # On expose p10/p90 comme une seule projection "saison courante".
    projections: list[ValueProjection] = []
    if inp.include_projections and resp.p10_dollars is not None:
        projections.append(ValueProjection(
            season=season,
            projected_war=NumericWithSource(
                value=float(resp.valeur_pct or 0.0),
                source="composant3.valuation_service.valeur_pct",
                unit="%cap",
            ),
            projected_salary_usd=NumericWithSource(
                value=float(resp.valeur_dollars),
                source="composant3.valuation_service.valeur_dollars",
                unit="USD",
            ),
            surplus_value_m=NumericWithSource(
                value=round(float(resp.surplus or 0.0) / 1_000_000, 2),
                source="composant3.valuation_service.surplus",
                unit="M$",
            ),
        ))

    return GetPlayerValueOutput(
        player_id=inp.player_id,
        full_name=resp.player_name or f"Joueur #{inp.player_id}",
        position=resp.position if hasattr(resp, "position") else "N/A",
        age=NumericWithSource(
            value=0.0,   # non exposé directement par PlayerValueResponse
            source="composant3.players.age",
        ),
        contract=contract,
        current_surplus_value_m=NumericWithSource(
            value=round(float(resp.surplus or 0.0) / 1_000_000, 2),
            source="composant3.valuation_service.surplus",
            unit="M$",
        ),
        war_current_season=war_current,
        projections=projections,
        percentile_rank=None,   # non exposé par valuation_service
        season=season,
    )