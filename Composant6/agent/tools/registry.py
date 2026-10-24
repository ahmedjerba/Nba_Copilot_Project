"""
agent/tools/registry.py
=======================
Point d'entrée unique entre la boucle d'exécution et les outils.
"""

from __future__ import annotations
import asyncio
from typing import Callable, Any
from pydantic import BaseModel, ValidationError

from agent.schemas.inputs import (
    SearchPlayersInput, GetPlayerValueInput, CheckMoveInput,
    ProposeMovesInput, FindComparablesInput, SearchRiskNewsInput,
)
from agent.schemas.outputs import ToolOutput, ToolError


# ---------------------------------------------------------------------------
# Structure d'un outil enregistré
# ---------------------------------------------------------------------------

class ToolDefinition(BaseModel):
    """
    Ce que le registry sait de chaque outil.
    `json_schema` est ce qu'on envoie au LLM pour qu'il sache quoi appeler.
    """
    name:        str
    description: str
    input_model: type[BaseModel]   # le modèle Pydantic d'entrée
    json_schema: dict              # généré automatiquement depuis input_model

    class Config:
        arbitrary_types_allowed = True


# ---------------------------------------------------------------------------
# Décorateur d'enregistrement
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, tuple[ToolDefinition, Callable]] = {}

def register_tool(
    name: str,
    description: str,
    input_model: type[BaseModel],
):
    """
    Décorateur qui enregistre une fonction comme outil.

    Usage :
        @register_tool(
            name="search_players",
            description="Recherche des joueurs par filtres.",
            input_model=SearchPlayersInput,
        )
        async def search_players(inp: SearchPlayersInput) -> SearchPlayersOutput:
            ...
    """
    def decorator(fn: Callable) -> Callable:
        definition = ToolDefinition(
            name=name,
            description=description,
            input_model=input_model,
            # Pydantic génère le JSON Schema — c'est ce qui part au LLM
            json_schema=input_model.model_json_schema(),
        )
        _REGISTRY[name] = (definition, fn)
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Dispatcher — appelé par agent.py
# ---------------------------------------------------------------------------

async def call_tool(tool_name: str, raw_params: dict[str, Any]) -> ToolOutput:
    """
    Valide les paramètres, appelle l'outil, retourne le résultat ou un ToolError.

    `raw_params` vient directement du LLM — c'est du JSON brut,
    on ne lui fait pas confiance avant validation Pydantic.
    """
    # 1. L'outil existe ?
    if tool_name not in _REGISTRY:
        return ToolError(
            tool_name=tool_name,
            error_code="UNKNOWN_TOOL",
            message=f"Outil inconnu : {tool_name}",
            retryable=False,
        )

    definition, fn = _REGISTRY[tool_name]

    # 2. Les paramètres sont valides ?
    try:
        validated_input = definition.input_model(**raw_params)
    except ValidationError as e:
        return ToolError(
            tool_name=tool_name,
            error_code="INVALID_INPUT",
            message=str(e),
            retryable=False,   # un input invalide ne se corrige pas tout seul
        )

    # 3. Appel de l'outil (sync ou async)
    try:
        if asyncio.iscoroutinefunction(fn):
            result = await fn(validated_input)
        else:
            result = fn(validated_input)
        return result

    except Exception as e:
        return ToolError(
            tool_name=tool_name,
            error_code="UPSTREAM_ERROR",
            message=str(e),
            retryable=True,    # erreur réseau ou DB → on peut retenter
        )


# ---------------------------------------------------------------------------
# Accès aux schémas — pour le prompt système du planificateur
# ---------------------------------------------------------------------------

def get_all_schemas() -> list[dict]:
    """
    Retourne la liste des JSON Schemas de tous les outils enregistrés.
    C'est ce bloc qu'on injecte dans le prompt du planificateur LLM.
    """
    return [
        {
            "name":        defn.name,
            "description": defn.description,
            "parameters":  defn.json_schema,
        }
        for defn, _ in _REGISTRY.values()
    ]

def get_tool_names() -> list[str]:
    """Liste des noms d'outils disponibles."""
    return list(_REGISTRY.keys())