"""
agent/schemas/agent_response.py
================================
Contrat de sortie de l'agent — les 5 types de cartes pour le composant 7.

Aucun chiffre brut ici : les cartes référencent des IDs,
le composant 7 résout les données depuis les ToolOutputs loggés.
"""

from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Annotated, Literal, Optional, Union
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ResponseType(str, Enum):
    PLAYER_CARD           = "player_card"
    TRADE_PROPOSAL        = "trade_proposal"
    RISK_ALERT            = "risk_alert"
    CLARIFICATION_REQUEST = "clarification_request"
    REFUSAL               = "refusal"

class RefusalReason(str, Enum):
    OUT_OF_SCOPE        = "out_of_scope"
    INSUFFICIENT_DATA   = "insufficient_data"
    CBA_BLOCKING        = "cba_blocking"
    HALLUCINATION_RISK  = "hallucination_risk"
    AMBIGUOUS_INTENT    = "ambiguous_intent"
    ETHICAL             = "ethical"

class ConfidenceLevel(str, Enum):
    HIGH   = "high"    # toutes les données sources dispo, CBA vérifié
    MEDIUM = "medium"  # données partielles ou projection utilisée
    LOW    = "low"     # estimation qualitative uniquement


# ---------------------------------------------------------------------------
# Metadata commune
# ---------------------------------------------------------------------------

class ResponseMetadata(BaseModel):
    """Traçabilité complète vers Langfuse."""
    agent_run_id:       str
    tool_call_ids:      list[str]   # IDs des appels qui ont produit les données
    planner_model:      str
    synthesizer_model:  str
    total_tokens_used:  int
    latency_ms:         int
    confidence:         ConfidenceLevel
    confidence_rationale: str       # le synthétiseur doit justifier, pas juste noter
    timestamp_utc:      datetime    = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# 1. PlayerCard
# ---------------------------------------------------------------------------

class PlayerCardHighlight(BaseModel):
    label:     str                                      # 'Valeur contractuelle'
    insight:   str  = Field(max_length=300)
    sentiment: Literal["positive", "neutral", "negative"]

class PlayerCard(BaseModel):
    response_type:       Literal[ResponseType.PLAYER_CARD] = ResponseType.PLAYER_CARD
    player_id:           int
    player_name:         str
    executive_summary:   str   = Field(max_length=600)  # prose pour le GM, sans chiffre inventé
    highlights:          list[PlayerCardHighlight] = Field(max_length=5)
    comparable_player_ids: list[int] = Field(default_factory=list)  # références, pas données
    risk_summary:        Optional[str] = Field(default=None, max_length=300)
    recommended_action:  Optional[str] = None
    metadata:            ResponseMetadata


# ---------------------------------------------------------------------------
# 2. TradeProposal
# ---------------------------------------------------------------------------

class TradeLegSummary(BaseModel):
    team_id:            int
    team_abbreviation:  str
    players_out_ids:    list[int]
    players_in_ids:     list[int]
    cap_impact_summary: str   # phrase descriptive, les chiffres sont dans CheckMoveOutput

class TradeProposal(BaseModel):
    response_type:          Literal[ResponseType.TRADE_PROPOSAL] = ResponseType.TRADE_PROPOSAL
    proposal_id:            str           # référence MoveProposal dans ProposeMovesOutput
    headline:               str  = Field(max_length=150)
    rationale:              str  = Field(max_length=800)
    legs:                   list[TradeLegSummary] = Field(min_length=2)
    cba_status:             Literal["legal", "illegal", "unchecked"]
    cba_notes:              list[str]     = Field(default_factory=list)
    strategic_fit_assessment: str         = Field(max_length=400)
    risks:                  list[str]     = Field(default_factory=list)
    metadata:               ResponseMetadata


# ---------------------------------------------------------------------------
# 3. RiskAlert
# ---------------------------------------------------------------------------

class RiskAlert(BaseModel):
    response_type:      Literal[ResponseType.RISK_ALERT] = ResponseType.RISK_ALERT
    player_ids:         list[int]
    player_names:       list[str]
    alert_headline:     str  = Field(max_length=200)
    alert_body:         str  = Field(max_length=1000)
    severity:           Literal["low", "medium", "high", "critical"]
    categories:         list[str]
    recommended_action: Optional[str] = None
    source_urls:        list[str]     = Field(default_factory=list)  # issues de SearchRiskNewsOutput
    metadata:           ResponseMetadata


# ---------------------------------------------------------------------------
# 4. ClarificationRequest
# ---------------------------------------------------------------------------

class ClarificationOption(BaseModel):
    key:   str   # 'A', 'B', '1'…
    label: str   = Field(max_length=150)

class ClarificationRequest(BaseModel):
    response_type:  Literal[ResponseType.CLARIFICATION_REQUEST] = ResponseType.CLARIFICATION_REQUEST
    question:       str  = Field(max_length=400)
    context:        str  = Field(max_length=300)   # pourquoi cette clarification est nécessaire
    options:        list[ClarificationOption] = Field(default_factory=list)
    blocking_step:  str                            # quel outil est bloqué
    metadata:       ResponseMetadata


# ---------------------------------------------------------------------------
# 5. Refusal
# ---------------------------------------------------------------------------

class Refusal(BaseModel):
    response_type:         Literal[ResponseType.REFUSAL] = ResponseType.REFUSAL
    reason:                RefusalReason
    explanation:           str  = Field(max_length=500)
    hard_stop:             bool  # True = définitif | False = conditionnel
    suggested_alternatives: list[str] = Field(default_factory=list)
    metadata:              ResponseMetadata


# ---------------------------------------------------------------------------
# Union discriminée — point d'entrée unique pour le composant 7
# ---------------------------------------------------------------------------

AgentResponse = Annotated[
    Union[
        PlayerCard,
        TradeProposal,
        RiskAlert,
        ClarificationRequest,
        Refusal,
    ],
    Field(discriminator="response_type"),
]