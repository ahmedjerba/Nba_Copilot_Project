"""
agent/schemas/outputs.py
========================
Schémas de sortie des 6 outils.

GUARDRAIL CENTRAL : tout champ numérique porte un champ source.
Le checker refusera toute synthèse avec un NumericWithSource sans source.
"""

from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Utilitaires partagés
# ---------------------------------------------------------------------------

class ToolError(BaseModel):
    """Retour d'erreur unifié. Le planificateur décide de retry ou escalader."""
    tool_name:  str
    error_code: str    # NOT_FOUND | INVALID_INPUT | UPSTREAM_TIMEOUT | …
    message:    str
    retryable:  bool          = Field(default=False)
    timestamp_utc: datetime   = Field(default_factory=datetime.utcnow)


class NumericWithSource(BaseModel):
    """
    Brique centrale du guardrail.
    Tout chiffre qui peut atterrir dans la réponse finale doit passer par là.
    """
    value:  float
    source: str   = Field(description="Ex: 'players.salary_2024_25', 'composant3.surplus_value'")
    unit:   Optional[str] = Field(default=None)  # 'USD', 'M$', 'wins', '%'


# ---------------------------------------------------------------------------
# 1. SearchPlayersOutput
# ---------------------------------------------------------------------------

class PlayerSummary(BaseModel):
    player_id:          int
    full_name:          str
    team_abbreviation:  str
    position:           str
    age:                NumericWithSource
    salary_usd:         NumericWithSource
    surplus_value_m:    NumericWithSource
    war:                Optional[NumericWithSource] = None

class SearchPlayersOutput(BaseModel):
    tool_name:         str      = "search_players"
    results:           list[PlayerSummary]
    total_found:       int
    query_interpreted: str
    timestamp_utc:     datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# 2. GetPlayerValueOutput
# ---------------------------------------------------------------------------

class ContractDetail(BaseModel):
    salary_usd:       NumericWithSource
    years_remaining:  NumericWithSource
    contract_type:    str   # 'rookie_scale' | 'max' | 'vet_min' | …
    team_option:      bool  = False
    player_option:    bool  = False
    no_trade_clause:  bool  = False

class ValueProjection(BaseModel):
    season:                str = Field(pattern=r"^\d{4}-\d{2}$")
    projected_war:         NumericWithSource
    projected_salary_usd:  NumericWithSource
    surplus_value_m:       NumericWithSource

class GetPlayerValueOutput(BaseModel):
    tool_name:              str     = "get_player_value"
    player_id:              int
    full_name:              str
    position:               str
    age:                    NumericWithSource
    contract:               ContractDetail
    current_surplus_value_m: NumericWithSource
    war_current_season:     Optional[NumericWithSource] = None
    projections:            list[ValueProjection]       = Field(default_factory=list)
    percentile_rank:        Optional[NumericWithSource] = None
    season:                 str
    timestamp_utc:          datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# 3. CheckMoveOutput
# ---------------------------------------------------------------------------

class CBAViolation(BaseModel):
    rule_id:           str            # 'APRON_HARD' | 'TRADE_SALARY_MATCHING' | …
    description:       str
    severity:          str            # 'blocking' | 'warning'
    remediation_hint:  Optional[str] = None

class SalaryMatchingDetail(BaseModel):
    team_id:           int
    team_abbreviation: str
    salary_out_usd:    NumericWithSource
    salary_in_usd:     NumericWithSource
    delta_usd:         NumericWithSource
    compliant:         bool

class CheckMoveOutput(BaseModel):
    tool_name:            str     = "check_move"
    move_type:            str
    is_legal:             bool
    violations:           list[CBAViolation]         = Field(default_factory=list)
    salary_matching:      list[SalaryMatchingDetail] = Field(default_factory=list)
    cap_impact_by_team:   dict[str, NumericWithSource] = Field(default_factory=dict)
    apron_status_after:   Optional[dict[str, str]]   = None
    notes:                list[str]                  = Field(default_factory=list)
    timestamp_utc:        datetime                   = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# 4. ProposeMovesOutput
# ---------------------------------------------------------------------------

class MoveProposal(BaseModel):
    proposal_id:            str
    move_type:              str
    description:            str
    players_involved:       list[int]
    estimated_surplus_gain_m: NumericWithSource
    cba_pre_checked:        bool
    cba_legal:              Optional[bool]           = None
    confidence_score:       NumericWithSource
    rationale:              str

class ProposeMovesOutput(BaseModel):
    tool_name:              str     = "propose_moves"
    team_id:                int
    objective_echoed:       str
    proposals:              list[MoveProposal]
    cap_space_available_usd: NumericWithSource
    timestamp_utc:          datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# 5. FindComparablesOutput
# ---------------------------------------------------------------------------

class ComparablePlayer(BaseModel):
    player_id:              int
    full_name:              str
    similarity_score:       NumericWithSource
    age_at_comparison:      NumericWithSource
    position:               str
    war_at_comparison:      Optional[NumericWithSource] = None
    salary_at_comparison_usd: Optional[NumericWithSource] = None
    season_reference:       str = Field(pattern=r"^\d{4}-\d{2}$")
    key_similarities:       list[str] = Field(default_factory=list)
    key_differences:        list[str] = Field(default_factory=list)

class FindComparablesOutput(BaseModel):
    tool_name:              str     = "find_comparables"
    reference_player_id:    int
    reference_player_name:  str
    comparables:            list[ComparablePlayer]
    method_used:            str
    timestamp_utc:          datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# 6. SearchRiskNewsOutput
# ---------------------------------------------------------------------------

class RiskSeverity(str, Enum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"

class RiskItem(BaseModel):
    risk_id:         str
    category:        str
    headline:        str
    summary:         str            = Field(max_length=500)
    severity:        RiskSeverity
    date_published:  Optional[datetime] = None
    source_url:      Optional[str]  = None
    source_name:     Optional[str]  = None
    impact_on_value: Optional[str]  = None

class SearchRiskNewsOutput(BaseModel):
    tool_name:             str          = "search_risk_news"
    player_id:             Optional[int] = None
    player_name_resolved:  str
    risk_items:            list[RiskItem]
    overall_risk_level:    RiskSeverity
    lookback_days_used:    int
    timestamp_utc:         datetime     = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Union type pour le dispatcher
# ---------------------------------------------------------------------------

ToolOutput = (
    SearchPlayersOutput
    | GetPlayerValueOutput
    | CheckMoveOutput
    | ProposeMovesOutput
    | FindComparablesOutput
    | SearchRiskNewsOutput
    | ToolError
)