"""
agent/schemas/inputs.py
=======================
Schémas d'entrée des 6 outils — ce que le planificateur envoie à chaque appel.
"""

from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Enums partagés
# ---------------------------------------------------------------------------

class NBAPosition(str, Enum):
    PG = "PG"
    SG = "SG"
    SF = "SF"
    PF = "PF"
    C  = "C"

class SortBy(str, Enum):
    SURPLUS_VALUE = "surplus_value"
    SALARY        = "salary"
    AGE           = "age"
    WAR           = "WAR"
    RELEVANCE     = "relevance"

class RiskCategory(str, Enum):
    INJURY       = "injury"
    BEHAVIORAL   = "behavioral"
    LEGAL        = "legal"
    TRADE_DEMAND = "trade_demand"
    SUSPENSION   = "suspension"
    ALL          = "all"

class MoveType(str, Enum):
    TRADE   = "trade"
    SIGN    = "sign"
    WAIVE   = "waive"
    EXTEND  = "extend"
    TWO_WAY = "two_way"


# ---------------------------------------------------------------------------
# 1. search_players
# ---------------------------------------------------------------------------

class SearchPlayersInput(BaseModel):
    """
    Recherche full-text + filtres dans la base joueurs.
    Au moins un filtre ou une query est obligatoire.
    """
    query:           Optional[str]            = Field(default=None)
    team_ids:        Optional[list[int]]      = Field(default=None)
    positions:       Optional[list[NBAPosition]] = Field(default=None)
    age_min:         Optional[int]            = Field(default=None, ge=18, le=45)
    age_max:         Optional[int]            = Field(default=None, ge=18, le=45)
    salary_min_usd:  Optional[int]            = Field(default=None, ge=0)
    salary_max_usd:  Optional[int]            = Field(default=None, ge=0)
    min_surplus_value: Optional[float]        = Field(default=None)
    sort_by:         SortBy                   = Field(default=SortBy.RELEVANCE)
    limit:           int                      = Field(default=10, ge=1, le=50)

    @model_validator(mode="after")
    def at_least_one_filter(self) -> "SearchPlayersInput":
        filters = [self.query, self.team_ids, self.positions,
                   self.age_min, self.age_max,
                   self.salary_min_usd, self.salary_max_usd,
                   self.min_surplus_value]
        if all(f is None for f in filters):
            raise ValueError("Au moins un filtre ou une query est requis.")
        return self

    @model_validator(mode="after")
    def salary_range_coherent(self) -> "SearchPlayersInput":
        if (self.salary_min_usd is not None and self.salary_max_usd is not None
                and self.salary_min_usd > self.salary_max_usd):
            raise ValueError("salary_min_usd doit être ≤ salary_max_usd.")
        return self

    @model_validator(mode="after")
    def age_range_coherent(self) -> "SearchPlayersInput":
        if (self.age_min is not None and self.age_max is not None
                and self.age_min > self.age_max):
            raise ValueError("age_min doit être ≤ age_max.")
        return self


# ---------------------------------------------------------------------------
# 2. get_player_value
# ---------------------------------------------------------------------------

class GetPlayerValueInput(BaseModel):
    """
    Récupère la fiche de valorisation complète d'un joueur (composant 3).
    """
    player_id:           int            = Field(..., gt=0)
    season:              Optional[str]  = Field(default=None, pattern=r"^\d{4}-\d{2}$")
    include_projections: bool           = Field(default=True)


# ---------------------------------------------------------------------------
# 3. check_move
# ---------------------------------------------------------------------------

class TradePlayerLeg(BaseModel):
    player_id: int = Field(..., gt=0)
    team_id:   int = Field(..., gt=0)

class CheckMoveInput(BaseModel):
    """
    Vérifie la conformité CBA d'un mouvement (composant 4).
    """
    move_type:                MoveType
    players_out:              list[TradePlayerLeg] = Field(default_factory=list)
    players_in:               list[TradePlayerLeg] = Field(default_factory=list)
    cash_considerations_usd:  Optional[int]        = Field(default=None, ge=0)
    target_team_id:           Optional[int]        = Field(default=None)
    proposed_salary_usd:      Optional[int]        = Field(default=None, ge=0)
    proposed_years:           Optional[int]        = Field(default=None, ge=1, le=5)

    @model_validator(mode="after")
    def trade_requires_two_legs(self) -> "CheckMoveInput":
        if self.move_type == MoveType.TRADE:
            if not self.players_out or not self.players_in:
                raise ValueError(
                    "Un trade requiert des joueurs dans players_out ET players_in."
                )
        return self


# ---------------------------------------------------------------------------
# 4. propose_moves
# ---------------------------------------------------------------------------

class ProposeMovesInput(BaseModel):
    """
    Demande au moteur de proposer des moves pour atteindre un objectif.
    """
    objective:          str                      = Field(..., min_length=10, max_length=500)
    team_id:            int                      = Field(..., gt=0)
    constraints:        Optional[dict[str, object]] = Field(default=None)
    exclude_player_ids: Optional[list[int]]      = Field(default=None)
    max_proposals:      int                      = Field(default=5, ge=1, le=20)


# ---------------------------------------------------------------------------
# 5. find_comparables
# ---------------------------------------------------------------------------

class FindComparablesInput(BaseModel):
    """
    Trouve les joueurs comparables via le composant 5.
    """
    player_id:          int           = Field(..., gt=0)
    n_comparables:      int           = Field(default=5, ge=1, le=20)
    same_position_only: bool          = Field(default=False)
    active_only:        bool          = Field(default=True)
    season:             Optional[str] = Field(default=None, pattern=r"^\d{4}-\d{2}$")


# ---------------------------------------------------------------------------
# 6. search_risk_news
# ---------------------------------------------------------------------------

class SearchRiskNewsInput(BaseModel):
    """
    Recherche d'informations de risque (blessures, incidents, rumeurs).
    """
    player_id:    Optional[int]             = Field(default=None, gt=0)
    player_name:  Optional[str]             = Field(default=None, min_length=2)
    categories: list[RiskCategory] = Field(default_factory=lambda: [RiskCategory.ALL])
    lookback_days: int                      = Field(default=90, ge=1, le=730)
    max_results:  int                       = Field(default=10, ge=1, le=50)

    @model_validator(mode="after")
    def player_identified(self) -> "SearchRiskNewsInput":
        if self.player_id is None and self.player_name is None:
            raise ValueError("Fournir player_id ou player_name.")
        return self