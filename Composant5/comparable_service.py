# comparable_service.py
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator
from sqlalchemy.engine import Engine

from similarity import find_similar_players, find_historical_comparables, ComparableResult

# ──────────────────────────────────────────────────────────────────
# Confidence par archétype (issu de validator.py)
# ──────────────────────────────────────────────────────────────────
ARCHETYPE_CONFIDENCE = {
    "franchise_player"  : "high",
    "superstar"         : "high",
    "star_fading"       : "high",
    "3and_d_wing"       : "high",
    "3and_d_specialist" : "high",
    "scoring_big"       : "high",
    "spot_up_shooter"   : "high",
    "shot_blocker"      : "high",
    "rim_protector"     : "medium",
    "declining_vet"     : "medium",
    "physical_big"      : "medium",
    "unclassified"      : "medium",
    "solid_rotation"    : "medium",
    "ascending_wing"    : "medium",
    "declining_star"    : "medium",
    "fringe_rotation"   : "low",
    "raw_prospect"      : "low",
    "developmental_big" : "low",
    "replacement_level" : "low",
}

CONFIDENCE_WARNINGS = {
    "replacement_level" : "Profil trop variable — comparables non fiables (-77% vs baseline).",
    "developmental_big" : "Profil en construction — trop peu de membres, comparables indicatifs seulement.",
    "fringe_rotation"   : "Opportunités imprévisibles — comparables quasi équivalents à la baseline.",
    "raw_prospect"      : "Trajectoire binaire — comparables indicatifs seulement.",
    "solid_rotation"    : "Profil stable à court terme mais dégradation sur N+3.",
    "declining_star"    : "Déclin non linéaire — comparables à interpréter avec prudence.",
}


# ──────────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────────
class ComparableMode(str, Enum):
    current    = "current"
    historical = "historical"


# ──────────────────────────────────────────────────────────────────
# Request
# ──────────────────────────────────────────────────────────────────
class ComparableRequest(BaseModel):
    mode              : ComparableMode          = ComparableMode.current
    player_id         : int                     = Field(..., description="NBA player_id")
    season            : str                     = Field(..., description="Format YYYY-YY ex: 2025-26")
    k                 : int                     = Field(5, ge=1, le=20)
    age_window        : float                   = Field(2.0, ge=0.5, le=5.0,
                                                        description="±années pour mode historical")
    min_similarity_pct: float                   = Field(0.0, ge=0.0, le=100.0,
                                                        description="Seuil minimum de similarité (0=désactivé)")
    weights           : Optional[dict[str, float]] = Field(
        None,
        description="Pondération features ex: {'blk_per100_shrunk': 2.0}"
    )
    role_filter       : Optional[str]           = Field(
        None,
        description="Filtrer résultats par archétype ex: 'rim_protector'"
    )

    @model_validator(mode="after")
    def validate_season_format(self) -> "ComparableRequest":
        import re
        if not re.match(r"^\d{4}-\d{2}$", self.season):
            raise ValueError(f"Format saison invalide : '{self.season}'. Attendu : YYYY-YY")
        return self


# ──────────────────────────────────────────────────────────────────
# Response
# ──────────────────────────────────────────────────────────────────
class ComparableItem(BaseModel):
    player_id              : int
    player_name            : str
    season                 : str
    age                    : float
    valuation_tier         : str
    archetype              : str
    archetype_conf         : float
    archetype_secondary    : Optional[str]
    similarity_pct         : float
    top_shared_features    : list[str]
    top_diverging_features : list[str]


class ComparableResponse(BaseModel):
    player_id            : int
    season               : str
    mode                 : ComparableMode
    archetype            : str
    archetype_conf       : float
    archetype_secondary  : Optional[str]
    confidence           : str
    warning              : Optional[str]
    comparables          : list[ComparableItem]
    total_found          : int
    role_filter_applied  : Optional[str]
    role_filter_matched  : bool                  # ← nouveau


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────
def _get_target_archetype(engine: Engine, player_id: int, season: str) -> dict:
    from sqlalchemy import text
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT archetype, archetype_conf,
                   archetype_secondary, archetype_secondary_conf,
                   is_soft_assigned, is_reclassified
            FROM player_archetypes
            WHERE player_id = :pid AND saison = :season
        """), {"pid": player_id, "season": season}).fetchone()

    if row is None:
        return {
            "archetype":                "unclassified",
            "archetype_conf":           0.0,
            "archetype_secondary":      None,
            "archetype_secondary_conf": None,
            "is_soft_assigned":         False,
            "is_reclassified":          False,
        }

    return {
        "archetype":                row.archetype,
        "archetype_conf":           row.archetype_conf,
        "archetype_secondary":      row.archetype_secondary,
        "archetype_secondary_conf": row.archetype_secondary_conf,
        "is_soft_assigned":         row.is_soft_assigned,
        "is_reclassified":          row.is_reclassified,
    }


def _to_comparable_item(r: ComparableResult) -> ComparableItem:
    return ComparableItem(
        player_id              = r.player_id,
        player_name            = r.player_name,
        season                 = r.season,
        age                    = round(r.age, 1),
        valuation_tier         = r.valuation_tier,
        archetype              = r.archetype,
        archetype_conf         = round(r.archetype_conf, 4),
        archetype_secondary    = r.archetype_secondary,
        similarity_pct         = round(r.similarity_pct, 1),
        top_shared_features    = r.top_shared_features,
        top_diverging_features = r.top_diverging_features,
    )


# ──────────────────────────────────────────────────────────────────
# Point d'entrée principal
# ──────────────────────────────────────────────────────────────────
def find(engine: Engine, request: ComparableRequest) -> ComparableResponse:

    # 1. Appel similarity.py
    if request.mode == ComparableMode.current:
        raw_results = find_similar_players(
            engine    = engine,
            player_id = request.player_id,
            season    = request.season,
            k         = request.k * 3 if request.role_filter else request.k,
            weights   = request.weights,
        )
    else:
        raw_results = find_historical_comparables(
            engine     = engine,
            player_id  = request.player_id,
            season     = request.season,
            age_window = request.age_window,
            k          = request.k * 3 if request.role_filter else request.k,
            weights    = request.weights,
        )

    # 2. Filtre min_similarity_pct
    if request.min_similarity_pct > 0.0:
        raw_results = [
            r for r in raw_results
            if r.similarity_pct >= request.min_similarity_pct
        ]

    # 3. Filtre role_filter
    role_filter_matched = False
    if request.role_filter:
        filtered = [
            r for r in raw_results
            if r.archetype == request.role_filter
            or r.archetype_secondary == request.role_filter
        ]
        if filtered:
            role_filter_matched = True
            results = filtered[:request.k]
        else:
            results = raw_results[:request.k]   # fallback silencieux → matched reste False
    else:
        results = raw_results[:request.k]

    # 4. Archétype cible
    target = _get_target_archetype(engine, request.player_id, request.season)

    # 5. Confidence + warning
    confidence = ARCHETYPE_CONFIDENCE.get(target["archetype"], "medium")
    warning    = CONFIDENCE_WARNINGS.get(target["archetype"])

    if target["is_soft_assigned"] and not warning:
        warning = "Joueur outlier HDBSCAN — archétype estimé par soft assignment."
    elif target["is_reclassified"] and not warning:
        warning = "Confiance d'assignement faible (<0.15) — archétype reclassifié."

    # 6. Construction réponse
    return ComparableResponse(
        player_id           = request.player_id,
        season              = request.season,
        mode                = request.mode,
        archetype           = target["archetype"],
        archetype_conf      = target["archetype_conf"],
        archetype_secondary = target["archetype_secondary"],
        confidence          = confidence,
        warning             = warning,
        comparables         = [_to_comparable_item(r) for r in results],
        total_found         = len(results),
        role_filter_applied = request.role_filter,
        role_filter_matched = role_filter_matched,
    )