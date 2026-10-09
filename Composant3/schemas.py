# ─────────────────────────────────────────────
# Composant3 / schemas.py
# Contrats d'entrée/sortie pour l'agent
# ─────────────────────────────────────────────

from pydantic import BaseModel, Field
from typing import Optional


class PlayerValueRequest(BaseModel):
    player_id: int   = Field(..., description="Identifiant NBA du joueur")
    saison:    str   = Field(..., description="Format 2024-25")


class FacteurExplicatif(BaseModel):
    feature:     str   = Field(..., description="Nom de la feature")
    coefficient: float = Field(..., description="Coefficient Ridge du GLM")
    direction:   str   = Field(..., description="hausse | baisse")


class PlayerValueResponse(BaseModel):
    # Identité
    player_id:   int
    player_name: Optional[str]
    saison:      str

    # Valorisation
    valeur_dollars: Optional[int]
    valeur_pct:     Optional[float]
    p10_dollars:    Optional[int]
    p90_dollars:    Optional[int]
    salary_cap:     Optional[float]

    # Surplus
    cap_hit:  Optional[int]
    surplus:  Optional[int]
    signal:   Optional[str]  # SOUS-PAYÉ 🟢 | SUR-PAYÉ 🔴 | SANS CONTRAT

    # Explication
    top_facteurs: list[FacteurExplicatif] = Field(
        default_factory=list,
        description="Top 5 features qui influencent le plus la valorisation"
    )

    # Erreur
    erreur: Optional[str] = None