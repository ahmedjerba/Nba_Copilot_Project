"""
Composant 4 — Moteur CBA
schemas.py — Modèles Pydantic

Principe :
- salary_value  = valeur CBA pour le salary matching (toujours 0 pour les picks)
- estimated_value = valeur réelle estimée (picks + joueurs) utilisée par l'agent
  pour évaluer si le trade est "intelligent" au-delà de la légalité
"""

from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MoveType(str, Enum):
    TRADE     = "trade"
    SIGN      = "sign"        # free agent signature
    WAIVE     = "waive"       # libération
    EXTEND    = "extend"      # extension de contrat


class ExceptionType(str, Enum):
    MLE_NON_TAXPAYER = "mle_non_taxpayer"   # 12.4M$
    MLE_TAXPAYER     = "mle_taxpayer"       # 5.2M$
    BI_ANNUAL        = "bi_annual"          # 4.5M$
    MINI_MLE         = "mini_mle"
    NONE             = "none"


class MoveStatus(str, Enum):
    LEGAL   = "legal"
    ILLEGAL = "illegal"


class PickRound(int, Enum):
    FIRST  = 1
    SECOND = 2


# ---------------------------------------------------------------------------
# Contract — miroir de la table contracts + infos enrichies
# ---------------------------------------------------------------------------

class Contract(BaseModel):
    """
    Représente le contrat d'un joueur.
    Champs DB : player_id, saison, cap_hit, years_remaining, base_salary
    Champs enrichis : player_name, team_id, exception_used, is_two_way
    """
    player_id       : int
    player_name     : str
    team_id         : int
    saison          : str                   # ex: "2024-25"

    # Depuis table contracts
    cap_hit         : int                   # en dollars
    base_salary     : int
    years_remaining : int

    # Enrichissements (pas en DB → valeurs par défaut)
    exception_used  : ExceptionType = ExceptionType.NONE
    is_two_way      : bool = False          # inféré si cap_hit < 600_000
    no_trade_clause : bool = False
    trade_kicker    : float = 0.0           # % d'augmentation si tradé (ex: 0.15 = +15%)

    # Valeur estimée par le Composant 3
    estimated_value : int | None = None     # dollars, depuis get_player_value()
    surplus_value   : int | None = None     # estimated_value - cap_hit

    @model_validator(mode="after")
    def infer_two_way(self) -> "Contract":
        if self.cap_hit < 600_000:
            self.is_two_way = True
        return self


# ---------------------------------------------------------------------------
# DraftPick
# ---------------------------------------------------------------------------

class DraftPick(BaseModel):
    """
    Représente un tour de draft inclus dans un trade.

    RÈGLE CBA : salary_value = 0 toujours (les picks ne comptent
    pas dans le salary matching).

    estimated_value : valorisation réelle depuis pick_values.yaml,
    utilisée uniquement par l'agent pour évaluer la qualité du trade.
    """
    year          : int
    round         : PickRound
    team_origin   : str                     # abbreviation ex: "IND", "NYK"
    protection    : str | None = None       # ex: "top-10 protected", "top-5 protected"

    # CBA matching → toujours 0
    salary_value  : int = Field(default=0, frozen=True)

    # Valorisation réelle → chargée depuis pick_values.yaml
    estimated_value : int | None = None     # ex: 12_000_000 pour un top-15 pick

    @property
    def label(self) -> str:
        prot = f" ({self.protection})" if self.protection else ""
        return f"{self.year} {self.round.value}st Round Pick [{self.team_origin}]{prot}"


# ---------------------------------------------------------------------------
# TeamPayroll — état financier d'une équipe
# Miroir de team_payrolls + liste des contrats actifs
# ---------------------------------------------------------------------------

class TeamPayroll(BaseModel):
    """
    État financier complet d'une équipe à un instant T.
    Construit depuis team_payrolls + contracts JOIN players.
    """
    team_id         : int
    team_name       : str
    saison          : str

    # Depuis table team_payrolls
    total_cap_hit   : int                   # masse salariale totale actuelle
    nb_joueurs      : int

    # Contrats actifs (chargés depuis contracts + players)
    contracts       : list[Contract] = Field(default_factory=list)

    # Exceptions déjà utilisées cette saison
    mle_used        : bool = False
    bi_annual_used  : bool = False

    # Calculés automatiquement
    @property
    def cap_space(self) -> int:
        """Cap space disponible (négatif si au-dessus du cap)."""
        from Composant4.cba_service import CBA  # import tardif pour éviter circularité
        return CBA.SALARY_CAP - self.total_cap_hit

    @property
    def tax_space(self) -> int:
        """Marge avant la luxury tax (négatif si au-dessus)."""
        from Composant4.cba_service import CBA
        return CBA.LUXURY_TAX - self.total_cap_hit

    @property
    def apron1_space(self) -> int:
        """Marge avant le first apron."""
        from Composant4.cba_service import CBA
        return CBA.FIRST_APRON - self.total_cap_hit

    @property
    def apron2_space(self) -> int:
        """Marge avant le second apron."""
        from Composant4.cba_service import CBA
        return CBA.SECOND_APRON - self.total_cap_hit

    @property
    def is_over_cap(self) -> bool:
        from Composant4.cba_service import CBA
        return self.total_cap_hit > CBA.SALARY_CAP

    @property
    def is_taxpayer(self) -> bool:
        from Composant4.cba_service import CBA
        return self.total_cap_hit > CBA.LUXURY_TAX

    @property
    def is_over_apron1(self) -> bool:
        from Composant4.cba_service import CBA
        return self.total_cap_hit > CBA.FIRST_APRON

    @property
    def is_over_apron2(self) -> bool:
        from Composant4.cba_service import CBA
        return self.total_cap_hit > CBA.SECOND_APRON


# ---------------------------------------------------------------------------
# Move — un mouvement (trade, signature, waive)
# ---------------------------------------------------------------------------

class Move(BaseModel):
    """
    Représente un mouvement soumis au moteur CBA.

    Pour un TRADE :
      - players_out  : joueurs envoyés par user_team
      - players_in   : joueurs reçus par user_team
      - picks_out    : picks envoyés
      - picks_in     : picks reçus

    Pour une SIGN :
      - players_in   : le joueur signé
      - exception    : exception utilisée

    Pour un WAIVE :
      - players_out  : le joueur libéré
    """
    move_type    : MoveType
    team_id      : int                          # équipe qui initie le move

    # Joueurs
    players_out  : list[Contract] = Field(default_factory=list)
    players_in   : list[Contract] = Field(default_factory=list)

    # Picks (trade uniquement)
    picks_out    : list[DraftPick] = Field(default_factory=list)
    picks_in     : list[DraftPick] = Field(default_factory=list)

    # Signature uniquement
    exception    : ExceptionType = ExceptionType.NONE
    salary_offer : int | None = None            # salaire proposé pour une signature

    # --- Propriétés de matching ---

    @property
    def salary_out(self) -> int:
        """Masse salariale envoyée (picks exclus — règle CBA)."""
        return sum(c.cap_hit for c in self.players_out)

    @property
    def salary_in(self) -> int:
        """Masse salariale reçue (picks exclus — règle CBA)."""
        return sum(c.cap_hit for c in self.players_in)

    @property
    def salary_delta(self) -> int:
        """Variation nette de masse salariale après le move."""
        return self.salary_in - self.salary_out

    # --- Valeur réelle (agent) ---

    @property
    def estimated_value_out(self) -> int:
        """Valeur totale envoyée (joueurs + picks estimés)."""
        joueurs = sum(c.estimated_value or 0 for c in self.players_out)
        picks   = sum(p.estimated_value or 0 for p in self.picks_out)
        return joueurs + picks

    @property
    def estimated_value_in(self) -> int:
        """Valeur totale reçue (joueurs + picks estimés)."""
        joueurs = sum(c.estimated_value or 0 for c in self.players_in)
        picks   = sum(p.estimated_value or 0 for p in self.picks_in)
        return joueurs + picks

    @property
    def value_delta(self) -> int:
        """
        Bilan de valeur réelle du trade.
        Positif = vous gagnez de la valeur.
        Négatif = vous cédez de la valeur.
        """
        return self.estimated_value_in - self.estimated_value_out


# ---------------------------------------------------------------------------
# MoveResult — résultat du moteur CBA
# ---------------------------------------------------------------------------

class RuleResult(BaseModel):
    """Résultat d'une règle individuelle."""
    rule    : str           # nom de la règle ex: "check_trade_matching"
    passed  : bool
    reason  : str           # motif explicite pour l'agent


class MoveResult(BaseModel):
    """
    Résultat complet retourné par is_legal().
    L'agent utilise ce modèle pour formuler sa réponse.
    """
    move            : Move
    status          : MoveStatus

    # Détail règle par règle
    rule_results    : list[RuleResult] = Field(default_factory=list)

    # Premier échec (None si légal)
    blocking_rule   : str | None = None
    blocking_reason : str | None = None

    # Bilan financier après le move
    cap_hit_after   : int | None = None     # masse salariale résultante
    cap_space_after : int | None = None

    # Bilan de valeur réelle (pour l'agent)
    value_delta     : int | None = None     # move.value_delta si calculé
    value_signal    : str | None = None     # "GAIN 🟢" | "PERTE 🔴" | "NEUTRE ⚪"

    @property
    def is_legal(self) -> bool:
        return self.status == MoveStatus.LEGAL

    @property
    def summary(self) -> str:
        """Résumé court pour l'agent."""
        if self.is_legal:
            return (
                f"✅ Trade légal. "
                f"Masse salariale après : {self.cap_hit_after:,}$. "
                f"Bilan valeur : {self.value_signal or 'non calculé'}."
            )
        return (
            f"❌ Trade illégal — {self.blocking_rule} : {self.blocking_reason}"
        )