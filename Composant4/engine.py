# Composant4/engine.py
"""
NBA Front-Office Copilot — Composant 4
Moteur de proposition de moves CBA.

Responsabilités :
- Charger l'état financier et le roster d'une équipe depuis la DB
- Générer des moves candidats (trade, sign, waive)
- Filtrer les moves légaux via is_legal (rules.py)
- Scorer et trier par value_delta
- Retourner les N meilleurs moves avec explication complète
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

from Composant4.schemas import (
    Contract,
    DraftPick,
    ExceptionType,
    Move,
    MoveResult,
    MoveStatus,
    MoveType,
    PickRound,
    TeamPayroll,
)
from Composant4.ruleees import is_legal

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SAISON = "2025-26"
DEFAULT_TOP_N  = 10

# Valeurs pick stub — remplacées quand draft_picks est en DB
PICK_VALUES: dict[str, dict[str, int]] = {
    "first": {
        "lottery":   12_000_000,   # top-14
        "mid":        5_500_000,   # 15-20
        "late":       2_500_000,   # 21-30
        "unknown":    6_000_000,   # protection inconnue
    },
    "second": {
        "early":      1_200_000,   # top-10 du 2nd tour
        "late":         500_000,   # 11-30 du 2nd tour
        "unknown":      800_000,
    },
}

# Décote temporelle : pick dans N années vaut X% de moins
PICK_YEAR_DISCOUNT: dict[int, float] = {
    0: 1.00,   # saison en cours
    1: 0.92,
    2: 0.82,
    3: 0.70,
    4: 0.58,
    5: 0.46,
}


# ---------------------------------------------------------------------------
# Internal dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RosterPlayer:
    """Représentation interne d'un joueur + contrat pour le moteur."""
    player_id:       int
    player_name:     str
    team_id:         int
    saison:          str
    cap_hit:         int
    base_salary:     int
    years_remaining: int
    contract_value:  int
    position:        Optional[str] = None
    age:             Optional[float] = None
    # Valeurs estimées par le Composant 3
    market_value:    Optional[int] = None    # Surplus net (valeur sportive - cap hit)
    estimated_market_value: Optional[int] = None # Vraie valeur marché brute en $
    surplus_value:   Optional[int] = None    # Surplus net en $ (identique au surplus)


@dataclass
class EngineResult:
    """Résultat complet retourné par propose_moves."""
    team_id:          int
    saison:           str
    team_state:       TeamPayroll
    legal_moves:      list[MoveResult]
    illegal_moves:    list[MoveResult]
    total_candidates: int
    top_n:            int
    warnings:         list[str] = field(default_factory=list)

    @property
    def legal_trades(self) -> list[MoveResult]:
        return [m for m in self.legal_moves if m.move.move_type == MoveType.TRADE]

    @property
    def legal_signs(self) -> list[MoveResult]:
        return [m for m in self.legal_moves if m.move.move_type == MoveType.SIGN]

    @property
    def legal_waives(self) -> list[MoveResult]:
        return [m for m in self.legal_moves if m.move.move_type == MoveType.WAIVE]

    @property
    def best_moves(self) -> list[MoveResult]:
        # Priorité aux trades et signatures légaux, puis waives
        sorted_moves = sorted(
            self.legal_moves,
            key=lambda m: (
                1 if m.move.move_type == MoveType.TRADE else (0 if m.move.move_type == MoveType.SIGN else -1),
                m.value_delta or 0,
            ),
            reverse=True,
        )
        return sorted_moves[: self.top_n]

    def summary(self) -> str:
        lines = [
            f"═══ Engine Report — Team {self.team_id} | {self.saison} ═══",
            f"Candidats générés  : {self.total_candidates}",
            f"Moves légaux       : {len(self.legal_moves)} (Trades: {len(self.legal_trades)}, Signatures: {len(self.legal_signs)}, Waives: {len(self.legal_waives)})",
            f"Moves illégaux     : {len(self.illegal_moves)}",
            f"Top {self.top_n} retenus  :",
        ]
        for i, m in enumerate(self.best_moves, 1):
            lines.append(
                f"\n  {i:>2}. [{m.value_signal}] {m.move.description} | cap après : {m.cap_hit_after:,}$"
            )
            if m.move.move_type == MoveType.TRADE:
                for pout in m.move.players_out:
                    s = pout.surplus_value if pout.surplus_value is not None else pout.estimated_value
                    s_str = f"{s:+,}$" if s is not None else "N/A"
                    lines.append(f"       📤 OUT: {pout.player_name} | Salaire: {pout.cap_hit:,}$ | Surplus C3: {s_str}")
                for pin in m.move.players_in:
                    s = pin.surplus_value if pin.surplus_value is not None else pin.estimated_value
                    s_str = f"{s:+,}$" if s is not None else "N/A"
                    lines.append(f"       📥 IN : {pin.player_name} | Salaire: {pin.cap_hit:,}$ | Surplus C3: {s_str}")
                lines.append(f"       📊 SURPLUS TOTAL DU TRADE : {m.value_delta:+,}$")
        if self.warnings:
            lines.append("\n⚠️  Warnings :")
            lines.extend(f"   - {w}" for w in self.warnings)
        return "\n".join(lines)

    def detailed_report(self) -> str:
        """Génère le rapport exhaustif au format texte avec salaire et surplus par joueur."""
        lines = [
            "═" * 78,
            f"  NBA COPILOT — RAPPORT D'ANALYSE FRONT-OFFICE (COMPOSANT 4)",
            f"  Équipe : {self.team_state.team_name} (ID: {self.team_id}) | Saison : {self.saison}",
            f"  Masse salariale actuelle : {self.team_state.total_cap_hit:,}$ | Joueurs : {self.team_state.nb_joueurs}",
            "═" * 78,
            "",
            "📊 VUE D'ENSEMBLE :",
            f"   • Candidats explorés : {self.total_candidates}",
            f"   • Moves légaux       : {len(self.legal_moves)} (Trades: {len(self.legal_trades)}, Signatures: {len(self.legal_signs)}, Waives: {len(self.legal_waives)})",
            f"   • Moves illégaux     : {len(self.illegal_moves)}",
            "",
            "═" * 78,
            "⭐ MEILLEURES OPTIONS DE TRADES (SOLVEUR CP-SAT & CBA)",
            "═" * 78,
        ]

        if not self.legal_trades:
            lines.append("  (Aucun trade légal direct trouvé — vérifier le statut Apron 2 ou élargir le pool)")
        else:
            for i, m in enumerate(self.legal_trades[:self.top_n], 1):
                lines.append(f"\n[OPTION {i}] {m.value_signal}")
                lines.append(m.breakdown())
                lines.append("-" * 78)

        if self.legal_signs:
            lines.append("\n" + "═" * 78)
            lines.append("✍️ OPPORTUNITÉS DE SIGNATURES (EXCEPTIONS CBA)")
            lines.append("═" * 78)
            for i, m in enumerate(self.legal_signs[:3], 1):
                lines.append(f"  {i}. {m.move.description} | Cap après : {m.cap_hit_after:,}$")

        if self.legal_waives:
            lines.append("\n" + "═" * 78)
            lines.append("✂️ OPTIONS DE LIBÉRATIONS (WAIVES / DEAD MONEY)")
            lines.append("═" * 78)
            lines.append("  ⚠️ RAPPEL CBA : Un contrat garanti coupé reste à 100% en Dead Cap.")
            for i, m in enumerate(self.legal_waives[:3], 1):
                lines.append(f"  {i}. {m.move.description} | {m.value_signal} (Perte sportive nette)")

        if self.warnings:
            lines.append("\n" + "═" * 78)
            lines.append("⚠️ AVERTISSEMENTS RÉGLEMENTAIRES")
            lines.append("═" * 78)
            for w in self.warnings:
                lines.append(f"   • {w}")

        return "\n".join(lines)

    def export_to_file(self, filepath: str = "Composant4/meilleures_options_c4.txt") -> str:
        """Exporte le rapport complet au format texte."""
        content = self.detailed_report()
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return filepath


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

from Composant4.config import DB_URL

def _get_conn() -> psycopg2.extensions.connection:
    url = DB_URL
    if not url:
        raise EnvironmentError("DB_URL manquant dans .env")
    return psycopg2.connect(url, cursor_factory=psycopg2.extras.RealDictCursor)


# engine.py — remplace _load_team_state entièrement

def _load_team_state(team_id: int, saison: str) -> TeamPayroll:
    sql = """
        SELECT tp.total_cap_hit, tp.total_base_salary, tp.nb_joueurs, t.team_name
        FROM team_payrolls tp
        JOIN teams t ON t.team_id = tp.team_id
        WHERE tp.team_id = %s AND tp.saison = %s
        LIMIT 1
    """
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (team_id, saison))
            row = cur.fetchone()

    if not row:
        raise ValueError(
            f"Aucun payroll trouvé pour team_id={team_id}, saison={saison}."
        )

    return TeamPayroll(
        team_id           = team_id,
        team_name         = row["team_name"],   # ← ajouté
        saison            = saison,
        total_cap_hit     = row["total_cap_hit"],
        total_base_salary = row["total_base_salary"],
        nb_joueurs        = row["nb_joueurs"],
        mle_used          = False,
        bi_annual_used    = False,
    )


# SQL avec predictions (market_value depuis surplus_value)
_SQL_ROSTER_WITH_PRED = """
    SELECT
        p.player_id,
        p.player_name,
        p.position,
        c.saison,
        c.cap_hit,
        c.base_salary,
        c.years_remaining,
        c.contract_value,
        ps.age,
        pr.surplus_value
    FROM   contracts  c
    JOIN   players    p  ON p.player_id  = c.player_id
    LEFT JOIN player_stats ps
           ON ps.player_id = c.player_id
          AND ps.saison    = c.saison
          AND ps.team_id   = %s
    LEFT JOIN predictions pr
           ON pr.player_id = c.player_id
          AND pr.saison    = c.saison
    WHERE  p.team_id = %s
      AND  c.saison  = %s
    ORDER  BY c.cap_hit DESC
"""

# SQL de fallback si la table predictions n'existe pas encore
_SQL_ROSTER_NO_PRED = """
    SELECT
        p.player_id,
        p.player_name,
        p.position,
        c.saison,
        c.cap_hit,
        c.base_salary,
        c.years_remaining,
        c.contract_value,
        ps.age,
        NULL::bigint AS surplus_value
    FROM   contracts  c
    JOIN   players    p  ON p.player_id  = c.player_id
    LEFT JOIN player_stats ps
           ON ps.player_id = c.player_id
          AND ps.saison    = c.saison
          AND ps.team_id   = %s
    WHERE  p.team_id = %s
      AND  c.saison  = %s
    ORDER  BY c.cap_hit DESC
"""


def _enrich_with_composant3(players: list[RosterPlayer], saison: str) -> None:
    """
    Enrichit in-place le champ market_value des RosterPlayer
    via Composant3.valuation_service.get_bulk_valuations
    pour les joueurs dont market_value est None.
    """
    missing_pids = [p.player_id for p in players if p.market_value is None]
    if not missing_pids:
        return

    try:
        from Composant3.valuation_service import get_bulk_valuations
        valuations = get_bulk_valuations(missing_pids, saison)
        for p in players:
            if p.market_value is None and p.player_id in valuations:
                surplus = valuations[p.player_id]
                if surplus is not None:
                    p.surplus_value = int(surplus)
                    p.market_value  = int(surplus)
                    p.estimated_market_value = int(p.cap_hit + surplus)
    except Exception as exc:
        logger.warning(
            "Composant 3 valuation non disponible pour saison=%s : %s",
            saison, exc
        )


def _load_team_roster(team_id: int, saison: str) -> list[RosterPlayer]:
    """
    Charge le roster complet avec contrats + stats de base.
    Tente de charger surplus_value depuis la table predictions si présente,
    sinon calcule directement la valorisation via Composant 3.
    """
    import psycopg2.errors as _pgerr

    rows = None
    with _get_conn() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(_SQL_ROSTER_WITH_PRED, (team_id, team_id, saison))
                rows = cur.fetchall()
            except _pgerr.UndefinedTable:
                conn.rollback()

    # Fallback : requête sans predictions
    if rows is None:
        with _get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(_SQL_ROSTER_NO_PRED, (team_id, team_id, saison))
                rows = cur.fetchall()

    roster = []
    for r in rows:
        roster.append(
            RosterPlayer(
                player_id       = r["player_id"],
                player_name     = r["player_name"],
                team_id         = team_id,
                saison          = r["saison"],
                cap_hit         = r["cap_hit"],
                base_salary     = r["base_salary"],
                years_remaining = r["years_remaining"],
                contract_value  = r["contract_value"],
                position        = r["position"],
                age             = r["age"],
                market_value    = r["surplus_value"],  # None si pas de prédiction en DB
            )
        )

    if not roster:
        logger.warning(
            "Roster vide pour team_id=%s, saison=%s", team_id, saison
        )
    else:
        # Calcul dynamique des surplus values via Composant 3
        _enrich_with_composant3(roster, saison)

    return roster


_SQL_AVAIL_WITH_PRED = """
    SELECT
        p.player_id,
        p.player_name,
        p.team_id,
        p.position,
        c.saison,
        c.cap_hit,
        c.base_salary,
        c.years_remaining,
        c.contract_value,
        ps.age,
        pr.surplus_value
    FROM   contracts  c
    JOIN   players    p  ON p.player_id = c.player_id
    LEFT JOIN player_stats ps
           ON ps.player_id = c.player_id
          AND ps.saison    = c.saison
          AND ps.team_id   = p.team_id
    LEFT JOIN predictions pr
           ON pr.player_id = c.player_id
          AND pr.saison    = c.saison
    WHERE  p.team_id != %s
      AND  c.saison   = %s
    ORDER  BY c.cap_hit DESC
    LIMIT  %s
"""

_SQL_AVAIL_NO_PRED = """
    SELECT
        p.player_id,
        p.player_name,
        p.team_id,
        p.position,
        c.saison,
        c.cap_hit,
        c.base_salary,
        c.years_remaining,
        c.contract_value,
        ps.age,
        NULL::bigint AS surplus_value
    FROM   contracts  c
    JOIN   players    p  ON p.player_id = c.player_id
    LEFT JOIN player_stats ps
           ON ps.player_id = c.player_id
          AND ps.saison    = c.saison
          AND ps.team_id   = p.team_id
    WHERE  p.team_id != %s
      AND  c.saison   = %s
    ORDER  BY c.cap_hit DESC
    LIMIT  %s
"""


def _load_available_players(
    exclude_team_id: int,
    saison: str,
    limit: int = 50,
) -> list[RosterPlayer]:
    """
    Charge des joueurs disponibles sur le marché (autres équipes).
    Tente de charger surplus_value depuis la table predictions si présente,
    sinon calcule directement la valorisation via Composant 3.
    """
    import psycopg2.errors as _pgerr

    rows = None
    with _get_conn() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(_SQL_AVAIL_WITH_PRED, (exclude_team_id, saison, limit))
                rows = cur.fetchall()
            except _pgerr.UndefinedTable:
                conn.rollback()

    # Fallback : requête sans predictions
    if rows is None:
        with _get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(_SQL_AVAIL_NO_PRED, (exclude_team_id, saison, limit))
                rows = cur.fetchall()

    available = [
        RosterPlayer(
            player_id       = r["player_id"],
            player_name     = r["player_name"],
            team_id         = r["team_id"],
            saison          = r["saison"],
            cap_hit         = r["cap_hit"],
            base_salary     = r["base_salary"],
            years_remaining = r["years_remaining"],
            contract_value  = r["contract_value"],
            position        = r["position"],
            age             = r["age"],
            market_value    = r["surplus_value"],  # None si pas de prédiction en DB
        )
        for r in rows
    ]

    # Calcul dynamique des surplus values via Composant 3
    _enrich_with_composant3(available, saison)

    return available


def _load_player(player_id: int, saison: str) -> RosterPlayer | None:
    """Charge un joueur individuel avec son contrat et sa valorisation."""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    p.player_id,
                    p.player_name,
                    p.team_id,
                    p.position,
                    c.saison,
                    c.cap_hit,
                    c.base_salary,
                    c.years_remaining,
                    c.contract_value,
                    ps.age,
                    NULL::bigint AS surplus_value
                FROM   contracts  c
                JOIN   players    p  ON p.player_id = c.player_id
                LEFT JOIN player_stats ps
                       ON ps.player_id = c.player_id
                      AND ps.saison    = c.saison
                      AND ps.team_id   = p.team_id
                WHERE  p.player_id = %s
                  AND  c.saison    = %s
                LIMIT 1
            """, (player_id, saison))
            row = cur.fetchone()

    if not row:
        return None

    player = RosterPlayer(
        player_id       = row["player_id"],
        player_name     = row["player_name"],
        team_id         = row["team_id"],
        saison          = row["saison"],
        cap_hit         = row["cap_hit"],
        base_salary     = row["base_salary"],
        years_remaining = row["years_remaining"],
        contract_value  = row["contract_value"],
        position        = row["position"],
        age             = row["age"],
        market_value    = None,
    )
    _enrich_with_composant3([player], saison)
    return player


# ---------------------------------------------------------------------------
# Pick valuation (stub — sera branché sur draft_picks plus tard)
# ---------------------------------------------------------------------------

def _estimate_pick_value(
    round_: int,
    protection: Optional[str],
    years_out: int,
) -> int:
    """
    Estime la valeur en $ d'un pick de draft.

    Args:
        round_      : 1 (first round) ou 2 (second round)
        protection  : 'lottery', 'mid', 'late', 'early', None
        years_out   : nombre d'années avant que le pick conveye

    Returns:
        Valeur estimée en dollars.

    Note:
        Stub conservatif — sera remplacé par lecture de draft_picks
        + modèle de valorisation quand la table sera alimentée.
    """
    round_key = "first" if round_ == 1 else "second"
    tier_map  = PICK_VALUES[round_key]

    if protection is None:
        # Pick non protégé → on utilise "unknown" comme base neutre
        base = tier_map["unknown"]
    else:
        prot = protection.lower()
        if round_ == 1:
            if "lottery" in prot or "top-14" in prot or "top-10" in prot:
                base = tier_map["lottery"]
            elif "mid" in prot or any(f"top-{n}" in prot for n in range(15, 21)):
                base = tier_map["mid"]
            else:
                base = tier_map["late"]
        else:
            if "early" in prot or any(f"top-{n}" in prot for n in range(1, 11)):
                base = tier_map["early"]
            else:
                base = tier_map["late"]

    # Décote temporelle
    discount = PICK_YEAR_DISCOUNT.get(years_out, 0.40)
    return int(base * discount)


# ---------------------------------------------------------------------------
# Move generators
# ---------------------------------------------------------------------------

def _generate_waive_candidates(
    roster:     list[RosterPlayer],
    state:      TeamPayroll,
    saison:     str,
    warnings:   list[str],
) -> list[Move]:
    """
    Génère des candidats de libération (waive).
    Cible : joueurs avec cap_hit élevé et contrat long restant.
    """
    candidates: list[Move] = []

    for player in roster:
        # Ne pas libérer un joueur avec un contrat max (trop coûteux en stretch)
        if player.cap_hit > 35_000_000:
            warnings.append(
                f"Libération de {player.player_name} ignorée — "
                f"contrat max ({player.cap_hit:,}$), trop coûteux en stretch."
            )
            continue

        move = Move(
            move_type    = MoveType.WAIVE,
            team_id      = state.team_id,
            players_out  = [
                Contract(
                    player_id       = player.player_id,
                    player_name     = player.player_name,
                    team_id         = state.team_id,
                    saison          = saison,
                    cap_hit         = player.cap_hit,
                    base_salary     = player.base_salary,
                    years_remaining = player.years_remaining,
                    estimated_value = player.estimated_market_value if player.estimated_market_value is not None else player.cap_hit,
                    surplus_value   = player.surplus_value if player.surplus_value is not None else player.market_value,
                )
            ],
            players_in   = [],
            salary_offer = None,
            exception    = ExceptionType.NONE,
            description  = (
                f"Libérer {player.player_name} "
                f"({player.cap_hit:,}$ | {player.years_remaining} ans restants)"
            ),
            picks_in     = [],
            picks_out    = [],
        )
        candidates.append(move)

    return candidates


def _generate_sign_candidates(
    state:    TeamPayroll,
    saison:   str,
    rules:    dict,
    warnings: list[str],
) -> list[Move]:
    """
    Génère des candidats de signature (MLE, BAE, cap space).
    Produit un move par exception disponible + un move via cap space si dispo.
    """
    candidates: list[Move] = []
    cap       = rules["thresholds"]["salary_cap"]
    tax       = rules["thresholds"]["luxury_tax"]
    apron1    = rules["thresholds"].get("first_apron")
    exc       = rules["exceptions"]

    cap_space = cap - state.total_cap_hit

    # --- Signature via cap space ---
    if cap_space > 2_000_000:
        offer = min(cap_space, 20_000_000)  # offre exemple = cap space dispo
        move  = Move(
            move_type    = MoveType.SIGN,
            team_id      = state.team_id,
            players_out  = [],
            players_in   = [],
            salary_offer = offer,
            exception    = ExceptionType.NONE,
            description  = f"Signature via cap space ({offer:,}$)",
            picks_in     = [],
            picks_out    = [],
        )
        candidates.append(move)

    # --- MLE Non-Taxpayer ---
    if not state.mle_used and state.total_cap_hit < tax:
        mle = exc["mle_non_taxpayer"]
        if apron1 is None or (state.total_cap_hit + mle) <= apron1:
            move = Move(
                move_type    = MoveType.SIGN,
                team_id      = state.team_id,
                players_out  = [],
                players_in   = [],
                salary_offer = mle,
                exception    = ExceptionType.MLE_NON_TAXPAYER,
                description  = f"Signature MLE non-taxpayer ({mle:,}$)",
                picks_in     = [],
                picks_out    = [],
            )
            candidates.append(move)

    # --- MLE Taxpayer ---
    if not state.mle_used and state.total_cap_hit >= tax:
        mle_t = exc["mle_taxpayer"]
        move  = Move(
            move_type    = MoveType.SIGN,
            team_id      = state.team_id,
            players_out  = [],
            players_in   = [],
            salary_offer = mle_t,
            exception    = ExceptionType.MLE_TAXPAYER,
            description  = f"Signature MLE taxpayer ({mle_t:,}$)",
            picks_in     = [],
            picks_out    = [],
        )
        candidates.append(move)

    # --- Bi-Annual Exception ---
    if not state.bi_annual_used and state.total_cap_hit < tax:
        bae  = exc["bi_annual"]
        move = Move(
            move_type    = MoveType.SIGN,
            team_id      = state.team_id,
            players_out  = [],
            players_in   = [],
            salary_offer = bae,
            exception    = ExceptionType.BI_ANNUAL,
            description  = f"Signature Bi-Annual Exception ({bae:,}$)",
            picks_in     = [],
            picks_out    = [],
        )
        candidates.append(move)

    return candidates


def _generate_trade_candidates(
    roster:           list[RosterPlayer],
    available:        list[RosterPlayer],
    state:            TeamPayroll,
    saison:           str,
    rules:            dict,
    warnings:         list[str],
    max_players_out:  int = 2,
    max_players_in:   int = 2,
) -> list[Move]:
    """
    Génère des candidats de trade (1-for-1 et 2-for-1 / 1-for-2).

    Logique :
    - Pour chaque joueur sortant (ou paire), cherche dans available
      les joueurs entrants compatibles avec le trade matching CBA.
    - Ajoute la valeur des picks (stub) dans value_delta.
    - Limite le nombre de candidats pour garder le moteur rapide.
    """
    from itertools import combinations

    candidates: list[Move] = []
    cap       = rules["thresholds"]["salary_cap"]
    pct       = rules["trade_matching"]["pct_above"]
    flat      = rules["trade_matching"]["flat_bonus"]
    above_flt = rules["trade_matching"]["above_cap_threshold"]

    # Groupes de joueurs sortants (1 ou 2)
    outgoing_groups: list[tuple[RosterPlayer, ...]] = [
        (p,) for p in roster
    ]
    if max_players_out >= 2:
        outgoing_groups += list(combinations(roster, 2))

    for out_group in outgoing_groups:
        salary_out = sum(p.cap_hit for p in out_group)

        # Calcul du max recevable selon position vs cap
        if state.total_cap_hit <= cap:
            max_in = int(salary_out * (1 + pct)) + flat
        else:
            max_in = salary_out + above_flt

        # Joueurs entrants compatibles (1 ou 2)
        incoming_groups: list[tuple[RosterPlayer, ...]] = [
            (p,) for p in available if p.cap_hit <= max_in
        ]
        if max_players_in >= 2:
            incoming_groups += [
                (a, b)
                for a, b in combinations(available, 2)
                if a.cap_hit + b.cap_hit <= max_in
                and a.team_id == b.team_id  # même équipe pour simplifier
            ]

        for in_group in incoming_groups:
            salary_in = sum(p.cap_hit for p in in_group)

            # Valeur delta = valeur reçue − valeur envoyée
            val_out = sum(p.market_value or p.cap_hit for p in out_group)
            val_in  = sum(p.market_value or p.cap_hit for p in in_group)

            # Picks : stub — 1 first round pick non protégé dans 1 an
            pick_value = _estimate_pick_value(
                round_=1, protection=None, years_out=1
            )

            names_out = " + ".join(p.player_name for p in out_group)
            names_in  = " + ".join(p.player_name for p in in_group)

            move = Move(
                move_type    = MoveType.TRADE,
                team_id      = state.team_id,
                players_out  = [
                    Contract(
                        player_id       = p.player_id,
                        player_name     = p.player_name,
                        team_id         = p.team_id,
                        saison          = saison,
                        cap_hit         = p.cap_hit,
                        base_salary     = p.base_salary,
                        years_remaining = p.years_remaining,
                        estimated_value = p.estimated_market_value if p.estimated_market_value is not None else p.cap_hit,
                        surplus_value   = p.surplus_value if p.surplus_value is not None else p.market_value,
                    )
                    for p in out_group
                ],
                players_in   = [
                    Contract(
                        player_id       = p.player_id,
                        player_name     = p.player_name,
                        team_id         = p.team_id,
                        saison          = saison,
                        cap_hit         = p.cap_hit,
                        base_salary     = p.base_salary,
                        years_remaining = p.years_remaining,
                        estimated_value = p.estimated_market_value if p.estimated_market_value is not None else p.cap_hit,
                        surplus_value   = p.surplus_value if p.surplus_value is not None else p.market_value,
                    )
                    for p in in_group
                ],
                salary_offer = None,
                exception    = ExceptionType.NONE,
                description  = (
                    f"Trade {names_out} → {names_in} "
                    f"(out: {salary_out:,}$ | in: {salary_in:,}$)"
                ),
                # Stub : on inclut 1 first round pick sortant symbolique
                picks_in     = [],
                picks_out    = [
                    DraftPick(
                        year            = int(saison[:4]) + 1,
                        round           = PickRound.FIRST,
                        team_origin     = "OWN",
                        protection      = None,
                        estimated_value = pick_value,
                    )
                ],
            )
            candidates.append(move)

            # Sécurité : cap à 500 candidats pour éviter explosion combinatoire
            if len(candidates) >= 500:
                warnings.append(
                    "Candidats trades plafonnés à 500 — "
                    "affinez les filtres pour explorer plus."
                )
                return candidates

    return candidates


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def propose_moves(
    team_id:  int,
    saison:   str = DEFAULT_SAISON,
    top_n:    int = DEFAULT_TOP_N,
    include_trades: bool = True,
    include_signs:  bool = True,
    include_waives: bool = True,
) -> EngineResult:
    """
    Point d'entrée principal du moteur CBA.

    Args:
        team_id         : ID de l'équipe dans la DB
        saison          : ex. "2025-26"
        top_n           : nombre de moves à retourner
        include_trades  : inclure les trades dans l'analyse
        include_signs   : inclure les signatures
        include_waives  : inclure les libérations

    Returns:
        EngineResult avec les moves triés par value_delta décroissant.

    Raises:
        ValueError  : team_id inconnu ou saison absente de la DB
        EnvironmentError : DB_URL manquant
    """
    from Composant4.cba_loader import load_cba_rules

    logger.info("propose_moves — team_id=%s | saison=%s", team_id, saison)
    warnings: list[str] = []

    # 1. Charger l'état financier de l'équipe
    state  = _load_team_state(team_id, saison)
    roster = _load_team_roster(team_id, saison)
    rules  = load_cba_rules(saison)

    logger.info(
        "État chargé — cap hit: %s$ | joueurs: %s",
        f"{state.total_cap_hit:,}", state.nb_joueurs,
    )

    # 2. Générer les candidats
    all_candidates: list[Move] = []

    if include_waives:
        waive_moves = _generate_waive_candidates(roster, state, saison, warnings)
        all_candidates.extend(waive_moves)
        logger.info("Waive candidates : %s", len(waive_moves))

    if include_signs:
        sign_moves = _generate_sign_candidates(state, saison, rules, warnings)
        all_candidates.extend(sign_moves)
        logger.info("Sign candidates  : %s", len(sign_moves))

    if include_trades:
        from Composant4.optimizer import optimize_trades
        available   = _load_available_players(team_id, saison)
        trade_moves = optimize_trades(
            roster, available, state, rules, saison, n_solutions=5
        )
        all_candidates.extend(trade_moves)
        logger.info("Trade candidates (CP-SAT) : %s", len(trade_moves))

    logger.info("Total candidats  : %s", len(all_candidates))

    # 3. Filtrer via is_legal
    legal:   list[MoveResult] = []
    illegal: list[MoveResult] = []

    for move in all_candidates:
        result = is_legal(move, state, saison)
        if result.status == MoveStatus.LEGAL:
            legal.append(result)
        else:
            illegal.append(result)

    logger.info(
        "Légaux: %s | Illégaux: %s", len(legal), len(illegal)
    )

    # 4. Trier les moves légaux par value_delta décroissant
    legal.sort(key=lambda r: r.value_delta, reverse=True)

    return EngineResult(
        team_id          = team_id,
        saison           = saison,
        team_state       = state,
        legal_moves      = legal,
        illegal_moves    = illegal,
        total_candidates = len(all_candidates),
        top_n            = top_n,
        warnings         = warnings,
    )


# ---------------------------------------------------------------------------
# API Avancée : Mode Cible (Target Acquisition & Franchise Partenaire)
# ---------------------------------------------------------------------------

def propose_trade_for_target(
    team_id:          int,
    target_player_id: int,
    saison:           str = DEFAULT_SAISON,
    n_solutions:      int = 3,
) -> list[MoveResult]:
    """
    Mode 'Target Acquisition' :
    L'utilisateur choisit un joueur cible qu'il souhaite acquérir.
    Le moteur identifie son équipe et calcule le package optimal de joueurs
    de notre effectif (plus d'éventuels compléments de l'autre équipe)
    pour conclure le trade en respectant strictement les règles CBA.

    Args:
        team_id          : ID de votre équipe
        target_player_id : ID du joueur visé
        saison           : ex. "2025-26"
        n_solutions      : nombre de propositions de packages

    Returns:
        Liste de MoveResult triés par faisabilité puis value_delta.
    """
    from Composant4.cba_loader import load_cba_rules
    from Composant4.optimizer import optimize_trades

    target = _load_player(target_player_id, saison)
    if not target:
        raise ValueError(
            f"Joueur cible ID {target_player_id} introuvable pour la saison {saison}."
        )

    if target.team_id == team_id:
        raise ValueError(
            f"Le joueur {target.player_name} appartient déjà à votre équipe (team_id={team_id})."
        )

    state  = _load_team_state(team_id, saison)
    roster = _load_team_roster(team_id, saison)
    rules  = load_cba_rules(saison)

    # Charger le roster adverse pour les fillers éventuels
    partner_roster = _load_team_roster(target.team_id, saison)
    available = [target] + [p for p in partner_roster if p.player_id != target.player_id]

    moves = optimize_trades(
        roster=roster,
        available=available,
        state=state,
        rules=rules,
        saison=saison,
        n_solutions=n_solutions,
        target_player_id=target_player_id,
        partner_team_id=target.team_id,
    )

    results: list[MoveResult] = []
    for move in moves:
        res = is_legal(move, state, saison)
        results.append(res)

    results.sort(key=lambda r: (r.status == MoveStatus.LEGAL, r.value_delta), reverse=True)
    return results


def propose_trades_with_team(
    team_id:         int,
    partner_team_id: int,
    saison:          str = DEFAULT_SAISON,
    n_solutions:     int = 5,
) -> list[MoveResult]:
    """
    Mode 'Négociation Franchise à Franchise' :
    Propose les meilleurs trades bilatéraux exclusivement entre team_id
    et partner_team_id.
    """
    from Composant4.cba_loader import load_cba_rules
    from Composant4.optimizer import optimize_trades

    if team_id == partner_team_id:
        raise ValueError("L'équipe partenaire doit être différente de votre équipe.")

    state          = _load_team_state(team_id, saison)
    roster         = _load_team_roster(team_id, saison)
    partner_roster = _load_team_roster(partner_team_id, saison)
    rules          = load_cba_rules(saison)

    if not partner_roster:
        return []

    moves = optimize_trades(
        roster=roster,
        available=partner_roster,
        state=state,
        rules=rules,
        saison=saison,
        n_solutions=n_solutions,
        partner_team_id=partner_team_id,
    )

    results: list[MoveResult] = []
    for move in moves:
        res = is_legal(move, state, saison)
        results.append(res)

    results.sort(key=lambda r: (r.status == MoveStatus.LEGAL, r.value_delta), reverse=True)
    return results