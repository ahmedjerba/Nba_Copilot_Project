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
    ExceptionType,
    Move,
    MoveResult,
    MoveStatus,
    MoveType,
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
    # Valeur estimée par le Composant 3
    market_value:    Optional[int] = None


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
    def best_moves(self) -> list[MoveResult]:
        return self.legal_moves[: self.top_n]

    def summary(self) -> str:
        lines = [
            f"═══ Engine Report — Team {self.team_id} | {self.saison} ═══",
            f"Candidats générés  : {self.total_candidates}",
            f"Moves légaux       : {len(self.legal_moves)}",
            f"Moves illégaux     : {len(self.illegal_moves)}",
            f"Top {self.top_n} retenus  :",
        ]
        for i, m in enumerate(self.best_moves, 1):
            lines.append(
                f"  {i:>2}. [{m.value_signal}] {m.move.description} "
                f"| cap après : {m.cap_hit_after:,}$"
            )
        if self.warnings:
            lines.append("\n⚠️  Warnings :")
            lines.extend(f"   - {w}" for w in self.warnings)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

from Composant4.config import DB_URL

def _get_conn() -> psycopg2.extensions.connection:
    url = DB_URL
    if not url:
        raise EnvironmentError("DB_URL manquant dans .env")
    return psycopg2.connect(url, cursor_factory=psycopg2.extras.RealDictCursor)


def _load_team_state(team_id: int, saison: str) -> TeamPayroll:
    """
    Charge TeamPayroll depuis team_payrolls.
    Lève ValueError si l'équipe est inconnue pour cette saison.
    """
    sql = """
        SELECT total_cap_hit, total_base_salary, nb_joueurs
        FROM   team_payrolls
        WHERE  team_id = %s AND saison = %s
        LIMIT  1
    """
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (team_id, saison))
            row = cur.fetchone()

    if not row:
        raise ValueError(
            f"Aucun payroll trouvé pour team_id={team_id}, saison={saison}. "
            f"Vérifiez que la DB est alimentée."
        )

    return TeamPayroll(
        team_id           = team_id,
        saison            = saison,
        total_cap_hit     = row["total_cap_hit"],
        total_base_salary = row["total_base_salary"],
        nb_joueurs        = row["nb_joueurs"],
        # mle_used / bi_annual_used → False par défaut (pas encore en DB)
        mle_used          = False,
        bi_annual_used    = False,
    )


def _load_team_roster(team_id: int, saison: str) -> list[RosterPlayer]:
    """
    Charge le roster complet avec contrats + stats de base.
    Joint players + contracts + player_stats pour enrichir chaque joueur.
    """
    sql = """
        SELECT
            p.player_id,
            p.player_name,
            p.position,
            c.saison,
            c.cap_hit,
            c.base_salary,
            c.years_remaining,
            c.contract_value,
            ps.age
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
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (team_id, team_id, saison))
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
            )
        )

    if not roster:
        logger.warning(
            "Roster vide pour team_id=%s, saison=%s", team_id, saison
        )

    return roster


def _load_available_players(
    exclude_team_id: int,
    saison: str,
    limit: int = 50,
) -> list[RosterPlayer]:
    """
    Charge des joueurs disponibles sur le marché (autres équipes).
    Utilisé pour générer des candidats de trade ou signature.
    """
    sql = """
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
            ps.age
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
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (exclude_team_id, saison, limit))
            rows = cur.fetchall()

    return [
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
        )
        for r in rows
    ]


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
            players_out  = [
                Contract(
                    player_id       = player.player_id,
                    player_name     = player.player_name,
                    saison          = saison,
                    cap_hit         = player.cap_hit,
                    base_salary     = player.base_salary,
                    years_remaining = player.years_remaining,
                    contract_value  = player.contract_value,
                )
            ],
            players_in   = [],
            salary_out   = player.cap_hit,
            salary_in    = 0,
            salary_delta = -player.cap_hit,
            salary_offer = None,
            exception    = ExceptionType.NONE,
            value_delta  = -(player.market_value or player.cap_hit // 2),
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
            players_out  = [],
            players_in   = [],
            salary_out   = 0,
            salary_in    = offer,
            salary_delta = offer,
            salary_offer = offer,
            exception    = ExceptionType.NONE,
            value_delta  = int(offer * 0.9),  # placeholder
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
                players_out  = [],
                players_in   = [],
                salary_out   = 0,
                salary_in    = mle,
                salary_delta = mle,
                salary_offer = mle,
                exception    = ExceptionType.MLE_NON_TAXPAYER,
                value_delta  = int(mle * 0.85),
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
            players_out  = [],
            players_in   = [],
            salary_out   = 0,
            salary_in    = mle_t,
            salary_delta = mle_t,
            salary_offer = mle_t,
            exception    = ExceptionType.MLE_TAXPAYER,
            value_delta  = int(mle_t * 0.85),
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
            players_out  = [],
            players_in   = [],
            salary_out   = 0,
            salary_in    = bae,
            salary_delta = bae,
            salary_offer = bae,
            exception    = ExceptionType.BI_ANNUAL,
            value_delta  = int(bae * 0.80),
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
                move_type = MoveType.TRADE,
                players_out = [
                    Contract(
                        player_id       = p.player_id,
                        player_name     = p.player_name,
                        saison          = saison,
                        cap_hit         = p.cap_hit,
                        base_salary     = p.base_salary,
                        years_remaining = p.years_remaining,
                        contract_value  = p.contract_value,
                    )
                    for p in out_group
                ],
                players_in = [
                    Contract(
                        player_id       = p.player_id,
                        player_name     = p.player_name,
                        saison          = saison,
                        cap_hit         = p.cap_hit,
                        base_salary     = p.base_salary,
                        years_remaining = p.years_remaining,
                        contract_value  = p.contract_value,
                    )
                    for p in in_group
                ],
                salary_out   = salary_out,
                salary_in    = salary_in,
                salary_delta = salary_in - salary_out,
                salary_offer = None,
                exception    = ExceptionType.NONE,
                value_delta  = (val_in + pick_value) - val_out,
                description  = (
                    f"Trade {names_out} → {names_in} "
                    f"(out: {salary_out:,}$ | in: {salary_in:,}$)"
                ),
                # Stub : on inclut 1 first round pick sortant symbolique
                picks_in  = [],
                picks_out = [
                    {
                        "round":      1,
                        "protection": None,
                        "years_out":  1,
                        "value":      pick_value,
                    }
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
        available   = _load_available_players(team_id, saison)
        trade_moves = _generate_trade_candidates(
            roster, available, state, saison, rules, warnings
        )
        all_candidates.extend(trade_moves)
        logger.info("Trade candidates : %s", len(trade_moves))

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