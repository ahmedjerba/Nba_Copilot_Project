# Composant4/optimizer.py
"""
NBA Front-Office Copilot — Composant 4
Optimiseur de trades par recherche opérationnelle (OR-Tools CP-SAT).

Expose :
    optimize_trades(roster, available, state, rules, saison, n_solutions=5)
        -> liste de Move (MoveType.TRADE) triés par value_delta décroissant.

Stratégie :
    - Variables binaires x_out[i] / y_in[j]
    - Contraintes CBA salary matching, roster size, trade réel
    - Objectif : maximiser value_delta
    - Pool de solutions par exclusion itérative (interdire chaque solution trouvée)
    - Timeout par appel solveur : 5 secondes
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from Composant4.schemas import (
    Contract,
    ExceptionType,
    Move,
    MoveType,
)

if TYPE_CHECKING:
    from Composant4.engine import RosterPlayer
    from Composant4.schemas import TeamPayroll

logger = logging.getLogger(__name__)

# Scale factor : OR-Tools travaille en entiers -> on divise les $ par 1 000
_SCALE = 1_000


def optimize_trades(
    roster:           list[RosterPlayer],
    available:        list[RosterPlayer],
    state:            TeamPayroll,
    rules:            dict,
    saison:           str,
    n_solutions:      int = 5,
    target_player_id: int | None = None,
    partner_team_id:  int | None = None,
    max_players_out:  int = 3,
    max_players_in:   int = 3,
) -> list[Move]:
    """
    Génère jusqu'à n_solutions trades bilatéraux diversifiés via CP-SAT.

    Garanties :
        - Tous les joueurs entrants proviennent STRICTEMENT de la même franchise adverse.
        - Supporte le mode ciblé (target_player_id ou partner_team_id).
        - Respecte les contraintes CBA de salary matching et de roster size.

    Args:
        roster           : joueurs de l'équipe (candidats à l'envoi)
        available        : joueurs d'autres équipes (candidats à la réception)
        state            : état financier de l'équipe (TeamPayroll)
        rules            : règles CBA chargées par cba_loader
        saison           : ex. "2025-26"
        n_solutions      : nombre max de solutions distinctes à retourner
        target_player_id : joueur précis à acquérir obligatoirement (mode cible)
        partner_team_id  : équipe partenaire spécifique (mode négociation franchise)
        max_players_out  : nombre maximum de joueurs envoyés
        max_players_in   : nombre maximum de joueurs reçus

    Returns:
        Liste de Move (MoveType.TRADE) triés par value_delta décroissant.
        Retourne [] si OR-Tools n'est pas installé ou si aucune solution n'est trouvée.
    """
    try:
        from ortools.sat.python import cp_model  # type: ignore
    except ImportError:
        logger.error(
            "OR-Tools non installé. "
            "Installez-le via : pip install ortools"
        )
        return []

    if not roster or not available:
        logger.warning("optimize_trades : roster ou available vide — aucun trade possible.")
        return []

    # ── Filtrage optionnel par équipe partenaire ──────────────────────────────
    if partner_team_id is not None:
        available = [p for p in available if p.team_id == partner_team_id]
        if not available:
            logger.warning("optimize_trades : aucun joueur disponible pour partner_team_id=%s", partner_team_id)
            return []

    # ── Seuils CBA ────────────────────────────────────────────────────────────
    cap       = rules["thresholds"]["salary_cap"]
    under_cap = state.total_cap_hit <= cap

    # ── Valeurs entières scalées (en milliers de $) ──────────────────────────
    val_out_scaled = [
        int((p.market_value or p.cap_hit) // _SCALE) for p in roster
    ]
    val_in_scaled = [
        int((p.market_value or p.cap_hit) // _SCALE) for p in available
    ]
    cap_out_scaled = [int(p.cap_hit // _SCALE) for p in roster]
    cap_in_scaled  = [int(p.cap_hit // _SCALE) for p in available]

    n_out = len(roster)
    n_in  = len(available)

    # Identifier les équipes adverses représentées
    partner_teams = sorted(list({p.team_id for p in available if p.team_id is not None}))

    solutions: list[Move] = []
    # Liste des combinaisons déjà trouvées à exclure
    excluded: list[tuple[list[int], list[int]]] = []

    for _ in range(n_solutions):
        model = cp_model.CpModel()

        # ── Variables binaires ────────────────────────────────────────────────
        x = [model.NewBoolVar(f"x_out_{i}") for i in range(n_out)]
        y = [model.NewBoolVar(f"y_in_{j}")  for j in range(n_in)]

        # Variables d'activation d'équipe partenaire
        z_team = {t: model.NewBoolVar(f"z_team_{t}") for t in partner_teams}

        # ── Contrainte A : Unicité de la franchise partenaire (Trade bilatéral) ──
        # Au plus 1 équipe partenaire pour l'ensemble des joueurs reçus
        model.Add(sum(z_team.values()) <= 1)

        # Chaque joueur reçu ne peut être pris que si son équipe est active
        for j, p in enumerate(available):
            if p.team_id in z_team:
                model.Add(y[j] <= z_team[p.team_id])

        # Si une équipe partenaire précise est imposée
        if partner_team_id is not None and partner_team_id in z_team:
            model.Add(z_team[partner_team_id] == 1)

        # Si un joueur cible précis est demandé (Target Acquisition)
        if target_player_id is not None:
            target_indices = [j for j, p in enumerate(available) if p.player_id == target_player_id]
            if target_indices:
                model.Add(y[target_indices[0]] == 1)
            else:
                logger.warning("optimize_trades : target_player_id=%s non trouvé dans available", target_player_id)
                return []

        # ── Contrainte 1 : salary matching CBA ───────────────────────────────
        salary_in_expr  = sum(cap_in_scaled[j]  * y[j] for j in range(n_in))
        salary_out_expr = sum(cap_out_scaled[i] * x[i] for i in range(n_out))

        apron2 = rules["thresholds"].get("second_apron")
        is_over_apron2 = bool(apron2 and state.total_cap_hit > apron2)

        if under_cap:
            model.Add(salary_in_expr * 4 <= salary_out_expr * 5 + 400)
        elif is_over_apron2:
            # Règle Second Apron CBA :
            # - Interdiction de recevoir plus de salaire qu'envoyé (100% max, 0$ de marge)
            # - Interdiction d'agréger plusieurs salaires entrants (1 seul joueur reçu)
            model.Add(salary_in_expr <= salary_out_expr)
            model.Add(sum(y[j] for j in range(n_in)) <= 1)
        else:
            model.Add(salary_in_expr <= salary_out_expr + 5_000)

        # ── Contrainte 2 : roster size (14 à 15 joueurs) ─────────────────────
        roster_delta = (
            sum(y[j] for j in range(n_in))
            - sum(x[i] for i in range(n_out))
        )
        model.Add(state.nb_joueurs + roster_delta >= 14)
        model.Add(state.nb_joueurs + roster_delta <= 15)

        # ── Contrainte 3 : limites du trade ──────────────────────────────────
        model.Add(sum(x[i] for i in range(n_out)) >= 1)
        model.Add(sum(y[j] for j in range(n_in))  >= 1)
        model.Add(sum(x[i] for i in range(n_out)) <= max_players_out)
        model.Add(sum(y[j] for j in range(n_in))  <= max_players_in)

        # ── Contraintes d'exclusion des solutions précédentes ─────────────────
        for prev_x_vals, prev_y_vals in excluded:
            same_x = [x[i] for i in range(n_out) if prev_x_vals[i] == 1]
            same_y = [y[j] for j in range(n_in)  if prev_y_vals[j] == 1]
            n_same = len(same_x) + len(same_y)
            if n_same == 0:
                continue
            model.Add(sum(same_x) + sum(same_y) <= n_same - 1)

        # ── Objectif : maximiser value_delta ─────────────────────────────────
        objective = (
            sum(val_in_scaled[j]  * y[j] for j in range(n_in))
            - sum(val_out_scaled[i] * x[i] for i in range(n_out))
        )
        model.Maximize(objective)

        # ── Résolution ────────────────────────────────────────────────────────
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 5.0

        status = solver.Solve(model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            logger.debug("CP-SAT : pas de nouvelle solution (status=%s)", status)
            break

        # ── Extraire la solution ──────────────────────────────────────────────
        x_vals = [int(solver.Value(x[i])) for i in range(n_out)]
        y_vals = [int(solver.Value(y[j])) for j in range(n_in)]

        players_out_list = [roster[i]    for i in range(n_out) if x_vals[i] == 1]
        players_in_list  = [available[j] for j in range(n_in)  if y_vals[j] == 1]

        if not players_out_list or not players_in_list:
            logger.warning("CP-SAT a retourné un trade sans joueurs — ignoré.")
            break

        # Enregistrer pour exclusion itérative
        excluded.append((x_vals, y_vals))

        salary_out_val = sum(p.cap_hit for p in players_out_list)
        salary_in_val  = sum(p.cap_hit for p in players_in_list)
        names_out = " + ".join(p.player_name for p in players_out_list)
        names_in  = " + ".join(p.player_name for p in players_in_list)

        move = Move(
            move_type   = MoveType.TRADE,
            team_id     = state.team_id,
            description = (
                f"Trade {names_out} -> {names_in} "
                f"(out: {salary_out_val:,}$ | in: {salary_in_val:,}$)"
            ),
            players_out = [
                Contract(
                    player_id       = p.player_id,
                    player_name     = p.player_name,
                    team_id         = p.team_id,
                    saison          = saison,
                    cap_hit         = p.cap_hit,
                    base_salary     = p.base_salary,
                    years_remaining = p.years_remaining,
                    estimated_value = p.market_value,
                    surplus_value   = p.market_value,
                )
                for p in players_out_list
            ],
            players_in = [
                Contract(
                    player_id       = p.player_id,
                    player_name     = p.player_name,
                    team_id         = p.team_id,
                    saison          = saison,
                    cap_hit         = p.cap_hit,
                    base_salary     = p.base_salary,
                    years_remaining = p.years_remaining,
                    estimated_value = p.market_value,
                    surplus_value   = p.market_value,
                )
                for p in players_in_list
            ],
            salary_offer = None,
            exception    = ExceptionType.NONE,
            picks_in     = [],
            picks_out    = [],
        )
        solutions.append(move)

        logger.debug(
            "CP-SAT solution %d/%d : %s -> %s (value_delta=%+.0f k$)",
            len(solutions), n_solutions,
            names_out, names_in,
            solver.ObjectiveValue(),
        )

    # Trier par value_delta décroissant
    solutions.sort(key=lambda m: m.value_delta, reverse=True)
    logger.info("optimize_trades : %d solution(s) trouvée(s)", len(solutions))
    return solutions
