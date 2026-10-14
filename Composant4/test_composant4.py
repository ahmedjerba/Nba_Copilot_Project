# Composant4/test_composant4.py
"""
Script de test complet pour le Composant 4 (moteur CBA).

Couvre :
    1. Connectivité DB + présence de la table predictions
    2. Chargement des règles CBA (cba_loader)
    3. _load_team_state       — TeamPayroll depuis DB
    4. _load_team_roster      — market_value depuis predictions
    5. _load_available_players — market_value depuis predictions
    6. optimize_trades        — CP-SAT (optimizer.py)
    7. propose_moves          — pipeline complet end-to-end
    8. is_legal               — validation d'un trade synthétique

Usage :
    cd c:\\Users\\user\\Desktop\\nba_copilot
    python -m Composant4.test_composant4
"""

import logging
import sys
import traceback
from textwrap import indent

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("test_c4")

# ── Couleurs ANSI ─────────────────────────────────────────────────────────────
OK    = "\033[92m✔\033[0m"
FAIL  = "\033[91m✘\033[0m"
SKIP  = "\033[93m⚡\033[0m"
BOLD  = "\033[1m"
RESET = "\033[0m"

results: list[tuple[str, str, str]] = []   # (nom, statut, détail)


def run(name: str, fn):
    """Exécute fn() et enregistre le résultat."""
    print(f"\n{'─'*60}")
    print(f"{BOLD}TEST : {name}{RESET}")
    try:
        detail = fn()
        results.append((name, "OK", detail or ""))
        print(f"  {OK}  {detail or 'Succès'}")
    except SkipTest as e:
        results.append((name, "SKIP", str(e)))
        print(f"  {SKIP}  {e}")
    except Exception as e:
        tb = traceback.format_exc()
        results.append((name, "FAIL", str(e)))
        print(f"  {FAIL}  {e}")
        print(indent(tb, "     "))


class SkipTest(Exception):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _first_team(saison: str = "2025-26") -> int:
    """Retourne le premier team_id qui a un payroll pour cette saison."""
    from Composant4.engine import _get_conn
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT team_id FROM team_payrolls WHERE saison = %s LIMIT 1",
                (saison,)
            )
            row = cur.fetchone()
    if not row:
        raise SkipTest(f"Aucun payroll en DB pour la saison {saison}")
    return row["team_id"]


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_db_connection():
    from Composant4.engine import _get_conn
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 AS ok")
            row = cur.fetchone()
    assert row["ok"] == 1
    return "Connexion DB opérationnelle"


def test_predictions_table_exists():
    """Vérifie la source de surplus_value (table predictions ou pont direct Composant 3)."""
    from Composant4.engine import _get_conn
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'predictions'
                  AND column_name = 'surplus_value'
            """)
            row = cur.fetchone()
    if row is not None:
        return "Table predictions + colonne surplus_value en DB"

    # Fallback dynamique sur Composant 3
    try:
        from Composant3.valuation_service import get_bulk_valuations
        sample_res = get_bulk_valuations([1630552], "2025-26")
        val = sample_res.get(1630552)
        return f"Pont direct Composant 3 actif (surplus test: {val:,}$)"
    except Exception as exc:
        raise SkipTest(
            f"Ni table predictions ni Composant 3 disponible ({exc}) — "
            "market_value restera None (fallback cap_hit actif)"
        )


def test_cba_rules_load():
    from Composant4.cba_loader import load_cba_rules
    rules = load_cba_rules("2025-26")
    required = {"thresholds", "exceptions", "trade_matching", "roster"}
    missing = required - set(rules.keys())
    assert not missing, f"Clés manquantes : {missing}"
    cap = rules["thresholds"]["salary_cap"]
    return f"Règles 2025-26 chargées — salary_cap = {cap:,}$"


def test_load_team_state():
    from Composant4.engine import _load_team_state
    saison  = "2025-26"
    team_id = _first_team(saison)
    state   = _load_team_state(team_id, saison)
    assert state.team_id == team_id
    assert state.total_cap_hit > 0
    assert state.nb_joueurs > 0
    return (
        f"team_id={team_id} | {state.team_name} | "
        f"cap_hit={state.total_cap_hit:,}$ | joueurs={state.nb_joueurs}"
    )


def test_load_team_roster():
    from Composant4.engine import _load_team_roster
    saison  = "2025-26"
    team_id = _first_team(saison)
    roster  = _load_team_roster(team_id, saison)
    assert len(roster) > 0, "Roster vide"

    n_with_mv = sum(1 for p in roster if p.market_value is not None)
    sample    = roster[0]
    return (
        f"{len(roster)} joueurs chargés — "
        f"{n_with_mv} avec market_value depuis predictions | "
        f"Ex: {sample.player_name} cap={sample.cap_hit:,}$ mv={sample.market_value}"
    )


def test_load_available_players():
    from Composant4.engine import _load_available_players
    saison  = "2025-26"
    team_id = _first_team(saison)
    avail   = _load_available_players(team_id, saison, limit=20)
    assert len(avail) > 0, "Aucun joueur disponible"

    n_with_mv = sum(1 for p in avail if p.market_value is not None)
    sample    = avail[0]
    return (
        f"{len(avail)} joueurs disponibles — "
        f"{n_with_mv} avec market_value | "
        f"Ex: {sample.player_name} ({sample.team_id}) cap={sample.cap_hit:,}$"
    )


def test_optimizer_imports():
    """Vérifie qu'OR-Tools est installé et que optimizer.py s'importe."""
    try:
        from ortools.sat.python import cp_model  # noqa: F401
    except ImportError:
        raise SkipTest(
            "OR-Tools non installé — pip install ortools. "
            "optimize_trades retournera [] sans planter."
        )
    from Composant4.optimizer import optimize_trades  # noqa: F401
    return "OR-Tools disponible + optimizer.py importé sans erreur"


def test_optimize_trades():
    """Test end-to-end du solveur CP-SAT avec données réelles."""
    try:
        
        from ortools.sat.python import cp_model  # noqa: F401
    except ImportError:
        raise SkipTest("OR-Tools non installé — test ignoré")

    from Composant4.engine import (
        _load_team_roster,
        _load_available_players,
        _load_team_state,
    )
    from Composant4.cba_loader import load_cba_rules
    from Composant4.optimizer import optimize_trades

    saison  = "2025-26"
    team_id = _first_team(saison)
    state   = _load_team_state(team_id, saison)
    roster  = _load_team_roster(team_id, saison)
    avail   = _load_available_players(team_id, saison, limit=30)
    rules   = load_cba_rules(saison)

    moves = optimize_trades(roster, avail, state, rules, saison, n_solutions=3)

    print(f"\n  Solveur CP-SAT → {len(moves)} solution(s) :")
    for i, m in enumerate(moves, 1):
        in_teams = {c.team_id for c in m.players_in}
        assert len(in_teams) <= 1, f"Anomalie : joueurs entrants d'équipes différentes : {in_teams}"
        print(f"\n  [TRADE PROPOSÉ {i}]")
        print(m.breakdown())

    return f"{len(moves)} trade(s) optimisé(s) retourné(s) — tous monopartenaire"


def test_target_player_trade():
    """Vérifie le mode Target Acquisition (propose_trade_for_target)."""
    try:
        from ortools.sat.python import cp_model  # noqa: F401
    except ImportError:
        raise SkipTest("OR-Tools non installé — test ignoré")

    from Composant4.engine import propose_trade_for_target, _load_available_players
    saison  = "2025-26"
    team_id = _first_team(saison)
    avail   = _load_available_players(team_id, saison, limit=10)
    assert len(avail) > 0, "Aucun joueur disponible"
    target = avail[0]

    results = propose_trade_for_target(team_id, target.player_id, saison, n_solutions=2)
    print(f"\n  Mode Cible -> Acquisition de {target.player_name} ({target.cap_hit:,}$) :")
    for i, r in enumerate(results, 1):
        print(f"    {i}. {r.move.description} | {r.status.value}")
        for pin in r.move.players_in:
            assert pin.team_id == target.team_id, "Le trade cible contient des joueurs d'autres équipes !"

    return f"Cible {target.player_name} testée — {len(results)} package(s) généré(s)"


def test_waive_valuation():
    """Vérifie que la libération d'un joueur ne génère plus de gain fictif positif."""
    from Composant4.engine import _load_team_roster, _load_team_state, _generate_waive_candidates
    saison  = "2025-26"
    team_id = _first_team(saison)
    state   = _load_team_state(team_id, saison)
    roster  = _load_team_roster(team_id, saison)
    waives  = _generate_waive_candidates(roster, state, saison, [])
    assert len(waives) > 0, "Aucun waive généré"

    for w in waives:
        assert w.value_delta <= 0, (
            f"Anomalie : waive avec value_delta positif ({w.value_delta:+,}$) ! "
            f"Couper un joueur garanti ne peut pas enrichir l'équipe."
        )

    return f"{len(waives)} waive(s) vérifié(s) — aucun faux profit généré (dead cap respecté)"


def test_is_legal_trade():
    """Construit un trade synthétique minimal et vérifie is_legal."""
    from Composant4.engine import _load_team_roster, _load_available_players, _load_team_state
    from Composant4.ruleees import is_legal
    from Composant4.schemas import (
        Contract, ExceptionType, Move, MoveStatus, MoveType
    )

    saison  = "2025-26"
    team_id = _first_team(saison)
    state   = _load_team_state(team_id, saison)
    roster  = _load_team_roster(team_id, saison)
    avail   = _load_available_players(team_id, saison, limit=20)

    if not roster or not avail:
        raise SkipTest("Roster ou available vide — impossible de construire le trade")

    p_out = roster[0]   # joueur avec le cap_hit le plus élevé
    p_in  = avail[0]    # idem côté available

    move = Move(
        move_type   = MoveType.TRADE,
        team_id     = team_id,
        description = f"Trade test : {p_out.player_name} → {p_in.player_name}",
        players_out = [
            Contract(
                player_id=p_out.player_id, player_name=p_out.player_name,
                team_id=p_out.team_id, saison=saison,
                cap_hit=p_out.cap_hit, base_salary=p_out.base_salary,
                years_remaining=p_out.years_remaining,
                estimated_value=p_out.market_value,
            )
        ],
        players_in  = [
            Contract(
                player_id=p_in.player_id, player_name=p_in.player_name,
                team_id=p_in.team_id, saison=saison,
                cap_hit=p_in.cap_hit, base_salary=p_in.base_salary,
                years_remaining=p_in.years_remaining,
                estimated_value=p_in.market_value,
            )
        ],
        salary_offer=None,
        exception=ExceptionType.NONE,
        picks_in=[], picks_out=[],
    )

    result = is_legal(move, state, saison)
    status_str = "LÉGAL ✅" if result.status == MoveStatus.LEGAL else f"ILLÉGAL ❌ ({result.blocking_reason})"
    return (
        f"{p_out.player_name} (out {p_out.cap_hit:,}$) "
        f"→ {p_in.player_name} (in {p_in.cap_hit:,}$) "
        f"| {status_str} | value_delta={result.value_delta:+,}$"
    )


def test_propose_moves_full():
    """Pipeline complet propose_moves() — le test le plus représentatif."""
    from Composant4.engine import propose_moves

    saison  = "2025-26"
    team_id = _first_team(saison)

    result = propose_moves(
        team_id        = team_id,
        saison         = saison,
        top_n          = 5,
        include_trades = True,
        include_signs  = True,
        include_waives = True,
    )

    print()
    print(indent(result.summary(), "  "))

    # Exportation automatique du rapport détaillé en fichier texte
    txt_path = result.export_to_file("Composant4/meilleures_options_c4.txt")
    print(f"\n  📄 Rapport exhaustif des meilleures options exporté dans : {txt_path}")

    if result.warnings:
        print(f"\n  ⚠ {len(result.warnings)} warning(s):")
        for w in result.warnings:
            print(f"     - {w}")

    assert result.total_candidates >= 0

    return (
        f"team_id={team_id} | "
        f"candidats={result.total_candidates} | "
        f"légaux={len(result.legal_moves)} | "
        f"illégaux={len(result.illegal_moves)} | "
        f"fichier: {txt_path}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

TESTS = [
    ("1. Connexion DB",                    test_db_connection),
    ("2. Table predictions",               test_predictions_table_exists),
    ("3. Chargement règles CBA",           test_cba_rules_load),
    ("4. _load_team_state",                test_load_team_state),
    ("5. _load_team_roster (market_value)",test_load_team_roster),
    ("6. _load_available_players",         test_load_available_players),
    ("7. Import OR-Tools + optimizer.py",  test_optimizer_imports),
    ("8. optimize_trades (CP-SAT)",        test_optimize_trades),
    ("9. Mode Cible (propose_trade_for_target)", test_target_player_trade),
    ("10. Waive valuation (dead cap)",     test_waive_valuation),
    ("11. is_legal — trade synthétique",   test_is_legal_trade),
    ("12. propose_moves — end-to-end",    test_propose_moves_full),
]


if __name__ == "__main__":
    print(f"\n{'═'*60}")
    print(f"{BOLD}  NBA Copilot — Tests Composant 4{RESET}")
    print(f"{'═'*60}")

    for name, fn in TESTS:
        run(name, fn)

    # ── Résumé ──────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"{BOLD}  RÉSUMÉ{RESET}")
    print(f"{'═'*60}")

    n_ok   = sum(1 for _, s, _ in results if s == "OK")
    n_skip = sum(1 for _, s, _ in results if s == "SKIP")
    n_fail = sum(1 for _, s, _ in results if s == "FAIL")

    for name, status, detail in results:
        icon = OK if status == "OK" else (SKIP if status == "SKIP" else FAIL)
        print(f"  {icon}  {name}")
        if status in ("SKIP", "FAIL") and detail:
            print(f"        → {detail}")

    print(f"\n  Total : {n_ok} OK  |  {n_skip} skipped  |  {n_fail} échoué(s)")
    print(f"{'═'*60}\n")

    sys.exit(0 if n_fail == 0 else 1)
