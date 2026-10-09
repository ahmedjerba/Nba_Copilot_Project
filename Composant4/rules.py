# Composant4/rules.py

from __future__ import annotations
from Composant4.schemas import (
    Contract, Move, MoveType, MoveResult, MoveStatus,
    RuleResult, TeamPayroll, ExceptionType
)

from Composant4.cba_loader import load_cba_rules


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ok(rule: str, reason: str) -> RuleResult:
    return RuleResult(rule=rule, passed=True, reason=reason)

def _fail(rule: str, reason: str) -> RuleResult:
    return RuleResult(rule=rule, passed=False, reason=reason)


# ---------------------------------------------------------------------------
# Règle 1 — Salary Cap
# ---------------------------------------------------------------------------

def check_salary_cap(state: TeamPayroll, move: Move, rules: dict) -> RuleResult:
    """
    Une équipe sous le cap ne peut pas dépasser le cap
    sauf via les exceptions (MLE, BAE).
    Une équipe déjà au-dessus peut rester au-dessus
    si elle utilise une exception valide.
    """
    RULE = "check_salary_cap"
    cap = rules["thresholds"]["salary_cap"]

    cap_after = state.total_cap_hit + move.salary_delta

    if move.move_type == MoveType.TRADE:
        # Un trade est toujours permis même au-dessus du cap
        # (le matching est vérifié séparément)
        return _ok(RULE, f"Trade — cap après : {cap_after:,}$ (règle matching s'applique)")

    if move.move_type == MoveType.SIGN:
        if state.total_cap_hit >= cap:
            # Au-dessus du cap → doit utiliser une exception
            if move.exception == ExceptionType.NONE:
                return _fail(
                    RULE,
                    f"Équipe au-dessus du cap ({state.total_cap_hit:,}$). "
                    f"Une exception (MLE, BAE) est requise pour signer."
                )
            return _ok(RULE, f"Signature via exception {move.exception.value} — cap après : {cap_after:,}$")

        if cap_after > cap:
            # Sous le cap mais la signature dépasse
            cap_space = cap - state.total_cap_hit
            return _fail(
                RULE,
                f"Signature de {move.salary_offer:,}$ dépasse le cap space disponible "
                f"({cap_space:,}$). Cap après : {cap_after:,}$."
            )

        return _ok(RULE, f"Signature dans le cap space — cap après : {cap_after:,}$")

    if move.move_type == MoveType.WAIVE:
        return _ok(RULE, f"Libération — cap après : {cap_after:,}$")

    return _ok(RULE, "Move autorisé")


# ---------------------------------------------------------------------------
# Règle 2 — Trade Matching
# ---------------------------------------------------------------------------

def check_trade_matching(state: TeamPayroll, move: Move, rules: dict) -> RuleResult:
    """
    Règle CBA trade matching :
    - Équipe SOUS le cap  : peut recevoir jusqu'à salary_out * 1.25 + 100k$
    - Équipe AU-DESSUS    : peut recevoir jusqu'à salary_out + 5M$
    Les picks ne comptent PAS dans le matching.
    """
    RULE = "check_trade_matching"

    if move.move_type != MoveType.TRADE:
        return _ok(RULE, "Non applicable — pas un trade")

    cap        = rules["thresholds"]["salary_cap"]
    pct        = rules["trade_matching"]["pct_above"]           # 0.25
    flat       = rules["trade_matching"]["flat_bonus"]           # 100_000
    above_flat = rules["trade_matching"]["above_cap_threshold"]  # 5_000_000

    salary_out = move.salary_out
    salary_in  = move.salary_in

    if salary_out == 0 and salary_in == 0:
        return _ok(RULE, "Trade de picks uniquement — pas de salary matching requis")

    # Calcul du max recevable
    if state.total_cap_hit <= cap:
        max_receivable = int(salary_out * (1 + pct)) + flat
        rule_label = f"sous le cap (max = salary_out × {1+pct} + {flat:,}$)"
    else:
        max_receivable = salary_out + above_flat
        rule_label = f"au-dessus du cap (max = salary_out + {above_flat:,}$)"

    if salary_in > max_receivable:
        return _fail(
            RULE,
            f"Salary matching dépassé. "
            f"Envoyé : {salary_out:,}$ | Reçu : {salary_in:,}$ | "
            f"Max autorisé : {max_receivable:,}$ ({rule_label})."
        )

    return _ok(
        RULE,
        f"Matching OK. Envoyé : {salary_out:,}$ | Reçu : {salary_in:,}$ | "
        f"Max autorisé : {max_receivable:,}$."
    )


# ---------------------------------------------------------------------------
# Règle 3 — Luxury Tax / Aprons
# ---------------------------------------------------------------------------

def check_luxury_tax(state: TeamPayroll, move: Move, rules: dict) -> RuleResult:
    """
    Vérifie les restrictions liées aux seuils tax / apron 1 / apron 2.
    - Apron 1 : interdit MLE non-taxpayer, BAE, trade aggregation
    - Apron 2 : interdit MLE, BAE, aggregation, S&T entrant
    """
    RULE = "check_luxury_tax"

    thresholds  = rules["thresholds"]
    apron1      = thresholds.get("first_apron")
    apron2      = thresholds.get("second_apron")
    ap1_rules   = rules.get("apron1_restrictions") or {}
    ap2_rules   = rules.get("apron2_restrictions") or {}

    cap_after = state.total_cap_hit + move.salary_delta

    # Apron 2
    if apron2 and cap_after > apron2:
        if move.move_type == MoveType.SIGN and move.exception != ExceptionType.NONE:
            return _fail(
                RULE,
                f"Équipe au-dessus du second apron ({apron2:,}$) après le move "
                f"({cap_after:,}$). Aucune exception autorisée."
            )
        if ap2_rules.get("no_trade_aggregation") and move.move_type == MoveType.TRADE:
            if len(move.players_in) > 1:
                return _fail(
                    RULE,
                    f"Second apron dépassé ({cap_after:,}$) : "
                    f"impossible d'agréger plusieurs salaires entrants dans un trade."
                )

    # Apron 1
    if apron1 and cap_after > apron1:
        if ap1_rules.get("no_mle_non_taxpayer"):
            if move.exception == ExceptionType.MLE_NON_TAXPAYER:
                return _fail(
                    RULE,
                    f"Équipe au-dessus du first apron ({apron1:,}$) après le move "
                    f"({cap_after:,}$). Le MLE non-taxpayer est interdit."
                )
        if ap1_rules.get("no_bi_annual"):
            if move.exception == ExceptionType.BI_ANNUAL:
                return _fail(
                    RULE,
                    f"Équipe au-dessus du first apron ({apron1:,}$). "
                    f"Le bi-annual exception est interdit."
                )

    return _ok(
        RULE,
        f"Seuils OK. Masse salariale après : {cap_after:,}$."
    )


# ---------------------------------------------------------------------------
# Règle 4 — Exceptions (MLE / BAE)
# ---------------------------------------------------------------------------

def check_exceptions(state: TeamPayroll, move: Move, rules: dict) -> RuleResult:
    """
    Vérifie que l'exception utilisée est :
    1. Disponible (pas déjà utilisée cette saison)
    2. Dans la limite autorisée
    3. Compatible avec le statut de l'équipe (taxpayer / apron)
    """
    RULE = "check_exceptions"

    if move.exception == ExceptionType.NONE:
        return _ok(RULE, "Aucune exception utilisée")

    exc     = rules["exceptions"]
    thres   = rules["thresholds"]
    tax     = thres["luxury_tax"]
    apron1  = thres.get("first_apron")

    cap_after = state.total_cap_hit + move.salary_delta

    # MLE Non-Taxpayer
    if move.exception == ExceptionType.MLE_NON_TAXPAYER:
        if state.mle_used:
            return _fail(RULE, "MLE non-taxpayer déjà utilisé cette saison.")
        if state.total_cap_hit >= tax:
            return _fail(
                RULE,
                f"Équipe taxpayer ({state.total_cap_hit:,}$ ≥ {tax:,}$). "
                f"Le MLE non-taxpayer n'est pas disponible."
            )
        if apron1 and cap_after > apron1:
            return _fail(
                RULE,
                f"L'utilisation du MLE non-taxpayer ferait dépasser "
                f"le first apron ({apron1:,}$). Cap après : {cap_after:,}$."
            )
        max_mle = exc["mle_non_taxpayer"]
        if move.salary_offer and move.salary_offer > max_mle:
            return _fail(
                RULE,
                f"Offre ({move.salary_offer:,}$) dépasse le MLE non-taxpayer "
                f"({max_mle:,}$)."
            )
        return _ok(RULE, f"MLE non-taxpayer valide — offre : {move.salary_offer:,}$")

    # MLE Taxpayer
    if move.exception == ExceptionType.MLE_TAXPAYER:
        if state.mle_used:
            return _fail(RULE, "MLE taxpayer déjà utilisé cette saison.")
        max_mle = exc["mle_taxpayer"]
        if move.salary_offer and move.salary_offer > max_mle:
            return _fail(
                RULE,
                f"Offre ({move.salary_offer:,}$) dépasse le MLE taxpayer "
                f"({max_mle:,}$)."
            )
        return _ok(RULE, f"MLE taxpayer valide — offre : {move.salary_offer:,}$")

    # Bi-Annual Exception
    if move.exception == ExceptionType.BI_ANNUAL:
        if state.bi_annual_used:
            return _fail(RULE, "Bi-annual exception déjà utilisée cette saison.")
        if state.total_cap_hit >= tax:
            return _fail(
                RULE,
                f"Équipe taxpayer ({state.total_cap_hit:,}$). "
                f"Le bi-annual exception n'est pas disponible."
            )
        max_bae = exc["bi_annual"]
        if move.salary_offer and move.salary_offer > max_bae:
            return _fail(
                RULE,
                f"Offre ({move.salary_offer:,}$) dépasse le bi-annual "
                f"({max_bae:,}$)."
            )
        return _ok(RULE, f"Bi-annual exception valide — offre : {move.salary_offer:,}$")

    return _ok(RULE, "Exception non reconnue — ignorée")


# ---------------------------------------------------------------------------
# Règle 5 — Roster Size
# ---------------------------------------------------------------------------

def check_roster_size(state: TeamPayroll, move: Move, rules: dict) -> RuleResult:
    """
    Vérifie que le roster reste entre 14 et 15 joueurs après le move.
    """
    RULE = "check_roster_size"

    roster   = rules["roster"]
    min_p    = roster["min_players"]   # 14
    max_p    = roster["max_players"]   # 15

    delta = len(move.players_in) - len(move.players_out)
    size_after = state.nb_joueurs + delta

    if size_after > max_p:
        return _fail(
            RULE,
            f"Roster après le move : {size_after} joueurs (max {max_p}). "
            f"Libérez un joueur avant."
        )
    if size_after < min_p:
        return _fail(
            RULE,
            f"Roster après le move : {size_after} joueurs (min {min_p})."
        )

    return _ok(RULE, f"Roster OK — {size_after} joueurs après le move.")


# ---------------------------------------------------------------------------
# Orchestrateur — is_legal
# ---------------------------------------------------------------------------

def is_legal(move: Move, state: TeamPayroll, saison: str = "2024-25") -> MoveResult:
    """
    Appelle toutes les règles dans l'ordre.
    Retourne le premier échec ou OK si toutes passent.
    """
    rules = load_cba_rules(saison)

    checks = [
        check_salary_cap(state, move, rules),
        check_trade_matching(state, move, rules),
        check_luxury_tax(state, move, rules),
        check_exceptions(state, move, rules),
        check_roster_size(state, move, rules),
    ]

    cap_after   = state.total_cap_hit + move.salary_delta
    cap_space   = rules["thresholds"]["salary_cap"] - cap_after
    value_delta = move.value_delta

    if value_delta > 1_000_000:
        value_signal = f"GAIN 🟢 +{value_delta:,}$"
    elif value_delta < -1_000_000:
        value_signal = f"PERTE 🔴 {value_delta:,}$"
    else:
        value_signal = "NEUTRE ⚪"

    # Premier échec
    for result in checks:
        if not result.passed:
            return MoveResult(
                move            = move,
                status          = MoveStatus.ILLEGAL,
                rule_results    = checks,
                blocking_rule   = result.rule,
                blocking_reason = result.reason,
                cap_hit_after   = cap_after,
                cap_space_after = cap_space,
                value_delta     = value_delta,
                value_signal    = value_signal,
            )

    return MoveResult(
        move            = move,
        status          = MoveStatus.LEGAL,
        rule_results    = checks,
        blocking_rule   = None,
        blocking_reason = None,
        cap_hit_after   = cap_after,
        cap_space_after = cap_space,
        value_delta     = value_delta,
        value_signal    = value_signal,
    )