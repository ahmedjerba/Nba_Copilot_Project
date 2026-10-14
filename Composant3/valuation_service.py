# ─────────────────────────────────────────────
# Composant3 / valuation_service.py
# get_player_value — outil exposé à l'agent
# ─────────────────────────────────────────────

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from Composant3.Valorisation.config import DB_URL, FEATURE_VERSION, SALARY_CAP
from Composant3.Valorisation.feature_engineering import build_features
from Composant3.Valorisation.model_registry import load_model
from Composant3.schemas import (
    PlayerValueRequest,
    PlayerValueResponse,
    FacteurExplicatif,
)

# ── Seuils de segment ─────────────────────────────────────────────────────────
CAP_PCT_MAX        = 0.30   # au-dessus → contrat max, hors modèle
CAP_PCT_LOW        = 0.05   # en-dessous → utiliser modèle middle_class
CAP_PCT_TWOWAY     = 0.001  # en-dessous → two-way, hors modèle

# ── Chargement des modèles au démarrage ───────────────────────────────────────
_payload_market = load_model(version="market_v1")
_model_market   = _payload_market["model"]
_flags_market   = _payload_market["fe_flags"]

_payload_middle = load_model(version="middle_v1")  # ← à créer
_model_middle   = _payload_middle["model"]
_flags_middle   = _payload_middle["fe_flags"]


# ══════════════════════════════════════════════════════════════════════════════
# Fonction principale
# ══════════════════════════════════════════════════════════════════════════════

def get_player_value(
    player_id: int,
    saison:    str,
) -> PlayerValueResponse:
    """
    Calcule la valeur marché et le surplus d'un joueur.

    Routing des modèles :
        cap_hit_pct < 0.001  → HORS MODÈLE (two-way)
        cap_hit_pct < 0.05   → modèle middle_class (rookie/vétéran min)
        cap_hit_pct ≤ 0.30   → modèle market (coeur)
        cap_hit_pct > 0.30   → HORS MODÈLE (contrat max)
    """
    engine = create_engine(DB_URL)

    with engine.connect() as conn:
        season_df = pd.read_sql(text("""
            SELECT pf.*, p.position, p.player_name,
                   p.draft_year, p.draft_round, p.draft_number
            FROM player_features pf
            LEFT JOIN players p ON pf.player_id = p.player_id
            WHERE pf.saison          = :saison
            AND   pf.feature_version = :v
        """), conn, params={"saison": saison, "v": FEATURE_VERSION})

        contract_df = pd.read_sql(text("""
            SELECT cap_hit FROM contracts
            WHERE player_id = :pid
            AND   saison    = :saison
        """), conn, params={"pid": player_id, "saison": saison})

    # ── Vérifications de base ─────────────────────────────────────────────
    if season_df.empty:
        return _error(player_id, saison,
                      f"Aucune feature trouvée pour la saison {saison}.")

    player_season = season_df[season_df["player_id"] == player_id]
    if player_season.empty:
        return _error(player_id, saison,
                      f"Joueur {player_id} introuvable pour {saison}.")

    player_name = player_season["player_name"].iloc[0]

    cap_saison = SALARY_CAP.get(saison)
    if cap_saison is None:
        return _error(player_id, saison,
                      f"Salary cap inconnu pour {saison}.")

    # ── Routing selon cap_hit_pct ─────────────────────────────────────────
    if contract_df.empty:
        # Pas de contrat → modèle market par défaut
        model_a_utiliser = _model_market
        flags_a_utiliser = _flags_market
        cap_hit          = None
        cap_hit_pct      = None
        avertissement    = None

    else:
        cap_hit     = float(contract_df["cap_hit"].iloc[0])
        cap_hit_pct = cap_hit / cap_saison

        # Two-way
        if cap_hit_pct < CAP_PCT_TWOWAY:
            return PlayerValueResponse(
                player_id=player_id,
                player_name=player_name,
                saison=saison,
                valeur_dollars=None,
                valeur_pct=None,
                p10_dollars=None,
                p90_dollars=None,
                salary_cap=cap_saison,
                cap_hit=round(cap_hit),
                surplus=None,
                signal="HORS MODÈLE 🚫",
                top_facteurs=[],
                erreur="Two-way contract — non modélisable.",
            )

        # Contrat max
        if cap_hit_pct > CAP_PCT_MAX:
            return PlayerValueResponse(
                player_id=player_id,
                player_name=player_name,
                saison=saison,
                valeur_dollars=None,
                valeur_pct=None,
                p10_dollars=None,
                p90_dollars=None,
                salary_cap=cap_saison,
                cap_hit=round(cap_hit),
                surplus=None,
                signal="HORS MODÈLE 🚫",
                top_facteurs=[],
                erreur=(
                    f"Contrat max ({cap_hit_pct:.1%} du cap) — "
                    "valeur non modélisable. Facteurs non-statistiques "
                    "(franchise player, leadership, marketabilité)."
                ),
            )

        # Rookie / vétéran minimum → modèle middle_class
        if cap_hit_pct < CAP_PCT_LOW:
            model_a_utiliser = _model_middle
            flags_a_utiliser = _flags_middle
            avertissement    = (
                f"Joueur sous 5% du cap ({cap_hit_pct:.1%}) — "
                "modèle middle_class utilisé comme proxy."
            )
        else:
            # Coeur du modèle market
            model_a_utiliser = _model_market
            flags_a_utiliser = _flags_market
            avertissement    = None

    # ── Feature engineering ───────────────────────────────────────────────
    season_fe, _, _ = build_features(season_df, **flags_a_utiliser)
    player_fe = season_fe[season_fe["player_id"] == player_id].copy()

    # ── Prédiction ────────────────────────────────────────────────────────
    pred         = model_a_utiliser.predict_player(player_fe, salary_cap=cap_saison)
    top_facteurs = _get_top_facteurs(model_a_utiliser, n=5)

    # ── Surplus ───────────────────────────────────────────────────────────
    if cap_hit is None:
        return PlayerValueResponse(
            player_id=player_id,
            player_name=player_name,
            saison=saison,
            **pred,
            cap_hit=None,
            surplus=None,
            signal="SANS CONTRAT",
            top_facteurs=top_facteurs,
            erreur=avertissement,
        )

    surplus = pred["valeur_dollars"] - cap_hit
    signal  = "SOUS-PAYÉ 🟢" if surplus > 0 else "SUR-PAYÉ 🔴"

    return PlayerValueResponse(
        player_id=player_id,
        player_name=player_name,
        saison=saison,
        **pred,
        cap_hit=round(cap_hit),
        surplus=round(surplus),
        signal=signal,
        top_facteurs=top_facteurs,
        erreur=avertissement,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _get_top_facteurs(
    model,
    n: int = 5,
) -> list[FacteurExplicatif]:
    try:
        params = model.glm_model.params
        params = params.drop("Intercept", errors="ignore")
        top    = params.abs().nlargest(n)
        return [
            FacteurExplicatif(
                feature=feat,
                coefficient=round(float(params[feat]), 4),
                direction="hausse" if params[feat] > 0 else "baisse",
            )
            for feat in top.index
        ]
    except Exception:
        return []


def _error(
    player_id: int,
    saison:    str,
    message:   str,
) -> PlayerValueResponse:
    return PlayerValueResponse(
        player_id=player_id,
        player_name=None,
        saison=saison,
        valeur_dollars=None,
        valeur_pct=None,
        p10_dollars=None,
        p90_dollars=None,
        salary_cap=None,
        cap_hit=None,
        surplus=None,
        signal=None,
        top_facteurs=[],
        erreur=message,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Batch valuation — Composant 4 bridge
# ══════════════════════════════════════════════════════════════════════════════

def get_bulk_valuations(
    player_ids: list[int],
    saison:     str,
) -> dict[int, int | None]:
    """
    Calcule le surplus_value (valeur_marché − cap_hit) pour une liste
    de joueurs en un seul passage sur la DB.

    Utilisé par le Composant 4 (engine.py) pour enrichir market_value
    sans créer de table predictions intermédiaire.

    Args:
        player_ids : liste d'IDs joueurs à valoriser
        saison     : ex. "2025-26" — doit correspondre à contracts.saison

    Returns:
        dict {player_id: surplus_value_en_dollars | None}
        None = joueur hors-modèle (max contract, two-way, features absentes)
    """
    import logging
    _log = logging.getLogger(__name__)

    if not player_ids:
        return {}

    cap_saison = SALARY_CAP.get(saison)
    if cap_saison is None:
        _log.warning("get_bulk_valuations : salary_cap inconnu pour %s", saison)
        return {pid: None for pid in player_ids}

    engine_db = create_engine(DB_URL)

    try:
        with engine_db.connect() as conn:
            # ── 1. Features de toute la saison (1 seule requête) ─────────────
            season_df = pd.read_sql(text("""
                SELECT pf.*, p.position, p.player_name,
                       p.draft_year, p.draft_round, p.draft_number
                FROM player_features pf
                LEFT JOIN players p ON pf.player_id = p.player_id
                WHERE pf.saison          = :saison
                AND   pf.feature_version = :v
            """), conn, params={"saison": saison, "v": FEATURE_VERSION})

            # ── 2. Contrats de tous les joueurs d'un coup ─────────────────────
            if not season_df.empty:
                pids_str = ",".join(str(p) for p in player_ids)
                contracts_df = pd.read_sql(text(f"""
                    SELECT player_id, cap_hit
                    FROM   contracts
                    WHERE  player_id IN ({pids_str})
                    AND    saison    = :saison
                """), conn, params={"saison": saison})
            else:
                contracts_df = pd.DataFrame(columns=["player_id", "cap_hit"])
    except Exception as exc:
        _log.error("get_bulk_valuations : erreur DB — %s", exc)
        return {pid: None for pid in player_ids}
    finally:
        engine_db.dispose()

    if season_df.empty:
        _log.warning(
            "get_bulk_valuations : aucune feature pour saison=%s "
            "(player_features vide ou feature_version incorrecte)", saison
        )
        return {pid: None for pid in player_ids}

    # Dict cap_hit par player_id
    contracts_map: dict[int, float] = {
        int(r["player_id"]): float(r["cap_hit"])
        for _, r in contracts_df.iterrows()
    }

    # ── 3. Build features une fois par modèle (pas N fois) ───────────────────
    try:
        season_fe_market, _, _ = build_features(season_df, **_flags_market)
    except Exception as exc:
        _log.error("get_bulk_valuations : build_features market failed — %s", exc)
        season_fe_market = pd.DataFrame()

    try:
        season_fe_middle, _, _ = build_features(season_df, **_flags_middle)
    except Exception as exc:
        _log.error("get_bulk_valuations : build_features middle failed — %s", exc)
        season_fe_middle = pd.DataFrame()

    # ── 4. Prédiction par joueur ──────────────────────────────────────────────
    results: dict[int, int | None] = {}

    for player_id in player_ids:
        try:
            cap_hit = contracts_map.get(player_id)
            if cap_hit is None:
                results[player_id] = None
                continue

            cap_hit_pct = cap_hit / cap_saison

            # Hors modèle : two-way ou max contract
            if cap_hit_pct < CAP_PCT_TWOWAY or cap_hit_pct > CAP_PCT_MAX:
                results[player_id] = None
                continue

            # Choix du modèle
            if cap_hit_pct < CAP_PCT_LOW:
                fe_season = season_fe_middle
                model     = _model_middle
            else:
                fe_season = season_fe_market
                model     = _model_market

            if fe_season.empty:
                results[player_id] = None
                continue

            player_fe = fe_season[fe_season["player_id"] == player_id].copy()
            if player_fe.empty:
                results[player_id] = None
                continue

            pred    = model.predict_player(player_fe, salary_cap=cap_saison)
            surplus = pred["valeur_dollars"] - cap_hit
            results[player_id] = round(surplus)

        except Exception as exc:
            _log.debug(
                "get_bulk_valuations : erreur pour player_id=%s — %s",
                player_id, exc
            )
            results[player_id] = None

    n_valued = sum(1 for v in results.values() if v is not None)
    _log.info(
        "get_bulk_valuations : %d/%d joueurs valorisés (saison=%s)",
        n_valued, len(player_ids), saison
    )
    return results