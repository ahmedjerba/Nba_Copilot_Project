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