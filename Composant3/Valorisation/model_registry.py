# ─────────────────────────────────────────────
# Composant3 / model_registry.py
# Sérialisation et chargement du modèle entraîné
# ─────────────────────────────────────────────

import joblib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from Composant3.Valorisation.config import SALARY_CAP_REF
from Composant3.Valorisation.feature_engineering import build_features, CAT_FEATURES
from Composant3.Valorisation.model import BetaValuationModel


# ── Chemins ───────────────────────────────────────────────────────────────────
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Sauvegarde
# ══════════════════════════════════════════════════════════════════════════════

def save_model(
    model: BetaValuationModel,
    fe_flags: dict,
    num_features: list[str],
    mae_oos: float,
    version: str | None = None,
) -> Path:
    """
    Sérialise le modèle + tout ce qu'il faut pour le réutiliser sans notebook.

    Sauvegarde deux fichiers :
        models/model_{version}.pkl   — le modèle (joblib)
        models/model_{version}.json  — les métadonnées (lisibles)

    Retourne le chemin du .pkl.
    """
    if version is None:
        version = datetime.now().strftime("%Y%m%d_%H%M")

    payload = {
        "model":        model,
        "fe_flags":     fe_flags,
        "num_features": num_features,
        "cat_features": CAT_FEATURES,
        "alpha":        model.alpha,
        "version":      version,
    }

    pkl_path  = MODELS_DIR / f"model_{version}.pkl"
    json_path = MODELS_DIR / f"model_{version}.json"

    # Sauvegarder le modèle
    joblib.dump(payload, pkl_path)

    # Sauvegarder les métadonnées lisibles
    meta = {
        "version":       version,
        "saved_at":      datetime.now().isoformat(),
        "alpha":         model.alpha,
        "fe_flags":      fe_flags,
        "num_features":  num_features,
        "cat_features":  CAT_FEATURES,
        "n_features":    len(num_features),
        "mae_oos":       round(mae_oos),
        "salary_cap_ref": SALARY_CAP_REF,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[model_registry] ✅ Modèle sauvegardé → {pkl_path.name}")
    print(f"[model_registry]    MAE OOS : {mae_oos:,.0f}$")
    print(f"[model_registry]    Features : {len(num_features)} num + {len(CAT_FEATURES)} cat")
    print(f"[model_registry]    alpha Ridge : {model.alpha}")

    return pkl_path


# ══════════════════════════════════════════════════════════════════════════════
# Chargement
# ══════════════════════════════════════════════════════════════════════════════

def load_model(version: str | None = None) -> dict:
    """
    Charge le modèle depuis le .pkl.
    version=None → charge le plus récent automatiquement.

    Retourne le payload complet :
        {model, fe_flags, num_features, cat_features, alpha, version}
    """
    if version is None:
        pkls = sorted(MODELS_DIR.glob("model_*.pkl"))
        if not pkls:
            raise FileNotFoundError(
                f"Aucun modèle trouvé dans {MODELS_DIR}. "
                "Lance d'abord save_model()."
            )
        pkl_path = pkls[-1]  # le plus récent
    else:
        pkl_path = MODELS_DIR / f"model_{version}.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(f"Modèle introuvable : {pkl_path}")

    payload = joblib.load(pkl_path)
    print(f"[model_registry] ✅ Modèle chargé ← {pkl_path.name}")
    print(f"[model_registry]    Features : {len(payload['num_features'])} num "
          f"+ {len(payload['cat_features'])} cat")
    print(f"[model_registry]    alpha Ridge : {payload['alpha']}")

    return payload


# ══════════════════════════════════════════════════════════════════════════════
# Lister les versions disponibles
# ══════════════════════════════════════════════════════════════════════════════

def list_models() -> pd.DataFrame:
    """
    Retourne un DataFrame avec toutes les versions disponibles et leurs métriques.
    """
    rows = []
    for json_path in sorted(MODELS_DIR.glob("model_*.json")):
        with open(json_path, encoding="utf-8") as f:
            meta = json.load(f)
        rows.append({
            "version":    meta["version"],
            "saved_at":   meta["saved_at"],
            "n_features": meta["n_features"],
            "alpha":      meta["alpha"],
            "mae_oos":    meta.get("mae_oos", None),
        })

    if not rows:
        print("[model_registry] Aucun modèle sauvegardé.")
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("saved_at", ascending=False)