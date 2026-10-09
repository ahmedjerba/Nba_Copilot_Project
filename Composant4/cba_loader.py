# Composant4/cba_loader.py
"""
Charge les règles CBA depuis le fichier YAML correspondant à la saison.
Cache en mémoire pour éviter de relire le disque à chaque appel.
"""

from __future__ import annotations

import functools
import logging
import os
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# Répertoire où sont stockés les fichiers YAML
RULES_DIR = Path(__file__).parent / "rules"

# Cache en mémoire : saison → dict de règles
_cache: dict[str, dict] = {}


def load_cba_rules(saison: str = "2025-26") -> dict:
    """
    Charge et retourne les règles CBA pour une saison donnée.

    Args:
        saison : format "YYYY-YY" (ex: "2025-26")

    Returns:
        dict avec les clés : thresholds, exceptions,
        trade_matching, apron1_restrictions,
        apron2_restrictions, roster

    Raises:
        FileNotFoundError : si le fichier YAML n'existe pas
        ValueError        : si le YAML est malformé
    """
    if saison in _cache:
        return _cache[saison]

    # Cherche d'abord cba_{saison}.yaml puis cba_rules_{saison}.yaml
    candidates = [
        RULES_DIR / f"cba_{saison}.yaml",
        RULES_DIR / f"cba_rules_{saison}.yaml",
        RULES_DIR / f"cba_{saison.replace('-', '_')}.yaml",
    ]

    yaml_path: Path | None = None
    for path in candidates:
        if path.exists():
            yaml_path = path
            break

    if yaml_path is None:
        searched = [str(p) for p in candidates]
        raise FileNotFoundError(
            f"Aucun fichier CBA trouvé pour la saison '{saison}'.\n"
            f"Chemins cherchés :\n" + "\n".join(f"  - {p}" for p in searched)
        )

    try:
        with yaml_path.open("r", encoding="utf-8") as f:
            rules = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"YAML malformé dans {yaml_path}: {e}") from e

    # Validation minimale des clés obligatoires
    required_keys = {"thresholds", "exceptions", "trade_matching", "roster"}
    missing = required_keys - set(rules.keys())
    if missing:
        raise ValueError(
            f"Fichier CBA incomplet — clés manquantes : {missing}"
        )

    logger.info("CBA rules chargées depuis %s", yaml_path)
    _cache[saison] = rules
    return rules


def clear_cache() -> None:
    """Vide le cache — utile pour les tests."""
    _cache.clear()
    logger.debug("Cache CBA vidé")


def list_available_saisons() -> list[str]:
    """
    Retourne la liste des saisons disponibles dans le répertoire rules/.
    """
    if not RULES_DIR.exists():
        return []

    saisons = []
    for f in RULES_DIR.glob("cba*.yaml"):
        # Extrait la saison depuis le nom de fichier
        name = f.stem  # ex: "cba_2025-26"
        parts = name.split("_", 1)
        if len(parts) == 2:
            saisons.append(parts[1])

    return sorted(saisons)