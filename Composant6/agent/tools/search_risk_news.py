"""
agent/tools/search_risk_news.py
===============================
Outil de veille et détection de signaux de risque via API externe (blessures, incidents, rumeurs).
"""

from __future__ import annotations
from datetime import datetime
from typing import Any, Optional
import httpx

from agent.schemas.inputs import SearchRiskNewsInput
from agent.schemas.outputs import (
    SearchRiskNewsOutput,
    RiskItem,
    RiskSeverity,
)
from agent.tools.registry import register_tool

RISK_NEWS_API_URL = "https://api.risknews.internal/search"
DEFAULT_TIMEOUT_SECONDS = 10.0


def _parse_severity(val: Any) -> RiskSeverity:
    """Normalise la sévérité vers l'énumération RiskSeverity."""
    if isinstance(val, RiskSeverity):
        return val
    val_str = str(val or "low").lower().strip()
    try:
        return RiskSeverity(val_str)
    except ValueError:
        return RiskSeverity.MEDIUM


@register_tool(
    name="search_risk_news",
    description="Recherche des signaux de risque sur un joueur (blessures, incidents, rumeurs). Source obligatoire sur chaque item retourné.",
    input_model=SearchRiskNewsInput,
)
async def search_risk_news(inp: SearchRiskNewsInput) -> SearchRiskNewsOutput:
    """
    Interroge l'API web externe RiskNews pour identifier les signaux d'alerte
    (blessures récentes, litiges, demandes de trade, suspensions) associés à un joueur.
    """
    categories_payload = [
        c.value if hasattr(c, "value") else str(c)
        for c in inp.categories
    ]

    params = {
        "player_id": inp.player_id,
        "player_name": inp.player_name,
        "categories": categories_payload,
        "lookback_days": inp.lookback_days,
        "max_results": inp.max_results,
    }

    # Appel web externe via httpx.AsyncClient avec gestion rigoureuse timeout & code HTTP
    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_SECONDS) as client:
            response = await client.get(RISK_NEWS_API_URL, params=params)
    except httpx.TimeoutException as exc:
        raise TimeoutError(
            f"Délai d'attente dépassé ({DEFAULT_TIMEOUT_SECONDS}s) lors de l'appel à l'API RiskNews ({RISK_NEWS_API_URL})."
        ) from exc
    except httpx.RequestError as exc:
        raise ConnectionError(
            f"Échec de connexion réseau vers l'API RiskNews ({RISK_NEWS_API_URL}) : {exc}"
        ) from exc

    if response.status_code != 200:
        raise RuntimeError(
            f"L'API RiskNews a retourné une erreur HTTP {response.status_code} : {response.text}"
        )

    data = response.json()
    if not isinstance(data, dict):
        raise ValueError("Format de réponse invalide retourné par l'API RiskNews (dictionnaire attendu).")

    raw_items = data.get("risk_items", [])
    risk_items: list[RiskItem] = []

    for idx, item in enumerate(raw_items):
        item_data = item if isinstance(item, dict) else vars(item)

        risk_id = str(item_data.get("risk_id", f"RISK_{idx+1:03d}"))

        # Règle stricte : Source obligatoire sur chaque item retourné
        source_name = item_data.get("source_name")
        source_url = item_data.get("source_url")
        if not source_name:
            source_name = "RiskNews Intelligence Feed"
        if not source_url:
            source_url = f"https://api.risknews.internal/alerts/{risk_id}"

        # Parsing de la date de publication
        date_published = None
        raw_date = item_data.get("date_published")
        if isinstance(raw_date, datetime):
            date_published = raw_date
        elif isinstance(raw_date, str):
            try:
                date_published = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
            except ValueError:
                date_published = None

        risk_items.append(
            RiskItem(
                risk_id=risk_id,
                category=str(item_data.get("category", "general")),
                headline=str(item_data.get("headline", "Alerte de risque")),
                summary=str(item_data.get("summary", ""))[:500],
                severity=_parse_severity(item_data.get("severity")),
                date_published=date_published,
                source_url=source_url,
                source_name=source_name,
                impact_on_value=item_data.get("impact_on_value"),
            )
        )

    # Calcul ou récupération du niveau global de risque
    if "overall_risk_level" in data:
        overall_risk = _parse_severity(data["overall_risk_level"])
    elif risk_items:
        severity_rank = {
            RiskSeverity.LOW: 1,
            RiskSeverity.MEDIUM: 2,
            RiskSeverity.HIGH: 3,
            RiskSeverity.CRITICAL: 4,
        }
        max_severity = max(risk_items, key=lambda x: severity_rank.get(x.severity, 1)).severity
        overall_risk = max_severity
    else:
        overall_risk = RiskSeverity.LOW

    player_name_resolved = str(
        data.get("player_name_resolved")
        or inp.player_name
        or f"Player #{inp.player_id}"
    )

    return SearchRiskNewsOutput(
        tool_name="search_risk_news",
        player_id=inp.player_id,
        player_name_resolved=player_name_resolved,
        risk_items=risk_items,
        overall_risk_level=overall_risk,
        lookback_days_used=inp.lookback_days,
        timestamp_utc=datetime.utcnow(),
    )
