# test_comparable_service.py
import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from sqlalchemy import create_engine
from config import DB_URL
from comparable_service import find, ComparableRequest, ComparableMode

engine = create_engine(DB_URL)

output = []

def log(text=""):
    print(text)
    output.append(text)

def display(response):
    log(f"  archetype        : {response.archetype} (conf={response.archetype_conf:.3f})")
    if response.archetype_secondary:
        log(f"  archetype_second : {response.archetype_secondary}")
    log(f"  confidence       : {response.confidence}")
    if response.warning:
        log(f"  ⚠️  warning       : {response.warning}")
    if response.role_filter_applied:
        matched = "✅" if response.role_filter_matched else "⚠️  FALLBACK"
        log(f"  role_filter      : {response.role_filter_applied} [{matched}]")
    log(f"  total_found      : {response.total_found}")
    log()
    for c in response.comparables:
        shared    = ", ".join(c.top_shared_features[:3])
        diverging = ", ".join(c.top_diverging_features[:2])
        sec = f" / {c.archetype_secondary}" if c.archetype_secondary else ""
        log(f"  {c.player_name:<28} {c.season}  age={c.age}  sim={c.similarity_pct:.1f}%  [{c.archetype}{sec}  conf={c.archetype_conf:.3f}]")
        log(f"    shared   : {shared}")
        log(f"    diverging: {diverging}")

# -----------------------------------------------------------------------
# TEST 1 — Validation Pydantic
# -----------------------------------------------------------------------
log("="*60)
log("TEST 1 — VALIDATION PYDANTIC")
log("="*60)

try:
    r = ComparableRequest(player_id=203507, season="2025/26")
except Exception as e:
    log(f"✅ Saison invalide rejetée : {e}")

try:
    r = ComparableRequest(player_id=203507, season="2025-26", k=99)
except Exception as e:
    log(f"✅ k=99 rejeté : {e}")

try:
    r = ComparableRequest(player_id=203507, season="2025-26", min_similarity_pct=150.0)
except Exception as e:
    log(f"✅ min_similarity_pct=150 rejeté : {e}")

r = ComparableRequest(player_id=203507, season="2025-26")
log(f"✅ Request minimale valide : mode={r.mode}, k={r.k}, age_window={r.age_window}")

r = ComparableRequest(
    mode              = ComparableMode.historical,
    player_id         = 203507,
    season            = "2025-26",
    k                 = 5,
    age_window        = 2.0,
    min_similarity_pct= 70.0,
    weights           = {"blk_per100_shrunk": 2.0},
    role_filter       = "franchise_player",
)
log(f"✅ Request complète valide : {r}")

# -----------------------------------------------------------------------
# TEST 2 — Joueurs (current + historical)
# -----------------------------------------------------------------------
TARGETS = [
    (203932,  "Aaron Gordon"),
    (1629130, "Duncan Robinson"),
    (203507,  "Giannis Antetokounmpo"),
    (203497,  "Rudy Gobert"),
    (1628368, "De'Aaron Fox"),
    (203110,  "Draymond Green"),
]

for player_id, name in TARGETS:
    for mode in [ComparableMode.current, ComparableMode.historical]:
        log()
        log("="*60)
        log(f"  {name} — {mode.value}")
        log("="*60)
        req  = ComparableRequest(
            mode      = mode,
            player_id = player_id,
            season    = "2025-26",
            k         = 5,
        )
        resp = find(engine, req)
        display(resp)

# -----------------------------------------------------------------------
# TEST 3 — Role filter
# -----------------------------------------------------------------------
log()
log("="*60)
log("TEST 3a — ROLE FILTER MATCH (Gobert, shot_blocker)")
log("="*60)
req  = ComparableRequest(
    player_id   = 203497,
    season      = "2025-26",
    k           = 5,
    role_filter = "shot_blocker",
)
resp = find(engine, req)
display(resp)

log()
log("="*60)
log("TEST 3b — ROLE FILTER FALLBACK (Gobert, rim_protector)")
log("="*60)
req  = ComparableRequest(
    player_id   = 203497,
    season      = "2025-26",
    k           = 5,
    role_filter = "rim_protector",
)
resp = find(engine, req)
display(resp)

# -----------------------------------------------------------------------
# TEST 4 — Weights
# -----------------------------------------------------------------------
log()
log("="*60)
log("TEST 4 — WEIGHTS (Giannis, blk x2)")
log("="*60)
req  = ComparableRequest(
    player_id = 203507,
    season    = "2025-26",
    k         = 5,
    weights   = {"blk_per100_shrunk": 2.0},
)
resp = find(engine, req)
display(resp)

# -----------------------------------------------------------------------
# TEST 5 — min_similarity_pct (Draymond historical, seuil 65%)
# -----------------------------------------------------------------------
log()
log("="*60)
log("TEST 5 — MIN SIMILARITY (Draymond historical, seuil=65%)")
log("="*60)
req  = ComparableRequest(
    mode               = ComparableMode.historical,
    player_id          = 203110,
    season             = "2025-26",
    k                  = 5,
    min_similarity_pct = 65.0,
)
resp = find(engine, req)
display(resp)

# -----------------------------------------------------------------------
# Écriture fichier
# -----------------------------------------------------------------------
with open("comparable_service_test.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output))

print("\n✅ Résultats écrits dans comparable_service_test.txt")