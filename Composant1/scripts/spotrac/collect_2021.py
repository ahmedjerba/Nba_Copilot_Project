import re, time, hashlib
from pathlib import Path
import requests
import pandas as pd
from bs4 import BeautifulSoup

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
    CACHE = PROJECT_ROOT / "ingestion/cache/cache_spotrac"
    CACHE.mkdir(parents=True, exist_ok=True)
    DATA_DIR = PROJECT_ROOT / "ingestion/data/raw"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    BASE = "https://www.spotrac.com"

    DELAI = 5.0
    HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; projet-perso-spike)"}

    SAISONS = [2021]
    METRIQUES = {"base_salary": "cap_base", "cap_hit": "cap_total", "contract_length": "contract_length", "contract_value": "contract_value"}

    # Vider le cache cap_hit pour forcer le re-téléchargement
    for f in CACHE.glob("cap_hit_*"):
        f.unlink()

    def get(nom, chemin):
        fichier = CACHE / f"{nom}_{hashlib.md5(chemin.encode()).hexdigest()[:8]}.html"
        meta = fichier.with_suffix(".status")
        if fichier.exists():
            print(f"  (cache) {nom}")
            return fichier.read_text(encoding="utf-8"), int(meta.read_text())
        time.sleep(DELAI)
        try:
            r = requests.get(BASE + chemin, headers=HEADERS, timeout=30)
        except Exception as e:
            print(f"  ERREUR réseau {nom}: {e}")
            return None, 0
        fichier.write_text(r.text, encoding="utf-8")
        meta.write_text(str(r.status_code))
        return r.text, r.status_code

    def parse(html, metrique, saison):
        soup = BeautifulSoup(html, "html.parser")
        rows = []
        for li in soup.select("li.list-group-item"):
            a = li.select_one("div.link a")
            if not a:
                continue
            m = re.search(r"/player/(\d+)", a.get("href", ""))
            pid = m.group(1) if m else None
            small = li.select_one("small")
            spans = li.select("span")
            montant_str = spans[-1].text.strip() if spans else None

            montant = None
            if montant_str:
                try:
                    montant = int(montant_str.replace("$", "").replace(",", ""))
                except ValueError:
                    pass

            team, position = None, None
            if small:
                parts = small.text.strip().split("|")
                if len(parts) >= 2:
                    team = parts[0].strip()
                    position = parts[1].strip()

            rows.append({
                "spotrac_player_id": pid,
                "player_name":       a.text.strip(),
                "team":              team,
                "position":          position,
                "metrique":          metrique,
                "saison":            saison,
                "montant":           montant,
                "montant_str":       montant_str,
            })
        return rows

    def get_all_pages(nom, metrique, saison, chemin_base):
        all_rows_local = []
        vus = set()
        page = 1
        while True:
            chemin = f"{chemin_base}/page/{page}" if page > 1 else chemin_base
            nom_page = f"{nom}_p{page}"
            html, statut = get(nom_page, chemin)
            if not html or statut != 200:
                print(f"  page {page} -> stop (statut {statut})")
                break
            rows = parse(html, metrique, saison)
            if not rows:
                print(f"  page {page} -> vide, stop")
                break
            # Détecter si Spotrac boucle sur la même page
            nouveaux = [r for r in rows if r["spotrac_player_id"] not in vus]
            if not nouveaux:
                print(f"  page {page} -> doublons détectés, stop")
                break
            for r in nouveaux:
                vus.add(r["spotrac_player_id"])
            all_rows_local.extend(nouveaux)
            print(f"  page {page} -> {len(nouveaux)} nouveaux joueurs")
            page += 1
        return all_rows_local

    # --- Extraction ---
    all_rows = []

    for saison in SAISONS:
        for metrique, sort_param in METRIQUES.items():
            nom = f"{metrique}_{saison}"
            chemin_base = f"/nba/rankings/player/_/year/{saison}/sort/{sort_param}"
            print(f"\n== {nom} ==")
            rows = get_all_pages(nom, metrique, saison, chemin_base)
            print(f"  -> {len(rows)} joueurs total")
            all_rows.extend(rows)

    # --- Résultat final ---
    df = pd.DataFrame(all_rows)
    print(f"\nTotal lignes brutes : {len(df)}")

    # Format wide (base_salary et cap_hit côte à côte)
    df_wide = df.pivot_table(
        index=["spotrac_player_id", "player_name", "team", "position", "saison"],
        columns="metrique",
        values="montant",
        aggfunc="first"
    ).reset_index()
    df_wide.columns.name = None

    print(f"Total joueurs/saisons : {len(df_wide)}")
    print(df_wide.head(10).to_string())

    df.to_csv(DATA_DIR / "spotrac_2021.csv", index=False)
    # df_wide.to_csv("", index=False)
    print("\nFichiers générés : spotrac_raw.csv + spotrac_wide.csv")