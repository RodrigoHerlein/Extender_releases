"""
wikipedia_crawler.py  —  Dataset: Vinos
========================================
Descarga artículos de Wikipedia sobre enología, viticultura, regiones
vitivinícolas, variedades de uva, maridaje, cata de vinos y temas afines.

Uso:
    pip install wikipedia-api
    python wikipedia_crawler.py

Retomable: volver a correr continúa desde donde quedó (seen_ids.txt).

Salida:
    articles.jsonl   — un artículo JSON por línea
    crawl_log.txt    — log de progreso
    seen_ids.txt     — IDs ya descargados
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from datetime import datetime

import wikipediaapi

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

ROOT_CATEGORIES: dict[str, str] = {
    # ── Vino general ──────────────────────────────────────────────────────
    "Wine":                                     "Wine",
    "Winemaking":                               "Wine/Winemaking",
    "Viticulture":                              "Wine/Viticulture",
    "Enology":                                  "Wine/Enology",

    # ── Variedades de uva ─────────────────────────────────────────────────
    "Grape_varieties":                          "Wine/Grapes",
    "Red_wine_grape_varieties":                 "Wine/Grapes/Red",
    "White_wine_grape_varieties":               "Wine/Grapes/White",
    "Cabernet_Sauvignon":                       "Wine/Grapes/CabSauv",
    "Merlot":                                   "Wine/Grapes/Merlot",
    "Chardonnay":                               "Wine/Grapes/Chardonnay",
    "Pinot_noir":                               "Wine/Grapes/PinotNoir",
    "Sauvignon_blanc":                          "Wine/Grapes/SauvBlanc",
    "Riesling":                                 "Wine/Grapes/Riesling",
    "Syrah":                                    "Wine/Grapes/Syrah",
    "Malbec":                                   "Wine/Grapes/Malbec",
    "Tempranillo":                              "Wine/Grapes/Tempranillo",
    "Nebbiolo":                                 "Wine/Grapes/Nebbiolo",
    "Sangiovese":                               "Wine/Grapes/Sangiovese",

    # ── Regiones vitivinícolas por continente/país ───────────────────────
    "Wine_regions":                             "Wine/Regions",
    "French_wine":                              "Wine/Regions/France",
    "Italian_wine":                             "Wine/Regions/Italy",
    "Spanish_wine":                             "Wine/Regions/Spain",
    "Argentine_wine":                           "Wine/Regions/Argentina",
    "Chilean_wine":                             "Wine/Regions/Chile",
    "American_wine":                            "Wine/Regions/USA",
    "Australian_wine":                          "Wine/Regions/Australia",
    "German_wine":                              "Wine/Regions/Germany",
    "Portuguese_wine":                          "Wine/Regions/Portugal",
    "South_African_wine":                       "Wine/Regions/SouthAfrica",
    "New_Zealand_wine":                         "Wine/Regions/NewZealand",
    "Wine_regions_of_France":                   "Wine/Regions/France/Regions",
    "Wine_regions_of_Italy":                    "Wine/Regions/Italy/Regions",

    # ── Tipos y estilos de vino ───────────────────────────────────────────
    "Red_wine":                                 "Wine/Types/Red",
    "White_wine":                               "Wine/Types/White",
    "Rosé_wine":                                "Wine/Types/Rose",
    "Sparkling_wine":                           "Wine/Types/Sparkling",
    "Champagne_(wine_region)":                  "Wine/Types/Sparkling/Champagne",
    "Fortified_wine":                           "Wine/Types/Fortified",
    "Dessert_wine":                             "Wine/Types/Dessert",
    "Natural_wine":                             "Wine/Types/Natural",
    "Organic_wine":                             "Wine/Types/Organic",

    # ── Cata y maridaje ───────────────────────────────────────────────────
    "Wine_tasting":                             "Wine/Tasting",
    "Wine_and_food_matching":                   "Wine/Pairing",
    "Wine_sommelier":                           "Wine/Tasting/Sommelier",
    "Wine_label":                               "Wine/Tasting/Labels",
    "Wine_rating":                              "Wine/Tasting/Ratings",

    # ── Producción y elaboración ──────────────────────────────────────────
    "Fermentation_in_winemaking":               "Wine/Winemaking/Fermentation",
    "Aging_of_wine":                            "Wine/Winemaking/Aging",
    "Oak_(wine)":                               "Wine/Winemaking/Oak",
    "Wine_cellar":                              "Wine/Winemaking/Cellar",
    "Harvesting_(wine)":                        "Wine/Winemaking/Harvest",
    "Terroir":                                  "Wine/Viticulture/Terroir",

    # ── Appellation systems ───────────────────────────────────────────────
    "Appellation":                              "Wine/Appellations",
    "Denominación_de_origen":                   "Wine/Appellations/Spain",
    "Appellation_d'origine_contrôlée":          "Wine/Appellations/France",

    # ── Cultura y economía ────────────────────────────────────────────────
    "Wine_culture":                             "Wine/Culture",
    "Wine_economics":                           "Wine/Economics",
    "Wine_tourism":                             "Wine/Culture/Tourism",
    "Wine_law":                                 "Wine/Law",
    "Wine_list":                                "Wine/Culture/Lists",
}

MAX_DEPTH        = 2
MAX_PER_CATEGORY = 100
MAX_TOTAL        = 5_000
DELAY_SECONDS    = 0.5
SUMMARY_CHARS    = 1_200


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def build_wiki() -> wikipediaapi.Wikipedia:
    return wikipediaapi.Wikipedia(
        language="en",
        user_agent="EXTENDER-vinos-dataset/1.0 (research project; contact: rodrigonherlein@gmail.com)",
        extract_format=wikipediaapi.ExtractFormat.WIKI,
    )


def clean_summary(text: str, max_chars: int) -> str:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return " ".join(lines)[:max_chars].strip()


def crawl_root(wiki, root_cat, base_path, seen_ids, out_fh, log_fh, max_articles) -> int:
    queue: deque[tuple[str, str, int]] = deque()
    queue.append((root_cat, base_path, 0))
    visited_cats: set[str] = set()
    articles_added = 0

    def log(msg):
        line = f"[{ts()}] {msg}"
        print(line); log_fh.write(line + "\n"); log_fh.flush()

    while queue and articles_added < max_articles:
        cat_name, cat_path, depth = queue.popleft()
        if cat_name in visited_cats: continue
        visited_cats.add(cat_name)
        log(f"  📁 {cat_path}  (depth={depth})")

        try:
            cat_page = wiki.page(f"Category:{cat_name}")
            time.sleep(DELAY_SECONDS)
        except Exception as e:
            log(f"     ⚠ Error categoría {cat_name}: {e}"); continue

        if not cat_page.exists():
            log(f"     ⚠ No existe: {cat_name}"); continue

        for title, member in cat_page.categorymembers.items():
            if articles_added >= max_articles: break

            if member.ns == wikipediaapi.Namespace.CATEGORY:
                if depth < MAX_DEPTH:
                    sub = title.replace("Category:", "")
                    if sub not in visited_cats:
                        queue.append((sub, cat_path + "/" + sub.replace("_"," "), depth + 1))
                continue
            if member.ns != wikipediaapi.Namespace.MAIN: continue

            page_id = str(member.pageid) if member.pageid else title
            if page_id in seen_ids: continue

            try:
                page = wiki.page(title); time.sleep(DELAY_SECONDS)
            except Exception as e:
                log(f"     ⚠ Error '{title}': {e}"); continue

            if not page.exists() or not page.summary: continue
            summary = clean_summary(page.summary, SUMMARY_CHARS)
            if len(summary) < 80: continue

            out_fh.write(json.dumps({"id": page_id, "url": page.fullurl, "title": page.title,
                "description": summary, "category": cat_path, "root_cat": root_cat},
                ensure_ascii=False) + "\n")
            out_fh.flush()
            seen_ids.add(page_id)
            articles_added += 1
            if articles_added % 50 == 0: log(f"     ✅ {articles_added} artículos en '{base_path}'")

    log(f"  ✅ '{base_path}' completado: {articles_added} artículos nuevos")
    return articles_added


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    articles_path = os.path.join(OUT_DIR, "articles.jsonl")
    seen_path     = os.path.join(OUT_DIR, "seen_ids.txt")
    log_path      = os.path.join(OUT_DIR, "crawl_log.txt")

    seen_ids: set[str] = set()
    if os.path.exists(seen_path):
        with open(seen_path, encoding="utf-8") as f:
            seen_ids = set(l.strip() for l in f if l.strip())
        print(f"[{ts()}] Retomando: {len(seen_ids):,} artículos ya descargados.")

    wiki = build_wiki()
    total_new = 0

    with open(articles_path, "a", encoding="utf-8") as out_fh, open(log_path, "a", encoding="utf-8") as log_fh:
        log_fh.write(f"\n\n=== CRAWL INICIADO {datetime.now().isoformat()} ===\n")
        log_fh.write("=== Dataset: Vinos ===\n")

        for root_cat, base_path in ROOT_CATEGORIES.items():
            if total_new >= MAX_TOTAL: break
            remaining = min(MAX_PER_CATEGORY, MAX_TOTAL - total_new)
            print(f"\n[{ts()}] ══ Root: {root_cat} → {base_path} (max {remaining})")
            log_fh.write(f"\n── Root: {root_cat} ──\n"); log_fh.flush()
            total_new += crawl_root(wiki, root_cat, base_path, seen_ids, out_fh, log_fh, remaining)

    with open(seen_path, "w", encoding="utf-8") as f:
        f.write("\n".join(seen_ids))

    print(f"\n[{ts()}] ══════════════════════════════════════════")
    print(f"[{ts()}] CRAWL COMPLETADO — Dataset: Vinos")
    print(f"[{ts()}]   Artículos nuevos esta sesión: {total_new:,}")
    print(f"[{ts()}]   Total acumulado:              {len(seen_ids):,}")
    print(f"[{ts()}]   Guardado en:                  {articles_path}")
    print(f"[{ts()}] ══════════════════════════════════════════")


if __name__ == "__main__":
    main()
