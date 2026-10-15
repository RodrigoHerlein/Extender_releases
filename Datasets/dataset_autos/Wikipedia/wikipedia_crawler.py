"""
wikipedia_crawler.py  —  Dataset: Autos / Automóviles
======================================================
Descarga artículos de Wikipedia sobre automóviles, ingeniería automotriz,
marcas y modelos, tecnología vehicular, Fórmula 1, autos eléctricos
y temas relacionados con la industria automotriz.

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
    # ── Automóviles general ───────────────────────────────────────────────
    "Automobiles":                              "Cars",
    "Automotive_industry":                      "Cars/Industry",
    "Car":                                      "Cars/General",
    "Automotive_engineering":                   "Cars/Engineering",
    "Vehicle_technology":                       "Cars/Technology",

    # ── Marcas y fabricantes ──────────────────────────────────────────────
    "Automobile_manufacturers":                 "Cars/Manufacturers",
    "German_automobile_manufacturers":          "Cars/Manufacturers/Germany",
    "Italian_automobile_manufacturers":         "Cars/Manufacturers/Italy",
    "Japanese_automobile_manufacturers":        "Cars/Manufacturers/Japan",
    "American_automobile_manufacturers":        "Cars/Manufacturers/USA",
    "Mercedes-Benz":                            "Cars/Manufacturers/Mercedes",
    "BMW":                                      "Cars/Manufacturers/BMW",
    "Volkswagen":                               "Cars/Manufacturers/VW",
    "Toyota":                                   "Cars/Manufacturers/Toyota",
    "Ford_Motor_Company":                       "Cars/Manufacturers/Ford",
    "Ferrari":                                  "Cars/Manufacturers/Ferrari",
    "Lamborghini":                              "Cars/Manufacturers/Lamborghini",
    "Porsche":                                  "Cars/Manufacturers/Porsche",
    "Tesla,_Inc.":                              "Cars/Manufacturers/Tesla",

    # ── Tipos de vehículos ────────────────────────────────────────────────
    "Electric_vehicles":                        "Cars/Types/Electric",
    "Hybrid_vehicles":                          "Cars/Types/Hybrid",
    "Sport_utility_vehicles":                   "Cars/Types/SUV",
    "Sedans":                                   "Cars/Types/Sedan",
    "Sports_cars":                              "Cars/Types/Sports",
    "Concept_cars":                             "Cars/Types/Concept",
    "Pickup_trucks":                            "Cars/Types/Pickup",
    "Luxury_vehicles":                          "Cars/Types/Luxury",
    "Autonomous_vehicles":                      "Cars/Types/Autonomous",

    # ── Ingeniería y componentes ──────────────────────────────────────────
    "Internal_combustion_engines":              "Cars/Engineering/Engine/ICE",
    "Electric_motors":                          "Cars/Engineering/Engine/Electric",
    "Transmission_(mechanics)":                 "Cars/Engineering/Transmission",
    "Suspension_(vehicle)":                     "Cars/Engineering/Suspension",
    "Braking_system":                           "Cars/Engineering/Brakes",
    "Automotive_safety":                        "Cars/Engineering/Safety",
    "Aerodynamics":                             "Cars/Engineering/Aerodynamics",
    "Fuel_injection":                           "Cars/Engineering/FuelInjection",
    "Turbocharger":                             "Cars/Engineering/Turbo",
    "Battery_electric_vehicle":                 "Cars/Engineering/BatteryEV",
    "Charging_station":                         "Cars/Technology/Charging",

    # ── Motores de combustión ─────────────────────────────────────────────
    "Four-stroke_engine":                       "Cars/Engineering/Engine/FourStroke",
    "Diesel_engine":                            "Cars/Engineering/Engine/Diesel",
    "V8_engine":                                "Cars/Engineering/Engine/V8",
    "Flat-six_engine":                          "Cars/Engineering/Engine/Flat6",

    # ── Competición y motorsport ─────────────────────────────────────────
    "Formula_One":                              "Cars/Racing/F1",
    "Motorsport":                               "Cars/Racing",
    "Le_Mans_24_Hours":                         "Cars/Racing/LeMans",
    "NASCAR":                                   "Cars/Racing/NASCAR",
    "Rally_racing":                             "Cars/Racing/Rally",
    "IndyCar_Series":                           "Cars/Racing/IndyCar",
    "World_Rally_Championship":                 "Cars/Racing/WRC",

    # ── Tecnología moderna ────────────────────────────────────────────────
    "Self-driving_car":                         "Cars/Tech/SelfDriving",
    "Advanced_driver-assistance_systems":       "Cars/Tech/ADAS",
    "Vehicle_to_grid":                          "Cars/Tech/V2G",
    "Hydrogen_vehicle":                         "Cars/Tech/Hydrogen",
    "Connected_car":                            "Cars/Tech/Connected",
    "Vehicle_dynamics":                         "Cars/Tech/Dynamics",

    # ── Historia e industria ──────────────────────────────────────────────
    "History_of_the_automobile":                "Cars/History",
    "Automotive_design":                        "Cars/Design",
    "Concept_cars":                             "Cars/Design/Concept",
    "Car_culture":                              "Cars/Culture",
    "Classic_cars":                             "Cars/History/Classic",
    "Automobile_safety":                        "Cars/Safety",
}

MAX_DEPTH        = 2
MAX_PER_CATEGORY = 100
MAX_TOTAL        = 6_000
DELAY_SECONDS    = 0.5
SUMMARY_CHARS    = 1_200


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def build_wiki() -> wikipediaapi.Wikipedia:
    return wikipediaapi.Wikipedia(
        language="en",
        user_agent="EXTENDER-autos-dataset/1.0 (research project; contact: rodrigonherlein@gmail.com)",
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
        log_fh.write("=== Dataset: Autos / Automóviles ===\n")

        for root_cat, base_path in ROOT_CATEGORIES.items():
            if total_new >= MAX_TOTAL: break
            remaining = min(MAX_PER_CATEGORY, MAX_TOTAL - total_new)
            print(f"\n[{ts()}] ══ Root: {root_cat} → {base_path} (max {remaining})")
            log_fh.write(f"\n── Root: {root_cat} ──\n"); log_fh.flush()
            total_new += crawl_root(wiki, root_cat, base_path, seen_ids, out_fh, log_fh, remaining)

    with open(seen_path, "w", encoding="utf-8") as f:
        f.write("\n".join(seen_ids))

    print(f"\n[{ts()}] ══════════════════════════════════════════")
    print(f"[{ts()}] CRAWL COMPLETADO — Dataset: Autos")
    print(f"[{ts()}]   Artículos nuevos esta sesión: {total_new:,}")
    print(f"[{ts()}]   Total acumulado:              {len(seen_ids):,}")
    print(f"[{ts()}]   Guardado en:                  {articles_path}")
    print(f"[{ts()}] ══════════════════════════════════════════")


if __name__ == "__main__":
    main()
