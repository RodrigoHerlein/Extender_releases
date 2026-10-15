"""
wikipedia_crawler.py  —  Dataset: Visualización de Datos
=========================================================
Descarga artículos de Wikipedia organizados por categorías del área de
Visualización de Datos / Information Visualization y los guarda en
articles.jsonl, listo para usar con WikipediaCorpus (el mismo que el
dataset de computación).

Uso:
    pip install wikipedia-api
    python wikipedia_crawler.py

Retomable: si se interrumpe, volver a correr desde el mismo directorio
continúa desde donde quedó (lee seen_ids.txt).

Salida:
    articles.jsonl   — un artículo JSON por línea
    crawl_log.txt    — log de progreso
    seen_ids.txt     — IDs ya descargados (para reanudar)
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import deque
from datetime import datetime

import wikipediaapi  # pip install wikipedia-api

# ─────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────

OUT_DIR = os.path.dirname(os.path.abspath(__file__))  # misma carpeta que este script

# Categorías raíz y su path en la taxonomía.
# El path se usa como campo "category" en cada artículo exportado.
ROOT_CATEGORIES: dict[str, str] = {
    # ── Núcleo de visualización ──────────────────────────────────────────
    "Data_visualization":              "DataViz",
    "Information_visualization":       "DataViz/InfoViz",
    "Scientific_visualization":        "DataViz/Scientific",

    # ── Gráficos y diagramas estadísticos ──────────────────────────────
    "Statistical_graphics":            "DataViz/Charts/Statistical",
    "Statistical_charts_and_diagrams": "DataViz/Charts/Statistical",
    "Diagram":                         "DataViz/Charts/Diagrams",
    "Infographics":                    "DataViz/Infographics",

    # ── Mapas y GIS ─────────────────────────────────────────────────────
    "Cartography":                     "DataViz/Maps/Cartography",
    "Geographic_information_systems":  "DataViz/Maps/GIS",
    "Thematic_maps":                   "DataViz/Maps/Thematic",

    # ── Diseño e interacción ─────────────────────────────────────────────
    "Information_design":              "DataViz/Design",
    "Human-computer_interaction":      "DataViz/HCI",
    "Visual_analytics":                "DataViz/Analytics",
    "User_interface":                  "DataViz/UI",

    # ── Herramientas y software ──────────────────────────────────────────
    "Data_visualization_software":     "DataViz/Tools",
    "Plotting_software":               "DataViz/Tools/Plotting",
    "Business_intelligence_software":  "DataViz/Tools/BI",

    # ── Estadística y análisis de datos ─────────────────────────────────
    "Exploratory_data_analysis":       "DataViz/DataScience/EDA",
    "Multivariate_statistics":         "DataViz/DataScience/Multivariate",
    "Data_science":                    "DataViz/DataScience",

    # ── Percepción visual y diseño gráfico ───────────────────────────────
    "Graphical_perception":            "DataViz/Perception",
    "Color_theory":                    "DataViz/Perception/Color",
    "Computer_graphics":               "DataViz/ComputerGraphics",
    "3D_computer_graphics":            "DataViz/ComputerGraphics/3D",
    "Graph_drawing":                   "DataViz/Graphs/Drawing",
    "Network_visualization":           "DataViz/Graphs/Networks",
}

MAX_DEPTH        = 3       # profundidad máxima de subcategorías
MAX_PER_CATEGORY = 150     # máximo artículos por categoría raíz
MAX_TOTAL        = 5_000   # tope global de artículos
DELAY_SECONDS    = 0.5     # pausa entre requests (respetar rate limit de Wikipedia)
SUMMARY_CHARS    = 1_200   # cuántos chars del intro usar como descripción

# ─────────────────────────────────────────────────────────


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def build_wiki() -> wikipediaapi.Wikipedia:
    return wikipediaapi.Wikipedia(
        language="en",
        user_agent="EXTENDER-dataviz-dataset/1.0 (research project; contact: rodrigonherlein@gmail.com)",
        extract_format=wikipediaapi.ExtractFormat.WIKI,
    )


def clean_summary(text: str, max_chars: int) -> str:
    """Devuelve el primer párrafo real truncado."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    summary = " ".join(lines)
    return summary[:max_chars].strip()


def crawl_root(
    wiki: wikipediaapi.Wikipedia,
    root_cat: str,
    base_path: str,
    seen_ids: set[str],
    out_fh,
    log_fh,
    max_articles: int,
) -> int:
    """
    BFS sobre categorías a partir de root_cat.
    Devuelve cantidad de artículos nuevos escritos.
    """
    queue: deque[tuple[str, str, int]] = deque()  # (cat_name, cat_path, depth)
    queue.append((root_cat, base_path, 0))
    visited_cats: set[str] = set()
    articles_added = 0

    def log(msg: str):
        line = f"[{ts()}] {msg}"
        print(line)
        log_fh.write(line + "\n")
        log_fh.flush()

    while queue and articles_added < max_articles:
        cat_name, cat_path, depth = queue.popleft()

        if cat_name in visited_cats:
            continue
        visited_cats.add(cat_name)

        log(f"  📁 {cat_path}  (depth={depth})")

        try:
            cat_page = wiki.page(f"Category:{cat_name}")
            time.sleep(DELAY_SECONDS)
        except Exception as e:
            log(f"     ⚠ Error al obtener categoría {cat_name}: {e}")
            continue

        if not cat_page.exists():
            log(f"     ⚠ Categoría no existe: {cat_name} — saltando")
            continue

        for title, member in cat_page.categorymembers.items():
            if articles_added >= max_articles:
                break

            # Subcategoría → encolar si no excede profundidad
            if member.ns == wikipediaapi.Namespace.CATEGORY:
                if depth < MAX_DEPTH:
                    sub_name = title.replace("Category:", "")
                    sub_path = (
                        cat_path + "/" +
                        sub_name.replace("_", " ").replace("Category:", "").strip()
                    ).replace("//", "/")
                    if sub_name not in visited_cats:
                        queue.append((sub_name, sub_path, depth + 1))
                continue

            # Artículo normal (namespace 0)
            if member.ns != wikipediaapi.Namespace.MAIN:
                continue

            page_id = str(member.pageid) if member.pageid else title
            if page_id in seen_ids:
                continue

            try:
                page = wiki.page(title)
                time.sleep(DELAY_SECONDS)
            except Exception as e:
                log(f"     ⚠ Error al obtener artículo '{title}': {e}")
                continue

            if not page.exists() or not page.summary:
                continue

            summary = clean_summary(page.summary, SUMMARY_CHARS)
            if len(summary) < 80:
                continue  # página de desambiguación o stub

            record = {
                "id":          page_id,
                "url":         page.fullurl,
                "title":       page.title,
                "description": summary,
                "category":    cat_path,
                "root_cat":    root_cat,
            }

            out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_fh.flush()
            seen_ids.add(page_id)
            articles_added += 1

            if articles_added % 50 == 0:
                log(f"     ✅ {articles_added} artículos en '{base_path}'")

    log(f"  ✅ '{base_path}' completado: {articles_added} artículos nuevos")
    return articles_added


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    articles_path = os.path.join(OUT_DIR, "articles.jsonl")
    seen_path     = os.path.join(OUT_DIR, "seen_ids.txt")
    log_path      = os.path.join(OUT_DIR, "crawl_log.txt")

    # Cargar IDs ya descargados (permite retomar si se interrumpe)
    seen_ids: set[str] = set()
    if os.path.exists(seen_path):
        with open(seen_path, encoding="utf-8") as f:
            seen_ids = set(l.strip() for l in f if l.strip())
        print(f"[{ts()}] Retomando: {len(seen_ids):,} artículos ya descargados.")

    wiki = build_wiki()
    total_new = 0

    with (
        open(articles_path, "a", encoding="utf-8") as out_fh,
        open(log_path, "a", encoding="utf-8") as log_fh,
    ):
        log_fh.write(f"\n\n=== CRAWL INICIADO {datetime.now().isoformat()} ===\n")
        log_fh.write(f"=== Dataset: Visualización de Datos ===\n")

        for root_cat, base_path in ROOT_CATEGORIES.items():
            if total_new >= MAX_TOTAL:
                break

            remaining = min(MAX_PER_CATEGORY, MAX_TOTAL - total_new)
            print(f"\n[{ts()}] ══ Root: {root_cat} → {base_path} (max {remaining})")
            log_fh.write(f"\n── Root: {root_cat} ──\n")
            log_fh.flush()

            added = crawl_root(
                wiki=wiki,
                root_cat=root_cat,
                base_path=base_path,
                seen_ids=seen_ids,
                out_fh=out_fh,
                log_fh=log_fh,
                max_articles=remaining,
            )
            total_new += added

    # Guardar seen_ids actualizado (para futuras reanudaciones)
    with open(seen_path, "w", encoding="utf-8") as f:
        f.write("\n".join(seen_ids))

    print(f"\n[{ts()}] ══════════════════════════════════════════")
    print(f"[{ts()}] CRAWL COMPLETADO — Dataset: Visualización de Datos")
    print(f"[{ts()}]   Artículos nuevos esta sesión: {total_new:,}")
    print(f"[{ts()}]   Total acumulado:              {len(seen_ids):,}")
    print(f"[{ts()}]   Guardado en:                  {articles_path}")
    print(f"[{ts()}] ══════════════════════════════════════════")


if __name__ == "__main__":
    main()
