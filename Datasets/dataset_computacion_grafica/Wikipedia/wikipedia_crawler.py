"""
wikipedia_crawler.py  —  Dataset: Computación Gráfica
======================================================
Descarga artículos de Wikipedia sobre Computer Graphics, renderizado,
animación 3D, gráficos por computadora y áreas relacionadas.

Uso:
    pip install wikipedia-api
    python wikipedia_crawler.py

Retomable: si se interrumpe, volver a correr continúa desde donde quedó.

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

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

ROOT_CATEGORIES: dict[str, str] = {
    # ── Núcleo de computación gráfica ───────────────────────────────────
    "Computer_graphics":                        "CG",
    "3D_computer_graphics":                     "CG/3D",
    "Computer_graphics_(images)":               "CG/Images",
    "Computer_graphics_algorithms":             "CG/Algorithms",

    # ── Renderizado ──────────────────────────────────────────────────────
    "Rendering_(computer_graphics)":            "CG/Rendering",
    "Ray_tracing_(graphics)":                   "CG/Rendering/RayTracing",
    "Global_illumination":                      "CG/Rendering/GlobalIllum",
    "Shading":                                  "CG/Rendering/Shading",
    "Texture_mapping":                          "CG/Rendering/Textures",
    "Real-time_rendering":                      "CG/Rendering/RealTime",

    # ── Modelado 3D ──────────────────────────────────────────────────────
    "3D_modeling":                              "CG/Modeling/3D",
    "Solid_modeling":                           "CG/Modeling/Solid",
    "Mesh_generation":                          "CG/Modeling/Mesh",
    "Geometric_modeling":                       "CG/Modeling/Geometric",
    "Procedural_generation":                    "CG/Modeling/Procedural",
    "Subdivision_surfaces":                     "CG/Modeling/Subdivision",

    # ── Animación ────────────────────────────────────────────────────────
    "Computer_animation":                       "CG/Animation",
    "Computer-generated_imagery":               "CG/Animation/CGI",
    "Motion_capture":                           "CG/Animation/MotionCapture",
    "Skeletal_animation":                       "CG/Animation/Skeletal",
    "Character_animation":                      "CG/Animation/Character",
    "Physics_simulation":                       "CG/Animation/Physics",

    # ── Imágenes y procesamiento ──────────────────────────────────────────
    "Image_processing":                         "CG/ImageProcessing",
    "Digital_image":                            "CG/ImageProcessing/Digital",
    "Computer_vision":                          "CG/Vision",
    "Image_segmentation":                       "CG/Vision/Segmentation",

    # ── GPU y hardware ────────────────────────────────────────────────────
    "Graphics_processing_unit":                 "CG/Hardware/GPU",
    "Graphics_hardware":                        "CG/Hardware",
    "OpenGL":                                   "CG/APIs/OpenGL",
    "DirectX":                                  "CG/APIs/DirectX",
    "Vulkan_(API)":                             "CG/APIs/Vulkan",
    "Shader":                                   "CG/APIs/Shaders",

    # ── Gráficos 2D y vectoriales ─────────────────────────────────────────
    "2D_computer_graphics":                     "CG/2D",
    "Vector_graphics":                          "CG/2D/Vector",
    "Raster_graphics":                          "CG/2D/Raster",
    "Digital_art":                              "CG/DigitalArt",

    # ── Efectos visuales y posproducción ────────────────────────────────
    "Visual_effects":                           "CG/VFX",
    "Digital_compositing":                      "CG/VFX/Compositing",
    "Particle_system":                          "CG/VFX/Particles",

    # ── Realidad virtual / aumentada ─────────────────────────────────────
    "Virtual_reality":                          "CG/VR",
    "Augmented_reality":                        "CG/AR",
    "Mixed_reality":                            "CG/MR",

    # ── Geometría computacional ───────────────────────────────────────────
    "Computational_geometry":                   "CG/Geometry",
    "Polygon_mesh":                             "CG/Geometry/PolygonMesh",
    "Bezier_curve":                             "CG/Geometry/Curves",
    "NURBS":                                    "CG/Geometry/NURBS",

    # ── Software y motores ────────────────────────────────────────────────
    "3D_computer_graphics_software":            "CG/Software",
    "Game_engines":                             "CG/Software/GameEngines",
    "Video_game_graphics":                      "CG/Games",
}

MAX_DEPTH        = 2
MAX_PER_CATEGORY = 120
MAX_TOTAL        = 6_000
DELAY_SECONDS    = 0.5
SUMMARY_CHARS    = 1_200

# ─────────────────────────────────────────────────────────


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def build_wiki() -> wikipediaapi.Wikipedia:
    return wikipediaapi.Wikipedia(
        language="en",
        user_agent="EXTENDER-computacion-grafica/1.0 (research project; contact: rodrigonherlein@gmail.com)",
        extract_format=wikipediaapi.ExtractFormat.WIKI,
    )


def clean_summary(text: str, max_chars: int) -> str:
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
    queue: deque[tuple[str, str, int]] = deque()
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
            log(f"     ⚠ Categoría no existe: {cat_name}")
            continue

        for title, member in cat_page.categorymembers.items():
            if articles_added >= max_articles:
                break

            if member.ns == wikipediaapi.Namespace.CATEGORY:
                if depth < MAX_DEPTH:
                    sub_name = title.replace("Category:", "")
                    sub_path = (cat_path + "/" + sub_name.replace("_", " ").replace("Category:", "").strip()).replace("//", "/")
                    if sub_name not in visited_cats:
                        queue.append((sub_name, sub_path, depth + 1))
                continue

            if member.ns != wikipediaapi.Namespace.MAIN:
                continue

            page_id = str(member.pageid) if member.pageid else title
            if page_id in seen_ids:
                continue

            try:
                page = wiki.page(title)
                time.sleep(DELAY_SECONDS)
            except Exception as e:
                log(f"     ⚠ Error '{title}': {e}")
                continue

            if not page.exists() or not page.summary:
                continue

            summary = clean_summary(page.summary, SUMMARY_CHARS)
            if len(summary) < 80:
                continue

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
        log_fh.write("=== Dataset: Computación Gráfica ===\n")

        for root_cat, base_path in ROOT_CATEGORIES.items():
            if total_new >= MAX_TOTAL:
                break
            remaining = min(MAX_PER_CATEGORY, MAX_TOTAL - total_new)
            print(f"\n[{ts()}] ══ Root: {root_cat} → {base_path} (max {remaining})")
            log_fh.write(f"\n── Root: {root_cat} ──\n")
            log_fh.flush()
            added = crawl_root(wiki, root_cat, base_path, seen_ids, out_fh, log_fh, remaining)
            total_new += added

    with open(seen_path, "w", encoding="utf-8") as f:
        f.write("\n".join(seen_ids))

    print(f"\n[{ts()}] ══════════════════════════════════════════")
    print(f"[{ts()}] CRAWL COMPLETADO — Dataset: Computación Gráfica")
    print(f"[{ts()}]   Artículos nuevos esta sesión: {total_new:,}")
    print(f"[{ts()}]   Total acumulado:              {len(seen_ids):,}")
    print(f"[{ts()}]   Guardado en:                  {articles_path}")
    print(f"[{ts()}] ══════════════════════════════════════════")


if __name__ == "__main__":
    main()
