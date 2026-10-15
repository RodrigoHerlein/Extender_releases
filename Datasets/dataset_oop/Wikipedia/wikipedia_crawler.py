"""
wikipedia_crawler.py  —  Dataset: Programación Orientada a Objetos
==================================================================
Descarga artículos de Wikipedia sobre OOP, patrones de diseño, lenguajes
orientados a objetos, principios SOLID, UML y áreas relacionadas.

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
    # ── OOP central ──────────────────────────────────────────────────────
    "Object-oriented_programming":              "OOP",
    "Object-oriented_programming_languages":    "OOP/Languages",
    "Object-oriented_design":                   "OOP/Design",
    "Object-oriented_analysis_and_design":      "OOP/OOAD",

    # ── Conceptos fundamentales ───────────────────────────────────────────
    "Class_(computer_programming)":             "OOP/Concepts/Class",
    "Inheritance_(object-oriented_programming)":"OOP/Concepts/Inheritance",
    "Polymorphism_(computer_science)":          "OOP/Concepts/Polymorphism",
    "Encapsulation_(computer_programming)":     "OOP/Concepts/Encapsulation",
    "Abstraction_(computer_science)":           "OOP/Concepts/Abstraction",
    "Type_system":                              "OOP/Concepts/Types",
    "Interface_(object-oriented_programming)":  "OOP/Concepts/Interfaces",

    # ── Patrones de diseño ────────────────────────────────────────────────
    "Software_design_patterns":                 "OOP/Patterns",
    "Creational_patterns":                      "OOP/Patterns/Creational",
    "Structural_patterns":                      "OOP/Patterns/Structural",
    "Behavioral_patterns":                      "OOP/Patterns/Behavioral",
    "Architectural_pattern_(computer_science)": "OOP/Patterns/Architectural",
    "Anti-patterns":                            "OOP/AntiPatterns",

    # ── Lenguajes OOP ─────────────────────────────────────────────────────
    "Java_(programming_language)":              "OOP/Languages/Java",
    "C++":                                      "OOP/Languages/Cpp",
    "Python_(programming_language)":            "OOP/Languages/Python",
    "C_Sharp_(programming_language)":           "OOP/Languages/CSharp",
    "Ruby_(programming_language)":              "OOP/Languages/Ruby",
    "Kotlin_(programming_language)":            "OOP/Languages/Kotlin",
    "Smalltalk":                                "OOP/Languages/Smalltalk",

    # ── Principios de diseño ──────────────────────────────────────────────
    "SOLID_(object-oriented_design)":           "OOP/Principles/SOLID",
    "Design_by_contract":                       "OOP/Principles/DBC",
    "Separation_of_concerns":                   "OOP/Principles/SoC",
    "Dependency_injection":                     "OOP/Principles/DI",
    "Inversion_of_control":                     "OOP/Principles/IoC",
    "Law_of_Demeter":                           "OOP/Principles/Demeter",

    # ── Metodologías y procesos ───────────────────────────────────────────
    "Agile_software_development":               "OOP/Methodologies/Agile",
    "Extreme_programming":                      "OOP/Methodologies/XP",
    "Test-driven_development":                  "OOP/Methodologies/TDD",
    "Refactoring":                              "OOP/Refactoring",

    # ── UML y modelado ────────────────────────────────────────────────────
    "Unified_Modeling_Language":                "OOP/UML",
    "Class_diagram":                            "OOP/UML/ClassDiagram",
    "Sequence_diagram":                         "OOP/UML/SequenceDiagram",

    # ── Frameworks y arquitecturas ────────────────────────────────────────
    "Software_framework":                       "OOP/Frameworks",
    "Model-view-controller":                    "OOP/Patterns/Architectural/MVC",
    "Model-view-viewmodel":                     "OOP/Patterns/Architectural/MVVM",
    "Spring_Framework":                         "OOP/Frameworks/Spring",
    "Django_(web_framework)":                   "OOP/Frameworks/Django",

    # ── Programación funcional (comparativa) ─────────────────────────────
    "Functional_programming":                   "OOP/Related/Functional",
    "Generic_programming":                      "OOP/Related/Generic",
    "Aspect-oriented_programming":              "OOP/Related/AOP",
    "Prototype-based_programming":              "OOP/Related/Prototype",

    # ── Software engineering general ─────────────────────────────────────
    "Software_engineering":                     "OOP/SoftEng",
    "Software_architecture":                    "OOP/SoftEng/Architecture",
    "Code_refactoring":                         "OOP/SoftEng/Refactoring",
    "Unit_testing":                             "OOP/SoftEng/Testing",
    "Version_control":                          "OOP/SoftEng/VersionControl",
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
        user_agent="EXTENDER-oop-dataset/1.0 (research project; contact: rodrigonherlein@gmail.com)",
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
        log_fh.write("=== Dataset: Programación Orientada a Objetos ===\n")

        for root_cat, base_path in ROOT_CATEGORIES.items():
            if total_new >= MAX_TOTAL: break
            remaining = min(MAX_PER_CATEGORY, MAX_TOTAL - total_new)
            print(f"\n[{ts()}] ══ Root: {root_cat} → {base_path} (max {remaining})")
            log_fh.write(f"\n── Root: {root_cat} ──\n"); log_fh.flush()
            total_new += crawl_root(wiki, root_cat, base_path, seen_ids, out_fh, log_fh, remaining)

    with open(seen_path, "w", encoding="utf-8") as f:
        f.write("\n".join(seen_ids))

    print(f"\n[{ts()}] ══════════════════════════════════════════")
    print(f"[{ts()}] CRAWL COMPLETADO — Dataset: Programación Orientada a Objetos")
    print(f"[{ts()}]   Artículos nuevos esta sesión: {total_new:,}")
    print(f"[{ts()}]   Total acumulado:              {len(seen_ids):,}")
    print(f"[{ts()}]   Guardado en:                  {articles_path}")
    print(f"[{ts()}] ══════════════════════════════════════════")


if __name__ == "__main__":
    main()
