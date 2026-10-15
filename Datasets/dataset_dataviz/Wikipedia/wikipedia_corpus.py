"""
wikipedia_corpus.py
===================
Drop-in replacement de CurlieCorpus para artículos de Wikipedia.

Espera que wikipedia_crawler.py haya generado articles.jsonl en el mismo
directorio (o en data_dir).

Uso en EXTENDER (reemplaza CurlieCorpus):

    from dataset_paper.Wikipedia.wikipedia_corpus import WikipediaCorpus

    corpus = WikipediaCorpus(
        data_dir="dataset_paper/Wikipedia",
        include_keywords=("Computers",),   # None = todo el dataset
    )

    # Después, en TopicExplorer, pasar corpus en lugar de CurlieCorpus:
    explorer = TopicExplorer(corpus=corpus, ...)
"""

from __future__ import annotations

import json
import os
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────
# WikipediaEntry  (misma interfaz que CurlieEntry)
# ─────────────────────────────────────────────────────────

@dataclass
class WikipediaEntry:
    url:         str
    title:       str
    description: str
    category:    str = ""
    source_tag:  str = "wikipedia"

    def text(self) -> str:
        """Texto combinado para embedding — mirrors CurlieEntry.text()."""
        cat_words = self.category.replace("/", " ").replace("_", " ")
        return f"{self.title}. {self.description} {cat_words}".strip()

    def to_dict(self) -> dict:
        return {
            "url":        self.url,
            "title":      self.title,
            "snippet":    self.description[:300],
            "category":   self.category,
            "source_tag": self.source_tag,
        }


# ─────────────────────────────────────────────────────────
# WikipediaCorpus
# ─────────────────────────────────────────────────────────

class WikipediaCorpus:
    """
    Corpus de Wikipedia compatible con la interfaz de CurlieCorpus.

    Parámetros
    ----------
    data_dir : str
        Carpeta que contiene articles.jsonl (salida de wikipedia_crawler.py).
    include_keywords : tuple[str] | None
        Si se provee, solo se incluyen artículos cuya categoría contenga
        alguna de las palabras clave.  None = incluir todo.
        Ejemplo: ("Computers/AI", "Computers/Programming")
    max_total_entries : int | None
        Límite opcional de entradas a cargar.
    verbose : bool
    """

    def __init__(
        self,
        data_dir:           str,
        include_keywords:   Optional[tuple[str, ...]] = None,
        max_total_entries:  Optional[int] = None,
        verbose:            bool = True,
    ):
        self.data_dir          = data_dir
        self.include_keywords  = include_keywords
        self.max_total_entries = max_total_entries
        self.verbose           = verbose

        self.entries: list[WikipediaEntry] = []
        self._load_with_cache()

    # ------------------------------------------------------------------
    # Cache

    def _cache_path(self) -> str:
        kw_tag = "_".join(self.include_keywords) if self.include_keywords else "all"
        kw_tag = kw_tag.replace("/", "-").replace(" ", "_")[:60]
        return os.path.join(self.data_dir, f".wiki_corpus_{kw_tag}.pkl")

    def _articles_path(self) -> str:
        return os.path.join(self.data_dir, "articles.jsonl")

    def _load_with_cache(self) -> None:
        articles_path = self._articles_path()
        if not os.path.exists(articles_path):
            raise FileNotFoundError(
                f"No se encontró articles.jsonl en {self.data_dir}.\n"
                "Primero corré wikipedia_crawler.py para descargar el dataset."
            )

        cache = self._cache_path()

        if os.path.isfile(cache):
            try:
                if os.path.getmtime(cache) > os.path.getmtime(articles_path):
                    if self.verbose:
                        print("WikipediaCorpus: cargando desde caché…")
                    with open(cache, "rb") as f:
                        self.entries = pickle.load(f)
                    if self.verbose:
                        print(f"✅ WikipediaCorpus listo (caché): {len(self.entries):,} entradas")
                    return
            except Exception as e:
                if self.verbose:
                    print(f"Caché inválido, recargando… ({e})")

        self._load()

        try:
            with open(cache, "wb") as f:
                pickle.dump(self.entries, f, protocol=pickle.HIGHEST_PROTOCOL)
            if self.verbose:
                print(f"💾 Caché guardado en {cache}")
        except Exception as e:
            if self.verbose:
                print(f"Advertencia: no se pudo guardar caché ({e})")

    # ------------------------------------------------------------------
    # Carga desde JSONL

    def _load(self) -> None:
        articles_path = self._articles_path()
        seen_ids: set[str] = set()

        if self.verbose:
            print(f"WikipediaCorpus: leyendo {articles_path} …")

        with open(articles_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Dedup por ID
                doc_id = rec.get("id") or rec.get("url", "")
                if doc_id in seen_ids:
                    continue
                seen_ids.add(doc_id)

                category = rec.get("category", "")

                # Filtro por keywords
                if self.include_keywords:
                    if not any(kw in category for kw in self.include_keywords):
                        continue

                self.entries.append(WikipediaEntry(
                    url         = rec.get("url", ""),
                    title       = rec.get("title", ""),
                    description = rec.get("description", ""),
                    category    = category,
                    source_tag  = rec.get("root_cat", "wikipedia"),
                ))

                if self.max_total_entries and len(self.entries) >= self.max_total_entries:
                    break

        if self.verbose:
            n_cats = len({e.category for e in self.entries})
            print(
                f"✅ WikipediaCorpus listo: {len(self.entries):,} artículos "
                f"en {n_cats:,} categorías"
            )

    # ------------------------------------------------------------------
    # Interfaz CurlieCorpus

    def texts(self) -> list[str]:
        return [e.text() for e in self.entries]

    def stats(self) -> dict:
        cats = defaultdict(int)
        for e in self.entries:
            cats[e.category] += 1
        top = sorted(cats.items(), key=lambda x: -x[1])[:20]
        return {
            "total":      len(self.entries),
            "categories": len(cats),
            "top_20":     top,
        }
