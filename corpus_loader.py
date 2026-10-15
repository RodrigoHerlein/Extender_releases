"""
corpus_loader.py
================
Loaders de corpus para el pipeline EXTENDER.

Clases disponibles
------------------
PaperDatasetCorpus  — corpus del dataset ODP/Curlie etiquetado (paper dataset).
WikipediaCorpus     — corpus de artículos de Wikipedia (salida de wikipedia_crawler.py).

Ambas exponen la misma interfaz que CurlieCorpus y pueden usarse como
drop-in replacement en TopicExplorer:

    from corpus_loader import PaperDatasetCorpus, WikipediaCorpus

    corpus = PaperDatasetCorpus(data_dir="Datasets/dataset_paper")
    # o
    corpus = WikipediaCorpus(data_dir="Datasets/dataset_paper/Wikipedia")

    exp = TopicExplorer(corpus=corpus, ...)
"""

from __future__ import annotations

import json
import os
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional


# ===========================================================================
# PaperEntry / PaperDatasetCorpus
# ===========================================================================

@dataclass
class PaperEntry:
    """Un documento del dataset de papers ODP/Curlie."""
    url:         str
    title:       str
    description: str
    category:    str = ""
    _full_text:  str = field(default="", repr=False)

    def text(self) -> str:
        """Texto combinado para embedding — mirrors CurlieEntry.text()."""
        cat_words = self.category.replace("/", " ").replace("_", " ")
        body = self._full_text or f"{self.title}. {self.description}"
        return f"{body} {cat_words}".strip()

    def to_dict(self) -> dict:
        """Serializa la entrada a un diccionario compatible con el pipeline.

        Returns:
            Diccionario con claves url, title, snippet, category y source_tag.
        """
        return {
            "url":        self.url,
            "title":      self.title,
            "snippet":    self.description[:300],
            "category":   self.category,
            "source_tag": "paper_dataset",
        }


class PaperDatasetCorpus:
    """
    Drop-in replacement for CurlieCorpus using the labeled ODP/Curlie
    paper dataset (texts/ + urls/ + topic_names.txt).

    Parameters
    ----------
    data_dir : str
        Path to the Dataset directory (contains topic_names.txt, texts/, urls/).
    max_docs_per_topic : int
        Limit documents loaded per topic (default 150 → ~80 k docs total).
    max_chars_per_doc : int
        Truncate each document text to this many characters before embedding.
    verbose : bool
        Print progress to stdout.
    """

    def __init__(
        self,
        data_dir:           str,
        max_docs_per_topic: int  = 150,
        max_chars_per_doc:  int  = 2000,
        verbose:            bool = True,
    ):
        self.data_dir           = data_dir
        self.max_docs_per_topic = max_docs_per_topic
        self.max_chars_per_doc  = max_chars_per_doc
        self.verbose            = verbose
        self.entries: list[PaperEntry] = []
        # Compatibility attribute — CurlieCorpus has include_keywords;
        # TopicExplorer._cache_path() uses it to build the cache key.
        self.include_keywords: tuple | None = None
        self._load_with_cache()

    def _cache_path(self) -> str:
        """Construye la ruta del archivo pickle de caché del corpus."""
        return os.path.join(
            self.data_dir,
            f".paper_corpus_{self.max_docs_per_topic}_{self.max_chars_per_doc}.pkl",
        )

    def _load_with_cache(self) -> None:
        """Carga el corpus desde caché pickle si es válido, o desde disco si no."""
        cache = self._cache_path()
        names_path = os.path.join(self.data_dir, "topic_names.txt")

        if os.path.isfile(cache):
            try:
                if os.path.getmtime(cache) > os.path.getmtime(names_path):
                    if self.verbose:
                        print("PaperDatasetCorpus: cargando desde caché...")
                    with open(cache, "rb") as f:
                        self.entries = pickle.load(f)
                    if self.verbose:
                        print(f"✅ PaperDatasetCorpus listo (caché): {len(self.entries):,} entradas")
                    return
            except Exception as e:
                if self.verbose:
                    print(f"Caché inválido, recargando desde disco... ({e})")

        self._load()
        try:
            with open(cache, "wb") as f:
                pickle.dump(self.entries, f, protocol=pickle.HIGHEST_PROTOCOL)
            if self.verbose:
                print(f"💾 Caché guardado en {cache}")
        except Exception as e:
            if self.verbose:
                print(f"Advertencia: no se pudo guardar el caché ({e})")

    def _load(self) -> None:
        """Carga el corpus desde disco leyendo topic_names.txt, texts/ y urls/."""
        names_path = os.path.join(self.data_dir, "topic_names.txt")
        topic_to_cat: dict[str, str] = {}
        with open(names_path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t", 1)
                if len(parts) == 2:
                    topic_to_cat[parts[0].strip()] = parts[1].strip()

        texts_root = os.path.join(self.data_dir, "texts")
        urls_root  = os.path.join(self.data_dir, "urls")

        _has_texts = os.path.isdir(texts_root) and any(
            os.path.isdir(os.path.join(texts_root, d)) for d in os.listdir(texts_root)
        ) if os.path.isdir(texts_root) else False
        if not _has_texts:
            import tarfile
            texts_tar = os.path.join(self.data_dir, "texts.tar.gz")
            urls_tar  = os.path.join(self.data_dir, "urls.tar.gz")
            if os.path.isfile(texts_tar):
                if self.verbose:
                    print("PaperDatasetCorpus: extrayendo texts.tar.gz ...")
                with tarfile.open(texts_tar, "r:gz") as tf:
                    tf.extractall(self.data_dir)
            if os.path.isfile(urls_tar):
                if self.verbose:
                    print("PaperDatasetCorpus: extrayendo urls.tar.gz ...")
                with tarfile.open(urls_tar, "r:gz") as tf:
                    tf.extractall(self.data_dir)

        if self.verbose:
            print(f"PaperDatasetCorpus: {len(topic_to_cat):,} topics — "
                  f"loading up to {self.max_docs_per_topic} docs/topic ...")

        for tid, category in sorted(topic_to_cat.items()):
            topic_dir = os.path.join(texts_root, tid)
            if not os.path.isdir(topic_dir):
                continue

            urls: list[str] = []
            url_file = os.path.join(urls_root, f"{tid}.txt")
            if os.path.isfile(url_file):
                try:
                    with open(url_file, encoding="utf-8", errors="ignore") as f:
                        urls = [ln.strip() for ln in f if ln.strip()]
                except Exception:
                    pass

            try:
                fnames = sorted(
                    [fn for fn in os.listdir(topic_dir) if fn.endswith(".txt")],
                    key=lambda fn: int(fn.replace(".txt", "")),
                )[: self.max_docs_per_topic]
            except Exception:
                fnames = sorted(
                    [fn for fn in os.listdir(topic_dir) if fn.endswith(".txt")]
                )[: self.max_docs_per_topic]

            for i, fname in enumerate(fnames):
                fpath = os.path.join(topic_dir, fname)
                try:
                    with open(fpath, encoding="utf-8", errors="ignore") as f:
                        raw = f.read(self.max_chars_per_doc)
                except Exception:
                    continue

                lines = raw.strip().split("\n", 2)
                title       = lines[0][:200] if lines else fname
                description = lines[1][:500] if len(lines) > 1 else ""
                url         = urls[i] if i < len(urls) else f"paper://{tid}/{fname}"

                self.entries.append(PaperEntry(
                    url         = url,
                    title       = title,
                    description = description,
                    category    = category,
                    _full_text  = raw,
                ))

        if self.verbose:
            n_cats = len(Counter(e.category for e in self.entries))
            print(f"✅ PaperDatasetCorpus listo: {len(self.entries):,} entradas "
                  f"en {n_cats:,} categorías")

    def texts(self) -> list[str]:
        """Retorna el texto de embedding de cada entrada del corpus."""
        return [e.text() for e in self.entries]


# ===========================================================================
# WikipediaEntry / WikipediaCorpus
# ===========================================================================

@dataclass
class WikipediaEntry:
    """Un artículo de Wikipedia."""
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


class WikipediaCorpus:
    """
    Corpus de Wikipedia compatible con la interfaz de CurlieCorpus.
    Espera que wikipedia_crawler.py haya generado articles.jsonl en data_dir.

    Parameters
    ----------
    data_dir : str
        Carpeta que contiene articles.jsonl (salida de wikipedia_crawler.py).
    include_keywords : tuple[str] | None
        Si se provee, solo se incluyen artículos cuya categoría contenga
        alguna de las palabras clave. None = incluir todo.
        Ejemplo: ("Computers/AI", "Computers/Programming")
    max_total_entries : int | None
        Límite opcional de entradas a cargar.
    verbose : bool
    """

    def __init__(
        self,
        data_dir:          str,
        include_keywords:  Optional[tuple[str, ...]] = None,
        max_total_entries: Optional[int] = None,
        verbose:           bool = True,
    ):
        self.data_dir          = data_dir
        self.include_keywords  = include_keywords
        self.max_total_entries = max_total_entries
        self.verbose           = verbose
        self.entries: list[WikipediaEntry] = []
        self._load_with_cache()

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

                doc_id = rec.get("id") or rec.get("url", "")
                if doc_id in seen_ids:
                    continue
                seen_ids.add(doc_id)

                category = rec.get("category", "")

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
