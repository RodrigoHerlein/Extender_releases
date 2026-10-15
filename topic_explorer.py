"""
topic_explorer.py
=================
Pipeline de descubrimiento de tópicos en una sola pasada.

  1. Embeddear el corpus con SBERT (cacheado en disco)
  2. Recuperar candidatos por similitud coseno con los aspect vectors del mapa
  3. Clusterizar con HDBSCAN via BERTopic
  4. Seleccionar el mejor nivel del árbol jerárquico
  5. Rankear y filtrar tópicos

Embeddings: all-MiniLM-L6-v2 (384 dim, pesos fijos).
"""

from __future__ import annotations

import hashlib
import inspect
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import hdbscan
import numpy as np
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from umap import UMAP

from concept_map_encoder import (
    ConceptMapEncoder, RelationAwareEncoder, PathAwareEncoder, HybridEncoder,
)
from extender_bertopic import EmbeddingEngine, ConceptMap

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ExplorerConfig:
    """Todos los hiperparámetros del TopicExplorer."""

    # ── Retrieval ────────────────────────────────────────────────────────
    top_k_per_aspect:   int   = 500
    min_aspect_sim:     float = 0.15
    min_doc_relevance:  float = 0.10

    # ── Clustering ───────────────────────────────────────────────────────
    min_cluster_size:   int   = 5
    min_samples:        int   = 2
    umap_n_components:  int   = 10
    umap_n_neighbors:   int   = 20
    random_state:       int   = 42

    # ── Selección de nivel del árbol ─────────────────────────────────────
    focus_weight:       float = 0.45
    specificity_weight: float = 0.35
    quality_weight:     float = 0.20
    min_nr_topics:      int   = 4
    max_nr_topics:      int   = 15

    # ── Filtro de categoría de inclusión ─────────────────────────────────
    domain_category_keywords: tuple = (
        "Science", "Technology", "Space", "Astronomy", "Astro",
        "Planetary", "Physics", "Geophysics", "Atmospheric", "Geology",
        "Academic", "Research", "Observatory", "Institute", "University",
    )
    domain_category_min_fraction: float = 0.40
    domain_category_penalty:      float = 0.05

    # ── Filtro de categoría de exclusión (descarte duro) ─────────────────
    # Tópicos donde más del exclude_category_max_fraction de sus documentos
    # caen en estas categorías son descartados completamente del output.
    # Útil para mapas no-terrestres donde ciencias terrestres aparecen
    # por overlap semántico (e.g. volcanes/crateres en el mapa de la Luna).
    exclude_category_keywords:     tuple = ()
    exclude_category_max_fraction: float = 0.30

    # ── Filtro de coherencia mínima ──────────────────────────────────────
    min_coherence: float = 0.20

    # ── Filtro de tamaño máximo ──────────────────────────────────────────
    # Tópicos con más del X% del candidate set son clusters residuales
    # genéricos ("space, science, technology") y se descartan.
    max_topic_fraction: float = 0.35

    # ── Output ───────────────────────────────────────────────────────────
    min_output_focus:   float = 0.05
    n_output_topics:    int   = 6

    # ── Ranking score: score = overlap × (rank_base + rank_coherence_w·coh + rank_focus_w·foc)
    rank_base:          float = 0.40
    rank_coherence_w:   float = 0.30
    rank_focus_w:       float = 0.30

    # ── Aspect vectors: umbral de similitud coseno para descartar un aspecto redundante ──
    aspect_sim_threshold: float = 0.97

    # ── Parámetros de representación (PathAware / Hybrid) ────────────────
    # Afectan a los encoders PathAwareEncoder y HybridEncoder.
    # max_depth: profundidad máxima de caminos raíz→nodo enumerados.
    # max_paths_per_node: cuántos caminos distintos se guardan por nodo.
    # hybrid_window_k: cuántas tripletas de la ventana se incluyen (solo Híbrida).
    # n_aspects: cantidad de aspect vectors por encoder (todos los encoders).
    path_max_depth:                  int   = 12
    path_max_paths_per_node:         int   = 200
    hybrid_window_k:                 int   = 3
    n_aspects:                       int   = 4

    # ── Cache ────────────────────────────────────────────────────────────
    # Directorio donde se guarda el archivo .npz con los embeddings del corpus.
    # None = mismo directorio que los archivos del corpus.
    cache_dir:          Optional[str] = None


# ---------------------------------------------------------------------------
# Resultado
# ---------------------------------------------------------------------------


@dataclass
class ExploredTopic:
    """Un tópico retornado por TopicExplorer."""
    topic_id:       int
    keywords:       list[str]
    doc_indices:    list[int]
    coherence:      float
    focus_score:    float
    label:          str = ""
    documents:      list[dict] = field(default_factory=list)
    centroid:       Optional[list[float]] = None  # centroide normalizado del tópico (384-dim)

    def __post_init__(self):
        if not self.label:
            self.label = "_".join(self.keywords[:3])


# ---------------------------------------------------------------------------
# TopicExplorer
# ---------------------------------------------------------------------------


class TopicExplorer:
    """Pipeline de descubrimiento de tópicos en una sola pasada.

    Pasos:
      1. Embeddea el corpus con SBERT (con caché en disco).
      2. Recupera candidatos por similitud coseno con los aspect vectors del mapa.
      3. Clusteriza con HDBSCAN via BERTopic.
      4. Selecciona el mejor nivel del árbol jerárquico.
      5. Rankea y filtra tópicos según coherencia, focus y categoría.
    """

    def __init__(
        self,
        corpus,
        concept_map:    ConceptMap,
        config:         Optional[ExplorerConfig] = None,
        focus_concepts: Optional[list[str]]      = None,
        encoder_class                            = ConceptMapEncoder,
        diagnostic:     bool                     = False,
    ):
        """Inicializa el TopicExplorer cargando el modelo SBERT y el encoder del mapa.

        Args:
            corpus: Corpus a explorar (debe exponer .entries y .texts()). Compatible con
                    PaperDatasetCorpus y WikipediaCorpus.
            concept_map: Mapa conceptual que define el dominio de búsqueda.
            config: Configuración del explorer. Si es None, usa ExplorerConfig().
            focus_concepts: Conceptos a enfatizar en la recuperación y el ranking.
            encoder_class: Clase de encoder a instanciar. Por defecto ConceptMapEncoder.
                           Se pueden pasar RelationAwareEncoder, PathAwareEncoder o
                           HybridEncoder para cambiar la estrategia de representación.
            diagnostic: Si True, captura las etiquetas crudas de HDBSCAN en
                        self._hdbscan_labels tras cada llamada a _fit_bertopic.
        """
        self.corpus      = corpus
        self.cm          = concept_map
        self.cfg         = config or ExplorerConfig()
        self.focus       = focus_concepts or []
        self._diagnostic = diagnostic
        self._log        = self._make_logger()

        # SBERT: los pesos del modelo son fijos, no necesita entrenamiento.
        # fit() solo inicializa el modelo en memoria (descarga ~80 MB la primera vez).
        self._log("🔧 Cargando modelo SBERT (all-MiniLM-L6-v2) …")
        self.engine = EmbeddingEngine(random_state=self.cfg.random_state)
        self.engine.fit()

        # Instanciar el encoder pasando solo los kwargs que acepta su __init__.
        # Esto permite usar ConceptMapEncoder (6 kwargs) o PathAwareEncoder /
        # HybridEncoder (que añaden max_depth, max_paths_per_node, window_k)
        # sin cambiar TopicExplorer.
        _all_enc_kwargs = dict(
            concept_map=concept_map,
            engine=self.engine,
            focus_concepts=self.focus,
            focus_multiplier=3.0,
            n_aspects=self.cfg.n_aspects,
            aspect_sim_threshold=self.cfg.aspect_sim_threshold,
            max_depth=self.cfg.path_max_depth,
            max_paths_per_node=self.cfg.path_max_paths_per_node,
            window_k=self.cfg.hybrid_window_k,
        )
        _sig = inspect.signature(encoder_class.__init__).parameters
        self.encoder = encoder_class(**{k: v for k, v in _all_enc_kwargs.items() if k in _sig})
        self._log(self.encoder.summary())

        self._corpus_embeddings: Optional[np.ndarray] = None
        self._candidate_indices: list[int]             = []
        self._candidate_embs:    Optional[np.ndarray]  = None
        self._candidate_texts:   list[str]             = []
        self._map_vec:           Optional[np.ndarray]  = None
        self.topics:             list[ExploredTopic]   = []

    # ------------------------------------------------------------------
    # API pública

    def run(self) -> list[ExploredTopic]:
        """Ejecuta el pipeline completo y retorna los tópicos rankeados."""
        t0 = time.time()
        self._log(f"\n{'='*60}")
        self._log(f"TopicExplorer — mapa: '{self.cm.title}'")
        self._log(f"Corpus: {len(self.corpus.entries):,} entradas  |  SBERT 384-dim")
        self._log("=" * 60)

        # Paso 1: embeddings del corpus (con caché)
        self._corpus_embeddings = self._get_corpus_embeddings()

        # Paso 2: recuperar candidatos por similitud coseno con los aspects
        self._candidate_indices = self._retrieve_candidates()
        n_raw = len(self._candidate_indices)
        self._log(f"\n📥 {n_raw:,} candidatos recuperados")

        if n_raw < self.cfg.min_cluster_size:
            self._log("⚠  Muy pocos candidatos — bajar min_aspect_sim o min_doc_relevance.")
            return []

        self._candidate_embs  = self._corpus_embeddings[self._candidate_indices]
        self._candidate_texts = [
            self.corpus.entries[i].text() for i in self._candidate_indices
        ]

        # Paso 3: calcular _map_vec y aplicar gate de relevancia
        self._compute_map_vec_and_gate()

        if len(self._candidate_indices) < self.cfg.min_cluster_size:
            self._log("⚠  Muy pocos candidatos después del gate de relevancia.")
            return []

        # Paso 4: clustering
        self._log("🔬 Clustering …")
        model, base_topics, orig_topics, hier_df = self._fit_bertopic(
            self._candidate_embs, self._candidate_texts
        )

        if model is None or base_topics == 0:
            self._log("⚠  El clustering no produjo tópicos.")
            return []

        self._log(f"   Tópicos base: {base_topics}")

        # Paso 5: seleccionar el mejor nivel del árbol jerárquico
        self._log("🌳 Seleccionando mejor nivel del árbol …")
        best_nr, level_scores, level_assignments = self._select_level(
            orig_topics, base_topics, hier_df
        )
        n_pass = level_scores.get(best_nr, {}).get("n_pass", "?")
        self._log(f"   Mejor nivel : nr_topics={best_nr}  ({n_pass} tópicos pasarían filtros)")
        for nr, s in sorted(level_scores.items()):
            self._log(
                f"     nr={nr:3d}  score={s['total']:.3f}  "
                f"(focus={s['focus']:.3f}  spec={s.get('specificity',0):.3f}  "
                f"qual={s.get('quality',0):.3f}  n_real={s.get('n_real_topics',0)})"
            )

        # Paso 6: extraer tópicos y rankear
        best_assignments = level_assignments[best_nr]
        self.topics = self._build_explored_topics(model, best_assignments, orig_topics)
        self.topics = self._rank_topics(self.topics)[:self.cfg.n_output_topics]

        elapsed = time.time() - t0
        self._log(f"\n✅ Listo en {elapsed:.1f}s — {len(self.topics)} tópicos retornados")
        return self.topics

    # ------------------------------------------------------------------
    # Paso 1: embeddings del corpus

    def _get_corpus_embeddings(self) -> np.ndarray:
        """Carga desde caché si es válida; sino embeddea y guarda."""
        cache_path = self._cache_path()
        if cache_path and os.path.exists(cache_path):
            self._log(f"\n💾 Cargando embeddings: {os.path.basename(cache_path)}")
            data = np.load(cache_path, allow_pickle=True)
            cached_urls = set(data["urls"].tolist())
            corpus_urls = {e.url for e in self.corpus.entries}
            if cached_urls == corpus_urls:
                embs = data["embeddings"].astype(np.float64)
                if embs.shape[1] == 384:
                    self._log(f"   ✓ Caché válida ({embs.shape[0]:,} entradas)")
                    return embs
                self._log("   ✗ Caché de dimensión incorrecta — reconstruyendo")
            else:
                self._log("   ✗ Caché desactualizada — reconstruyendo")

        self._log(f"\n🔢 Embeddiendo {len(self.corpus.entries):,} entradas con SBERT …")
        self._log("   (esto tarda ~5-10 min la primera vez; las siguientes usan caché)")
        texts  = self.corpus.texts()
        embs_l = []
        for i in range(0, len(texts), 256):
            embs_l.append(self.engine.embed(texts[i : i + 256]))
            if (i // 256) % 20 == 0 and i > 0:
                self._log(f"   {i:,}/{len(texts):,} embeddidos …")
        result = np.vstack(embs_l).astype(np.float64)

        if cache_path:
            self._log(f"💾 Guardando caché: {os.path.basename(cache_path)}")
            np.savez_compressed(
                cache_path,
                embeddings=result.astype(np.float32),
                urls=np.array([e.url for e in self.corpus.entries], dtype=object),
            )
        return result

    # ------------------------------------------------------------------
    # Paso 2: retrieval

    def _retrieve_candidates(self) -> list[int]:
        """
        Para cada aspect vector recupera los top_k_per_aspect documentos
        del corpus por similitud coseno.
        Un documento entra como candidato si pasa ambos gates:
          - similitud con el aspect   >= min_aspect_sim
          - similitud con el centroid >= min_doc_relevance
        """
        norms  = np.linalg.norm(self._corpus_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normed = self._corpus_embeddings / norms

        centroid      = self.encoder.centroid
        centroid_sims = normed @ centroid
        scores: dict[int, float] = {}

        for aspect in self.encoder.aspects:
            aspect_sims = normed @ aspect
            k = min(self.cfg.top_k_per_aspect, len(aspect_sims))
            top_idx = np.argpartition(aspect_sims, -k)[-k:]
            for idx in top_idx:
                a_sim = float(aspect_sims[idx])
                c_sim = float(centroid_sims[idx])
                if a_sim < self.cfg.min_aspect_sim:
                    continue
                if c_sim < self.cfg.min_doc_relevance:
                    continue
                if idx not in scores or scores[idx] < a_sim:
                    scores[idx] = a_sim

        return sorted(scores.keys(), key=lambda i: float(centroid_sims[i]), reverse=True)

    # ------------------------------------------------------------------
    # Paso 3: map_vec y gate de relevancia

    def _compute_map_vec_and_gate(self) -> None:
        """Calcula el vector del mapa y aplica un gate de relevancia al candidate set.

        Embeddea los top-30 términos ponderados del mapa para obtener _map_vec,
        luego descarta candidatos con similitud < min_doc_relevance, garantizando
        retener al menos min_cluster_size * 4 documentos.
        """
        map_text = " ".join(
            t for t, _ in sorted(
                self.cm.term_weights(focus_concepts=self.focus, focus_multiplier=3.0).items(),
                key=lambda x: -x[1],
            )[:30]
        )
        v = self.engine.embed([map_text])[0]
        n = np.linalg.norm(v)
        self._map_vec = v / n if n > 0 else v

        cand_norms = np.linalg.norm(self._candidate_embs, axis=1, keepdims=True)
        cand_norms[cand_norms == 0] = 1.0
        refit_sims = (self._candidate_embs / cand_norms) @ self._map_vec

        gate      = self.cfg.min_doc_relevance
        keep_mask = refit_sims >= gate
        min_keep  = max(self.cfg.min_cluster_size * 4, 30)
        if int(keep_mask.sum()) < min_keep:
            keep_mask = np.zeros(len(refit_sims), dtype=bool)
            keep_mask[np.argsort(refit_sims)[-min_keep:]] = True

        keep_idx = np.where(keep_mask)[0].tolist()
        n_before = len(self._candidate_indices)
        self._candidate_indices = [self._candidate_indices[i] for i in keep_idx]
        self._candidate_embs    = self._candidate_embs[keep_idx]
        self._candidate_texts   = [self._candidate_texts[i] for i in keep_idx]
        self._log(
            f"🔧 Gate de relevancia: {len(self._candidate_indices):,} / {n_before:,} "
            f"candidatos retenidos (sim >= {gate:.2f})"
        )

    # ------------------------------------------------------------------
    # Paso 4: clustering

    def _fit_bertopic(self, embs: np.ndarray, texts: list[str]):
        """Ajusta BERTopic sobre el subconjunto de candidatos.

        Adapta automáticamente n_neighbors y n_components de UMAP al tamaño
        del corpus. Construye la jerarquía de tópicos si hay >= 2 tópicos base.

        Args:
            embs: Array de embeddings de los candidatos, shape (N, 384).
            texts: Textos paralelos a los embeddings.

        Returns:
            Tupla (model, base_count, orig_topics, hier_df). Retorna
            (None, 0, [], None) si el clustering falla o el corpus es muy pequeño.
        """
        n = len(texts)
        if n < self.cfg.min_cluster_size:
            return None, 0, [], None

        n_neighbors  = min(self.cfg.umap_n_neighbors, max(2, n // 3))
        n_components = min(self.cfg.umap_n_components, max(1, n - 2))

        model = BERTopic(
            umap_model=UMAP(
                n_neighbors=n_neighbors, n_components=n_components,
                min_dist=0.0, metric="cosine",
                random_state=self.cfg.random_state,
            ),
            hdbscan_model=hdbscan.HDBSCAN(
                min_cluster_size=self.cfg.min_cluster_size,
                min_samples=self.cfg.min_samples,
                metric="euclidean",
                cluster_selection_method="eom",
                prediction_data=True,
            ),
            vectorizer_model=CountVectorizer(
                stop_words="english", min_df=2, ngram_range=(1, 2),
            ),
            ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
            verbose=False,
        )
        try:
            model.fit_transform(texts, embeddings=embs)
        except Exception as exc:
            logger.warning("BERTopic falló (%d docs): %s", n, exc)
            return None, 0, [], None

        info       = model.get_topic_info()
        base_count = int((info["Topic"] >= 0).sum())
        orig       = list(model.topics_)
        if self._diagnostic:
            self._hdbscan_labels = orig

        hier_df = None
        if base_count >= 2:
            try:
                hier_df = model.hierarchical_topics(texts)
            except Exception as exc:
                logger.warning("hierarchical_topics falló: %s", exc)

        return model, base_count, orig, hier_df

    # ------------------------------------------------------------------
    # Paso 5: selección de nivel del árbol

    def _select_level(self, orig_topics, base_count, hier_df):
        """Selecciona el mejor número de tópicos evaluando niveles del árbol jerárquico.

        Para cada nr en [min_nr_topics, max_nr_topics] calcula un score combinando
        focus (similitud al mapa), specificity (distribución de tamaños de tópicos)
        y quality (fracción de tópicos que pasan los filtros de salida).

        Args:
            orig_topics: Asignaciones de tópico originales de HDBSCAN (lista de ints).
            base_count: Número de tópicos base producidos por HDBSCAN.
            hier_df: DataFrame jerárquico de BERTopic, o None.

        Returns:
            Tupla (best_nr, level_scores, level_assignments) donde:
              - best_nr: el nr_topics óptimo.
              - level_scores: dict {nr: {focus, specificity, quality, total, ...}}.
              - level_assignments: dict {nr: lista de asignaciones de tópico}.
        """
        n_docs   = len(orig_topics)

        merge_tree: list[tuple[float, int, int]] = []
        if hier_df is not None:
            try:
                for _, row in hier_df.iterrows():
                    merge_tree.append((
                        float(row["Distance"]),
                        int(row["Child_Left_ID"]),
                        int(row["Child_Right_ID"]),
                    ))
                merge_tree.sort(key=lambda x: x[0])
            except Exception as exc:
                logger.warning("No se pudo parsear el árbol: %s", exc)

        node_leaves: dict[int, set] = {i: {i} for i in range(base_count)}
        for dist, left, right in merge_tree:
            parent_id = max(
                (k for k in node_leaves if k >= base_count),
                default=base_count - 1,
            ) + 1
            left_leaves  = node_leaves.get(left,  {left}  if left  < base_count else set())
            right_leaves = node_leaves.get(right, {right} if right < base_count else set())
            node_leaves[parent_id] = left_leaves | right_leaves

        def assignments_at_nr(nr: int) -> list[int]:
            if nr >= base_count or not merge_tree:
                return list(orig_topics)
            uf = list(range(base_count))

            def find(x):
                root = x
                while uf[root] != root:
                    root = uf[root]
                while uf[x] != root:
                    uf[x], x = root, uf[x]
                return root

            def union(a, b):
                ra, rb = find(a), find(b)
                if ra == rb:
                    return False
                uf[ra] = rb
                return True

            n_groups = base_count
            for dist, left, right in merge_tree:
                if n_groups <= nr:
                    break
                left_leaves  = node_leaves.get(left,  {left}  if left  < base_count else set())
                right_leaves = node_leaves.get(right, {right} if right < base_count else set())
                if not left_leaves or not right_leaves:
                    continue
                all_leaves = left_leaves | right_leaves
                reps = list(all_leaves)
                for i in range(1, len(reps)):
                    if union(reps[0], reps[i]):
                        n_groups -= 1
                    if n_groups <= nr:
                        break

            return [
                (find(tid) if 0 <= tid < base_count else -1)
                for tid in orig_topics
            ]

        lo = max(self.cfg.min_nr_topics, 2)
        hi = min(self.cfg.max_nr_topics, base_count)
        nr_candidates = sorted(set(list(range(lo, hi + 1)) + [base_count]))

        scores:      dict[int, dict] = {}
        assignments: dict[int, list] = {}
        prev_n_real = -1

        kws_d    = self.cfg.domain_category_keywords
        exc_kws  = self.cfg.exclude_category_keywords
        min_frac = self.cfg.domain_category_min_fraction
        max_exc  = self.cfg.exclude_category_max_fraction
        max_docs = int(n_docs * self.cfg.max_topic_fraction)

        def in_domain(doc):
            return any(kw in doc.get("category", "") for kw in kws_d)

        def is_excluded(doc):
            return exc_kws and any(kw in doc.get("category", "") for kw in exc_kws)

        for nr in nr_candidates:
            asgn          = assignments_at_nr(nr)
            asgn_arr      = np.array(asgn)
            unique_topics = [t for t in set(asgn) if t >= 0]
            n_real        = len(unique_topics)

            if n_real == prev_n_real and nr != base_count:
                if nr - 1 in scores:
                    scores[nr]      = dict(scores[nr - 1])
                    assignments[nr] = list(assignments[nr - 1])
                    continue
            prev_n_real = n_real

            if not unique_topics:
                continue

            topic_sizes, topic_focuses = [], []
            n_pass = 0

            for tid in unique_topics:
                doc_idxs = np.where(asgn_arr == tid)[0]
                if len(doc_idxs) == 0:
                    continue
                size = len(doc_idxs)
                topic_sizes.append(size)

                t_embs   = self._candidate_embs[doc_idxs]
                centroid = t_embs.mean(axis=0)
                cn       = np.linalg.norm(centroid)
                coh  = float((t_embs @ (centroid / cn if cn > 0 else centroid)).mean())
                foc  = float((centroid / cn if cn > 0 else centroid) @ self._map_vec)
                topic_focuses.append(foc)

                global_ids = [self._candidate_indices[i] for i in doc_idxs if i < len(self._candidate_indices)]
                docs = [self.corpus.entries[gi].to_dict() for gi in global_ids if gi < len(self.corpus.entries)]
                frac_dom = sum(1 for d in docs if in_domain(d)) / len(docs) if docs else 0.0
                frac_exc = sum(1 for d in docs if is_excluded(d)) / len(docs) if docs else 0.0

                if (coh >= self.cfg.min_coherence and foc >= self.cfg.min_output_focus
                        and frac_dom >= min_frac and size <= max_docs
                        and (not exc_kws or frac_exc <= max_exc)):
                    n_pass += 1

            if not topic_sizes:
                continue

            total_docs = sum(topic_sizes)
            n_t        = len(topic_sizes)
            focus      = sum(f * s for f, s in zip(topic_focuses, topic_sizes)) / total_docs

            if n_t > 1:
                sa   = np.array(topic_sizes, dtype=float)
                ss   = np.sort(sa)
                gini = (2 * np.sum(np.arange(1, n_t+1) * ss) / (n_t * ss.sum()) - (n_t + 1) / n_t)
                specificity = 1.0 - float(np.clip(gini, 0, 1))
            else:
                specificity = 0.0

            quality = n_pass / n_t if n_t > 0 else 0.0

            scores[nr] = {
                "focus":         focus,
                "specificity":   specificity,
                "quality":       quality,
                "n_real_topics": n_real,
                "n_pass":        n_pass,
                "total": (
                    self.cfg.focus_weight       * focus
                    + self.cfg.specificity_weight * specificity
                    + self.cfg.quality_weight     * quality
                ),
            }
            assignments[nr] = asgn

        if not scores:
            asgn = list(orig_topics)
            assignments[base_count] = asgn
            scores[base_count] = {"focus": 0, "specificity": 0, "quality": 0,
                                   "total": 0, "n_real_topics": base_count, "n_pass": 0}

        best_nr = max(scores, key=lambda k: scores[k]["total"])
        return best_nr, scores, assignments

    # ------------------------------------------------------------------
    # Paso 6: construir tópicos

    def _build_explored_topics(self, model, assignments, orig_topics):
        """Construye la lista de ExploredTopic a partir de las asignaciones del nivel seleccionado.

        Para cada cluster: calcula centroide, coherencia, focus_score, keywords
        con TF-IDF y lista de documentos ordenados por similitud al centroide.

        Args:
            model: Modelo BERTopic ajustado.
            assignments: Lista de asignaciones de tópico para el nivel seleccionado.
            orig_topics: Asignaciones originales de HDBSCAN (no usadas directamente).

        Returns:
            Lista de ExploredTopic sin filtrar ni rankear.
        """
        asgn_arr = np.array(assignments)
        embs     = self._candidate_embs
        results: list[ExploredTopic] = []

        merged_ids = sorted(t for t in set(assignments) if t >= 0)
        if not merged_ids:
            return results

        try:
            tfidf = TfidfVectorizer(
                stop_words="english", ngram_range=(1, 2),
                min_df=2, max_features=5000, sublinear_tf=True,
            )
            tfidf_matrix = tfidf.fit_transform(self._candidate_texts)
            vocab    = tfidf.get_feature_names_out()
            tfidf_ok = True
        except Exception:
            tfidf_ok = False

        for merged_id in merged_ids:
            local_ids = np.where(asgn_arr == merged_id)[0].tolist()
            if not local_ids:
                continue

            cluster_embs = embs[local_ids]
            centroid     = cluster_embs.mean(axis=0)
            cn           = np.linalg.norm(centroid)
            centroid_n   = centroid / cn if cn > 0 else centroid

            coherence   = float((cluster_embs @ centroid_n).mean())
            focus_score = float(centroid_n @ self._map_vec)

            if tfidf_ok:
                cluster_tfidf = np.asarray(tfidf_matrix[local_ids].mean(axis=0)).flatten()
                top_j    = np.argsort(cluster_tfidf)[-10:][::-1]
                keywords = [vocab[j] for j in top_j if cluster_tfidf[j] > 0]
            else:
                keywords = [f"topic_{merged_id}"]

            global_ids = [self._candidate_indices[i] for i in local_ids if i < len(self._candidate_indices)]
            docs = [self.corpus.entries[gi].to_dict() for gi in global_ids if gi < len(self.corpus.entries)]

            if len(local_ids) > 1:
                sims  = cluster_embs @ centroid_n
                order = np.argsort(sims)[::-1]
                docs  = [docs[i] for i in order if i < len(docs)]

            results.append(ExploredTopic(
                topic_id    = merged_id,
                keywords    = keywords,
                doc_indices = global_ids,
                coherence   = coherence,
                focus_score = focus_score,
                documents   = docs,
                centroid    = centroid_n.tolist(),
            ))

        return results

    # ------------------------------------------------------------------
    # Paso 7: filtrar y rankear

    def _rank_topics(self, topics: list[ExploredTopic]) -> list[ExploredTopic]:
        """Filtra y rankea tópicos según focus, coherencia, tamaño y categoría.

        Aplica en orden: filtro por focus mínimo, coherencia mínima, tamaño máximo,
        exclusión de categorías, y finalmente puntúa con overlap de términos del mapa.
        Tópicos con fracción de documentos en dominio < domain_category_min_fraction
        reciben un multiplicador de penalización.

        Args:
            topics: Lista de ExploredTopic a filtrar y rankear.

        Returns:
            Lista de tópicos ordenada de mayor a menor puntuación.
        """
        n_candidates = len(self._candidate_indices)

        if self.cfg.min_output_focus > 0:
            n_before = len(topics)
            topics   = [t for t in topics if t.focus_score >= self.cfg.min_output_focus]
            if len(topics) < n_before:
                self._log(f"   Descartados {n_before - len(topics)} tópico(s): focus < {self.cfg.min_output_focus:.2f}")

        if self.cfg.min_coherence > 0:
            n_before = len(topics)
            topics   = [t for t in topics if t.coherence >= self.cfg.min_coherence]
            if len(topics) < n_before:
                self._log(f"   Descartados {n_before - len(topics)} tópico(s): coherencia < {self.cfg.min_coherence:.2f}")

        if self.cfg.max_topic_fraction > 0 and n_candidates > 0:
            max_docs = int(n_candidates * self.cfg.max_topic_fraction)
            n_before = len(topics)
            topics   = [t for t in topics if len(t.documents) <= max_docs]
            if len(topics) < n_before:
                self._log(f"   Descartados {n_before - len(topics)} tópico(s): >{max_docs} docs")

        exc_kws = self.cfg.exclude_category_keywords
        max_exc = self.cfg.exclude_category_max_fraction
        if exc_kws:
            def frac_excluded(t):
                if not t.documents:
                    return 0.0
                return sum(
                    1 for d in t.documents
                    if any(kw in d.get("category", "") for kw in exc_kws)
                ) / len(t.documents)

            n_before = len(topics)
            topics   = [t for t in topics if frac_excluded(t) <= max_exc]
            if len(topics) < n_before:
                self._log(f"   Descartados {n_before - len(topics)} tópico(s): exclusión de categoría")

        tw        = self.cm.term_weights(focus_concepts=self.focus, focus_multiplier=3.0)
        map_terms = set(tw.keys())
        inc_kws   = self.cfg.domain_category_keywords
        min_frac  = self.cfg.domain_category_min_fraction
        penalty   = self.cfg.domain_category_penalty

        def in_domain(doc):
            return any(kw in doc.get("category", "") for kw in inc_kws)

        def score(t: ExploredTopic) -> float:
            overlap = sum(tw.get(w, 0.0) for w in frozenset(t.keywords) & map_terms)
            base    = overlap * (self.cfg.rank_base + self.cfg.rank_coherence_w * t.coherence + self.cfg.rank_focus_w * t.focus_score)
            if t.documents:
                frac_in = sum(1 for d in t.documents if in_domain(d)) / len(t.documents)
                if frac_in < min_frac:
                    base *= penalty
            return base

        ranked = sorted(topics, key=score, reverse=True)

        for t in ranked:
            if t.documents:
                n      = len(t.documents)
                frac_i = sum(1 for d in t.documents if in_domain(d)) / n
                self._log(
                    f"   📊 '{t.label[:40]}': focus={t.focus_score:.3f}  "
                    f"coh={t.coherence:.3f}  n={n}  dom={frac_i:.0%}"
                    + ("  ⚠ penalizado" if frac_i < min_frac else "")
                )

        return ranked

    # ------------------------------------------------------------------
    # Cache

    def _cache_path(self) -> Optional[str]:
        """Construye la ruta del archivo de caché de embeddings (.npz).

        La clave del caché incluye número de entradas, include_keywords y
        el nombre del modelo (sbert384) para invalidación automática.

        Returns:
            Ruta absoluta al archivo .npz, o None si no hay directorio disponible.
        """
        cache_dir = self.cfg.cache_dir or self.corpus.data_dir
        if not cache_dir:
            return None
        key = (
            f"{len(self.corpus.entries)}"
            f"|{'|'.join(self.corpus.include_keywords or [])}"
            f"|sbert384"
        )
        h = hashlib.md5(key.encode()).hexdigest()[:10]
        return os.path.join(cache_dir, f"corpus_embeddings_sbert_{h}.npz")

    def _make_logger(self):
        """Crea una función de logging que imprime siempre a stdout.

        Returns:
            Función _log(msg) que imprime el mensaje.
        """
        def _log(msg: str) -> None:
            print(msg)
        return _log


# ---------------------------------------------------------------------------
# Helpers de salida
# ---------------------------------------------------------------------------


def print_topics(topics: list[ExploredTopic], top_docs: int = 3) -> None:
    """Imprime los tópicos descubiertos a stdout en formato legible.

    Args:
        topics: Lista de ExploredTopic a imprimir.
        top_docs: Número de documentos representativos a mostrar por tópico.
    """
    print(f"\n{'='*65}")
    print(f"Topic Explorer — {len(topics)} tópicos sugeridos")
    print("=" * 65)
    for i, t in enumerate(topics, 1):
        print(f"\n📌 Tópico {i}: {t.label}")
        print(f"   Keywords   : {', '.join(t.keywords[:8])}")
        print(f"   Coherencia : {t.coherence:.3f}   Focus: {t.focus_score:.3f}   Docs: {len(t.documents)}")
        for d in t.documents[:top_docs]:
            print(f"     • {d['title'][:65]}")
            if d.get("snippet"):
                print(f"       {d['snippet'][:110]}…")
            print(f"       [{d.get('category','?')[:55]}]")
            print(f"       🔗 {d['url']}")
    print("\n" + "=" * 65)


def save_topics_json(topics: list[ExploredTopic], path: str) -> None:
    """Serializa los tópicos descubiertos a un archivo JSON.

    Args:
        topics: Lista de ExploredTopic a serializar.
        path: Ruta del archivo JSON de salida.
    """
    import json
    data = [
        {
            "topic_id":    i,
            "label":       t.label,
            "keywords":    t.keywords,
            "coherence":   round(t.coherence, 4),
            "focus_score": round(t.focus_score, 4),
            "documents":   t.documents,
        }
        for i, t in enumerate(topics, 1)
    ]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    print(f"💾 Resultados guardados en {path}")
