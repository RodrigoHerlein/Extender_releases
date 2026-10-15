"""
concept_map_encoder.py
======================
Converts a ConceptMap into:
  1. A single dense query vector (weighted centroid of concept embeddings)
  2. A ranked list of query strings for BM25 pre-filtering (optional speedup)
"""

from __future__ import annotations

import numpy as np
from typing import Optional
from collections import defaultdict

# Import from the refactored extender module
from extender_bertopic import ConceptMap, EmbeddingEngine, tokenise, _DOMAIN_SEEDS


# ---------------------------------------------------------------------------
# ConceptMapEncoder
# ---------------------------------------------------------------------------


class ConceptMapEncoder:
    """Codifica un mapa conceptual en vectores densos para búsqueda semántica.

    Genera un vector centroide (promedio ponderado de embeddings de nodos)
    y hasta n_aspects aspect vectors (sub-centroides por vecindad) para
    representar múltiples facetas del mapa.

    Attributes:
        cm: Mapa conceptual fuente.
        engine: Motor de embeddings SBERT.
        focus_set: Conjunto de conceptos a enfatizar.
        focus_multiplier: Factor de peso para conceptos en foco.
        aspect_sim_threshold: Umbral coseno para descartar aspects redundantes.
        n_aspects: Número máximo de aspect vectors a generar.
    """

    def __init__(
        self,
        concept_map:          ConceptMap,
        engine:               EmbeddingEngine,
        focus_concepts:       Optional[list[str]] = None,
        focus_multiplier:     float               = 3.0,
        n_aspects:            int                 = 4,
        aspect_sim_threshold: float               = 0.97,
    ):
        """Inicializa el encoder.

        Args:
            concept_map: Mapa conceptual a codificar.
            engine: Motor de embeddings ya fiteado.
            focus_concepts: Conceptos a enfatizar con mayor peso.
            focus_multiplier: Factor multiplicador para conceptos en foco.
            n_aspects: Número máximo de aspect vectors a generar (incluye el centroide).
            aspect_sim_threshold: Similitud coseno máxima para considerar dos aspects distintos.
        """
        self.cm                   = concept_map
        self.engine               = engine
        self.focus_set            = set(focus_concepts or [])
        self.focus_multiplier     = focus_multiplier
        self.aspect_sim_threshold = aspect_sim_threshold
        self.n_aspects        = n_aspects

        self._term_weights = concept_map.term_weights(
            focus_concepts=focus_concepts,
            focus_multiplier=focus_multiplier,
        )
        self._topo = concept_map.topological_weights()

        # Computed on first access
        self._centroid:     Optional[np.ndarray]       = None
        self._aspects:      Optional[list[np.ndarray]] = None
        self._label_embs:   Optional[dict[str, np.ndarray]] = None

    # ------------------------------------------------------------------
    # Main interface

    @property
    def centroid(self) -> np.ndarray:
        """Vector centroide normalizado del mapa (calculado bajo demanda).

        Returns:
            Array unitario de 384 dimensiones.
        """
        if self._centroid is None:
            self._centroid = self._build_centroid()
        return self._centroid

    @property
    def aspects(self) -> list[np.ndarray]:
        """Lista de aspect vectors normalizados (calculados bajo demanda).

        El primer elemento siempre es el centroide global. Los siguientes
        representan sub-zonas del mapa por vecindad de conceptos.

        Returns:
            Lista de arrays unitarios de 384 dimensiones.
        """
        if self._aspects is None:
            self._aspects = self._build_aspects()
        return self._aspects

    @property
    def label_embeddings(self) -> dict[str, np.ndarray]:
        """Per-concept normalised embedding vectors."""
        if self._label_embs is None:
            labels = list(self.cm.nodes.keys())
            raw    = self.engine.embed(labels)
            self._label_embs = {}
            for label, vec in zip(labels, raw):
                n = np.linalg.norm(vec)
                self._label_embs[label] = vec / n if n > 0 else vec
        return self._label_embs

    def query_strings(self, max_queries: int = 12) -> list[str]:
        """Genera strings de consulta a partir de los arcos del mapa para BM25.

        Ordena los arcos por peso topológico y construye queries "src dst guard".

        Args:
            max_queries: Número máximo de queries a retornar.

        Returns:
            Lista de strings de consulta.
        """
        topo = self._topo
        focus = self.focus_set
        mult  = self.focus_multiplier

        # Score edges
        scored: list[tuple[float, str, str]] = []
        for src, link, dst in self.cm.edges:
            w = topo.get(src, 0.0) + topo.get(dst, 0.0)
            if src in focus or dst in focus:
                w *= mult
            scored.append((w, src, dst))
        scored.sort(reverse=True)

        queries: list[str] = []
        seen:    set[str]  = set()
        title_toks = tokenise(self.cm.title)[:3]
        guard = " ".join(title_toks)

        for _, src, dst in scored:
            q = f"{src} {dst} {guard}".strip()
            if q not in seen:
                queries.append(q)
                seen.add(q)
            if len(queries) >= max_queries:
                break

        return queries

    def summary(self) -> str:
        """Retorna un resumen legible del encoder con estadísticas del mapa.

        Returns:
            String multilínea con título, número de conceptos, arcos,
            conceptos en foco, top-10 términos y número de aspects.
        """
        top_terms = sorted(
            self._term_weights.items(), key=lambda x: -x[1]
        )[:10]
        lines = [
            f"ConceptMapEncoder: '{self.cm.title}'",
            f"  Concepts : {len(self.cm.nodes)}",
            f"  Edges    : {len(self.cm.edges)}",
            f"  Focus    : {', '.join(self.focus_set) or 'none'}",
            f"  Top terms: {', '.join(t for t, _ in top_terms)}",
            f"  Aspects  : {len(self.aspects)} vectors",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private builders

    def _build_centroid(self) -> np.ndarray:
        """Construye el centroide como promedio ponderado de embeddings de nodos.

        Los pesos combinan el score topológico y el multiplicador de foco.

        Returns:
            Vector unitario de 384 dimensiones.
        """
        labels  = list(self.cm.nodes.keys())
        weights = np.array([
            self._topo.get(label, 0.1) *
            (self.focus_multiplier if label in self.focus_set else 1.0)
            for label in labels
        ])
        weights /= weights.sum()

        raw_embs = self.engine.embed(labels)   # (N, dim)
        centroid = (raw_embs * weights[:, None]).sum(axis=0)
        n = np.linalg.norm(centroid)
        return centroid / n if n > 0 else centroid

    def _build_aspects(self) -> list[np.ndarray]:
        """Construye aspect vectors a partir de vecindades de conceptos top-rankeados.

        Para cada concepto top: calcula el centroide de su vecindad (label + 4 vecinos)
        y lo agrega como aspect si es suficientemente distinto de los ya existentes.

        Returns:
            Lista de hasta n_aspects vectores unitarios de 384 dimensiones.
        """
        aspects: list[np.ndarray] = [self.centroid]

        if self.n_aspects <= 1:
            return aspects

        # Top concepts by topological weight, focused concepts first
        ranked = sorted(
            self.cm.nodes.keys(),
            key=lambda lbl: self._topo.get(lbl, 0.0) *
                            (self.focus_multiplier if lbl in self.focus_set else 1.0),
            reverse=True,
        )

        le = self.label_embeddings
        seen_vecs: list[np.ndarray] = [self.centroid]

        for label in ranked:
            if len(aspects) >= self.n_aspects:
                break

            # Neighbourhood centroid: label + its direct neighbours
            neighbours = self.cm.nodes[label].neighbors[:4]
            group      = [label] + neighbours
            group_embs = np.vstack([le[g] for g in group if g in le])
            if len(group_embs) == 0:
                continue

            vec = group_embs.mean(axis=0)
            n   = np.linalg.norm(vec)
            vec = vec / n if n > 0 else vec

            # Skip if too similar to an existing aspect
            if any(float(np.dot(vec, sv)) > self.aspect_sim_threshold for sv in seen_vecs):
                continue

            aspects.append(vec)
            seen_vecs.append(vec)

        return aspects


# ===========================================================================
# Funciones auxiliares para caminos raíz→nodo
# ===========================================================================

def build_undirected_adjacency(cm):
    """Adyacencia no dirigida preservando la tripleta original (src, link, dst)."""
    adj = defaultdict(list)
    for src, link, dst in cm.edges:
        adj[src].append((dst, (src, link, dst)))
        adj[dst].append((src, (src, link, dst)))
    return adj


def all_root_paths(cm, max_depth=12, max_paths_per_node=200):
    """
    Enumera todos los caminos simples (sin repetir nodos) desde la raíz del
    mapa hasta cada nodo alcanzable, incluyendo los arcos traversados.

    Returns:
        dict { nodo: [ [(src, link, dst), ...], ... ] }
    """
    root = cm._root or next(iter(cm.nodes))
    adj = build_undirected_adjacency(cm)
    paths_by_node = defaultdict(list)

    def dfs(current, visited, path_edges):
        if current != root:
            if len(paths_by_node[current]) < max_paths_per_node:
                paths_by_node[current].append(list(path_edges))
            else:
                return
        if len(path_edges) >= max_depth:
            return
        for neighbor, triple in adj[current]:
            if neighbor in visited:
                continue
            visited.add(neighbor)
            path_edges.append(triple)
            dfs(neighbor, visited, path_edges)
            path_edges.pop()
            visited.discard(neighbor)

    dfs(root, {root}, [])
    return dict(paths_by_node)


def path_to_text(path_edges):
    """Serializa una lista de tripletas como una única oración sin repetir nodos intermedios."""
    if not path_edges:
        return ""
    parts = [path_edges[0][0]]  # nodo raíz
    for s, l, d in path_edges:
        parts.append(l)
        parts.append(d)
    return " ".join(parts)


def collapse_2cycles(relations):
    """
    Si existen 'A link1 B' y 'B link2 A', conserva solo la primera aparición.
    Evita que ciclos bidireccionales inflen el peso de nodos genéricos (caso Volcanism).
    """
    seen_pairs = set()
    cleaned = []
    for s, l, d in relations:
        if (d, s) in seen_pairs:
            continue
        seen_pairs.add((s, d))
        cleaned.append((s, l, d))
    return cleaned


def all_root_paths_from_relations(root, relations, max_depth=12, max_paths_per_node=200):
    """
    Como all_root_paths(), pero opera sobre una lista de relaciones ya procesada
    (por ejemplo, después de collapse_2cycles) en vez de un ConceptMap.
    """
    adj = defaultdict(list)
    for s, l, d in relations:
        adj[s].append((d, (s, l, d)))
        adj[d].append((s, (s, l, d)))

    paths_by_node = defaultdict(list)

    def dfs(current, visited, path_edges):
        if current != root:
            if len(paths_by_node[current]) < max_paths_per_node:
                paths_by_node[current].append(list(path_edges))
            else:
                return
        if len(path_edges) >= max_depth:
            return
        for neighbor, triple in adj[current]:
            if neighbor in visited:
                continue
            visited.add(neighbor)
            path_edges.append(triple)
            dfs(neighbor, visited, path_edges)
            path_edges.pop()
            visited.discard(neighbor)

    dfs(root, {root}, [])
    return dict(paths_by_node)


# ===========================================================================
# RelationAwareEncoder
# ===========================================================================

class RelationAwareEncoder:
    """
    Representa el mapa conceptual embeddendo tripletas completas "src link dst"
    como oraciones, en vez de nodos aislados. Da a SBERT contexto relacional.

    Soluciona el desalineamiento semántico en mapas donde los términos aislados
    no capturan el dominio (ej: "Gram positive", "prokaryotes" → Infectious Diseases).
    """

    def __init__(self, concept_map, engine, focus_concepts=None,
                 focus_multiplier=3.0, n_aspects=4, aspect_sim_threshold=0.97):
        self.cm                   = concept_map
        self.engine               = engine
        self.focus_set            = set(focus_concepts or [])
        self.focus_multiplier     = focus_multiplier
        self.n_aspects            = n_aspects
        self.aspect_sim_threshold = aspect_sim_threshold
        self._topo                = concept_map.topological_weights()
        self._term_weights = concept_map.term_weights(
            focus_concepts=focus_concepts, focus_multiplier=focus_multiplier)
        self._centroid = None
        self._aspects  = None

    def _triple_texts_and_weights(self):
        """Genera textos y pesos a partir de las tripletas del mapa.

        Cada tripleta "src link dst" produce un texto; su peso es el promedio
        de los pesos topológicos de src y dst, multiplicado si alguno está en foco.
        Si el mapa no tiene arcos, retorna los nodos aislados como fallback.

        Returns:
            Tupla (texts, weights) con listas paralelas de strings y floats.
        """
        texts, weights = [], []
        for src, link, dst in self.cm.edges:
            text = f"{src} {link} {dst}".strip()
            w = (self._topo.get(src, 0.1) + self._topo.get(dst, 0.1)) / 2.0
            if src in self.focus_set or dst in self.focus_set:
                w *= self.focus_multiplier
            texts.append(text)
            weights.append(w)
        if not texts:  # fallback: nodos aislados
            for node in self.cm.nodes:
                texts.append(node)
                weights.append(self._topo.get(node, 0.1))
        return texts, weights

    @property
    def centroid(self):
        """Vector centroide normalizado calculado sobre tripletas (calculado bajo demanda).

        Returns:
            Array unitario de 384 dimensiones.
        """
        if self._centroid is None:
            texts, weights = self._triple_texts_and_weights()
            w = np.array(weights, dtype=float)
            w /= w.sum()
            embs = self.engine.embed(texts)
            vec  = (embs * w[:, None]).sum(axis=0)
            n    = np.linalg.norm(vec)
            self._centroid = vec / n if n > 0 else vec
        return self._centroid

    @property
    def aspects(self):
        """Lista de aspect vectors calculados sobre tripletas (calculados bajo demanda).

        Returns:
            Lista de arrays unitarios de 384 dimensiones.
        """
        if self._aspects is None:
            self._aspects = self._build_aspects()
        return self._aspects

    def _build_aspects(self):
        """Construye aspect vectors seleccionando tripletas de mayor peso que sean suficientemente distintas.

        Returns:
            Lista de hasta n_aspects vectores unitarios.
        """
        texts, weights = self._triple_texts_and_weights()
        w    = np.array(weights, dtype=float)
        embs = self.engine.embed(texts)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embs_n = embs / norms
        aspects = [self.centroid]
        seen    = [self.centroid]
        for i in np.argsort(w)[::-1]:
            if len(aspects) >= self.n_aspects:
                break
            vec = embs_n[i]
            if any(float(np.dot(vec, sv)) > self.aspect_sim_threshold for sv in seen):
                continue
            aspects.append(vec)
            seen.append(vec)
        return aspects

    def summary(self):
        """Retorna un resumen legible del encoder.

        Returns:
            String con título, número de arcos y número de aspects.
        """
        return (f"RelationAwareEncoder: '{self.cm.title}'\n"
                f"  Edges: {len(self.cm.edges)}  Aspects: {len(self.aspects)}")


# ===========================================================================
# PathAwareEncoder
# ===========================================================================

class PathAwareEncoder:
    """
    Representa el mapa mediante TODOS los caminos simples raíz→nodo, incluyendo
    los arcos traversados. Captura el contexto acumulado del camino completo
    (ej: "VirusChar → is a → Microorganism → infects → Host → causes → Disease").
    """

    def __init__(self, concept_map, engine, focus_concepts=None,
                 focus_multiplier=3.0, n_aspects=4,
                 max_depth=12, max_paths_per_node=200,
                 aspect_sim_threshold=0.97):
        self.cm                   = concept_map
        self.engine               = engine
        self.focus_set            = set(focus_concepts or [])
        self.focus_multiplier     = focus_multiplier
        self.n_aspects            = n_aspects
        self.max_depth            = max_depth
        self.max_paths_per_node   = max_paths_per_node
        self.aspect_sim_threshold = aspect_sim_threshold
        self._topo         = concept_map.topological_weights()
        self._term_weights = concept_map.term_weights(
            focus_concepts=focus_concepts, focus_multiplier=focus_multiplier)
        self._paths_by_node = None
        self._centroid = None
        self._aspects  = None

    @property
    def paths_by_node(self):
        """Diccionario de caminos raíz→nodo enumerados (calculado bajo demanda).

        Returns:
            Dict {nodo: [[tripleta, ...], ...]} con todos los caminos simples desde la raíz.
        """
        if self._paths_by_node is None:
            self._paths_by_node = all_root_paths(
                self.cm,
                max_depth=self.max_depth,
                max_paths_per_node=self.max_paths_per_node,
            )
        return self._paths_by_node

    def _path_texts_and_weights(self):
        """Genera textos y pesos a partir de los caminos raíz→nodo.

        El peso de cada camino es el peso topológico del nodo destino dividido
        por el número de caminos que llegan a ese nodo.

        Returns:
            Tupla (texts, weights) con listas paralelas de strings y floats.
        """
        texts, weights = [], []
        for node, paths in self.paths_by_node.items():
            if not paths:
                continue
            w_node = self._topo.get(node, 0.1)
            if node in self.focus_set:
                w_node *= self.focus_multiplier
            w_each = w_node / len(paths)
            for path_edges in paths:
                texts.append(path_to_text(path_edges))
                weights.append(w_each)
        if not texts:  # fallback
            for node in self.cm.nodes:
                texts.append(node)
                weights.append(self._topo.get(node, 0.1))
        return texts, weights

    @property
    def centroid(self):
        """Vector centroide normalizado calculado sobre caminos raíz→nodo (bajo demanda).

        Returns:
            Array unitario de 384 dimensiones.
        """
        if self._centroid is None:
            texts, weights = self._path_texts_and_weights()
            w    = np.array(weights, dtype=float)
            w   /= w.sum()
            embs = self.engine.embed(texts)
            vec  = (embs * w[:, None]).sum(axis=0)
            n    = np.linalg.norm(vec)
            self._centroid = vec / n if n > 0 else vec
        return self._centroid

    @property
    def aspects(self):
        """Lista de aspect vectors construidos sobre caminos raíz→nodo (bajo demanda).

        Returns:
            Lista de arrays unitarios de 384 dimensiones.
        """
        if self._aspects is None:
            self._aspects = self._build_aspects()
        return self._aspects

    def _build_aspects(self):
        """Construye aspect vectors seleccionando caminos de mayor peso que sean suficientemente distintos.

        Returns:
            Lista de hasta n_aspects vectores unitarios.
        """
        texts, weights = self._path_texts_and_weights()
        w    = np.array(weights, dtype=float)
        embs = self.engine.embed(texts)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embs_n = embs / norms
        aspects = [self.centroid]
        seen    = [self.centroid]
        for i in np.argsort(w)[::-1]:
            if len(aspects) >= self.n_aspects:
                break
            vec = embs_n[i]
            if any(float(np.dot(vec, sv)) > self.aspect_sim_threshold for sv in seen):
                continue
            aspects.append(vec)
            seen.append(vec)
        return aspects

    def summary(self):
        """Retorna un resumen legible del PathAwareEncoder.

        Returns:
            String con título, nodos alcanzados, caminos totales y número de aspects.
        """
        n_paths = sum(len(p) for p in self.paths_by_node.values())
        return (f"PathAwareEncoder: '{self.cm.title}'\n"
                f"  Nodos alcanzados: {len(self.paths_by_node)}  "
                f"Caminos totales: {n_paths}  Aspects: {len(self.aspects)}")


# ===========================================================================
# HybridEncoder
# ===========================================================================

def _build_hybrid_texts_and_weights(cm, root_label, topo, focus_set, focus_multiplier,
                                    window_k=3, max_depth=12, max_paths_per_node=200):
    """
    Para cada nodo, elige la representación según profundidad mínima desde la raíz:
        depth == 1 → etiqueta aislada del nodo
        depth <= 3 → tripleta anclada a la raíz: "root :: src link dst"
        depth  > 3 → ventana de camino anclada:  "root :: últimas k tripletas"
    Además aplica collapse_2cycles para evitar inflar nodos genéricos con ciclos.
    """
    clean_edges = collapse_2cycles(cm.edges)
    paths_by_node = all_root_paths_from_relations(
        root_label, clean_edges,
        max_depth=max_depth, max_paths_per_node=max_paths_per_node,
    )
    texts, weights = [], []
    for node, paths in paths_by_node.items():
        if not paths:
            continue
        depth  = min(len(p) for p in paths)
        w_node = topo.get(node, 0.1)
        if node in focus_set:
            w_node *= focus_multiplier
        used_paths = paths
        w_each     = w_node / len(used_paths)
        for path_edges in used_paths:
            if depth <= 1:
                text = node
            elif depth <= 3:
                text = f"{root_label} :: " + path_to_text([path_edges[-1]])
            else:
                window = path_edges[-window_k:]
                text   = f"{root_label} :: " + path_to_text(window)
            texts.append(text)
            weights.append(w_each)
    if not texts:  # fallback
        for node in cm.nodes:
            texts.append(node)
            weights.append(topo.get(node, 0.1))
    return texts, weights


class HybridEncoder:
    """
    Combina las 3 representaciones anteriores:
    - Colapsa ciclos de 2 nodos (collapse_2cycles)
    - Elige por nodo entre etiqueta aislada / tripleta anclada / ventana anclada
      según la profundidad del nodo respecto de la raíz.
    """

    def __init__(self, concept_map, engine, focus_concepts=None,
                 focus_multiplier=3.0, n_aspects=4, window_k=3,
                 max_depth=12, max_paths_per_node=200,
                 aspect_sim_threshold=0.97):
        self.cm                      = concept_map
        self.engine                  = engine
        self.focus_set               = set(focus_concepts or [])
        self.focus_multiplier        = focus_multiplier
        self.n_aspects               = n_aspects
        self.window_k                = window_k
        self.max_depth               = max_depth
        self.max_paths_per_node      = max_paths_per_node
        self.aspect_sim_threshold    = aspect_sim_threshold
        self._topo         = concept_map.topological_weights()
        self._term_weights = concept_map.term_weights(
            focus_concepts=focus_concepts, focus_multiplier=focus_multiplier)
        self._root_label    = concept_map._root or next(iter(concept_map.nodes))
        self._texts_weights = None
        self._centroid      = None
        self._aspects       = None

    def _get_texts_weights(self):
        """Obtiene (con caché) los textos y pesos del encoder híbrido.

        Returns:
            Tupla (texts, weights) calculada por _build_hybrid_texts_and_weights().
        """
        if self._texts_weights is None:
            self._texts_weights = _build_hybrid_texts_and_weights(
                self.cm, self._root_label, self._topo,
                self.focus_set, self.focus_multiplier,
                window_k=self.window_k,
                max_depth=self.max_depth,
                max_paths_per_node=self.max_paths_per_node,
            )
        return self._texts_weights

    @property
    def centroid(self):
        """Vector centroide normalizado del encoder híbrido (bajo demanda).

        Returns:
            Array unitario de 384 dimensiones.
        """
        if self._centroid is None:
            texts, weights = self._get_texts_weights()
            w    = np.array(weights, dtype=float)
            w   /= w.sum()
            embs = self.engine.embed(texts)
            vec  = (embs * w[:, None]).sum(axis=0)
            n    = np.linalg.norm(vec)
            self._centroid = vec / n if n > 0 else vec
        return self._centroid

    @property
    def aspects(self):
        """Lista de aspect vectors del encoder híbrido (bajo demanda).

        Returns:
            Lista de arrays unitarios de 384 dimensiones.
        """
        if self._aspects is None:
            self._aspects = self._build_aspects()
        return self._aspects

    def _build_aspects(self):
        """Construye aspect vectors del encoder híbrido seleccionando textos de mayor peso distintos.

        Returns:
            Lista de hasta n_aspects vectores unitarios.
        """
        texts, weights = self._get_texts_weights()
        w    = np.array(weights, dtype=float)
        embs = self.engine.embed(texts)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embs_n = embs / norms
        aspects = [self.centroid]
        seen    = [self.centroid]
        for i in np.argsort(w)[::-1]:
            if len(aspects) >= self.n_aspects:
                break
            vec = embs_n[i]
            if any(float(np.dot(vec, sv)) > self.aspect_sim_threshold for sv in seen):
                continue
            aspects.append(vec)
            seen.append(vec)
        return aspects

    def summary(self):
        """Retorna un resumen legible del HybridEncoder.

        Returns:
            String con título, número de unidades de texto y número de aspects.
        """
        texts, _ = self._get_texts_weights()
        return (f"HybridEncoder: '{self.cm.title}'\n"
                f"  Unidades de texto: {len(texts)}  Aspects: {len(self.aspects)}")
