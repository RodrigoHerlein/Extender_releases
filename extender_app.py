"""
extender_app.py
===============
Servidor Flask para la interfaz EXTENDER.
Inicio: python extender_app.py
UI:     http://localhost:5000
"""

import itertools
import json
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter, deque

# ── Auto-instalar dependencias pesadas si faltan ───────────────────────────────
def _pip(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

def _ensure_deps():
    missing = []
    for mod, pkg in [("flask","flask"), ("sentence_transformers","sentence-transformers"),
                     ("umap","umap-learn"), ("hdbscan","hdbscan"), ("bertopic","bertopic")]:
        try: __import__(mod)
        except ImportError: missing.append(pkg)
    if missing:
        print(f"Instalando dependencias faltantes: {missing}")
        _pip(*missing)
        print("Instalación completa.")

_ensure_deps()

import numpy as np
from flask import Flask, jsonify, request, send_from_directory, Response

# ── Log buffer + SSE broadcast ────────────────────────────────────────────────
_log_buffer      = deque(maxlen=3000)   # circular buffer of log lines
_log_subscribers = []                   # list of queue.Queue, one per SSE client
_log_lock        = threading.Lock()

def _log(msg: str):
    """Append a line to the log buffer and push it to all SSE subscribers."""
    for line in str(msg).splitlines():
        with _log_lock:
            _log_buffer.append(line)
            for q in _log_subscribers[:]:
                try:
                    q.put_nowait(line)
                except Exception:
                    pass

class _TeeStream:
    """Wrap stdout/stderr so every write also goes to _log()."""
    def __init__(self, original):
        self._orig = original
        self._buf  = ""

    def write(self, s):
        self._orig.write(s)
        self._buf += s
        # Flush on newline so partial lines accumulate nicely
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line:
                _log(line)

    def flush(self):
        if self._buf:
            _log(self._buf)
            self._buf = ""
        self._orig.flush()

    def __getattr__(self, name):
        return getattr(self._orig, name)

# Redirect stdout/stderr globally so ALL print() — including background threads — go to _log
sys.stdout = _TeeStream(sys.stdout)
sys.stderr = _TeeStream(sys.stderr)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
CODIGO_DIR        = BASE_DIR  # app now lives inside Pipeline_EXTENDER
CODIGO_PAPER_DIR  = os.path.join(BASE_DIR, "..", "Experimento_Paper")
DATASETS_DIR      = os.path.join(CODIGO_DIR, "Datasets")
PAPER_DATASET_DIR        = os.path.join(DATASETS_DIR, "dataset_paper", "Dataset_paper_original")
WIKIPEDIA_DIR            = os.path.join(DATASETS_DIR, "dataset_paper", "Wikipedia")
DATAVIZ_WIKIPEDIA_DIR    = os.path.join(DATASETS_DIR, "dataset_dataviz", "Wikipedia")
CG_WIKIPEDIA_DIR         = os.path.join(DATASETS_DIR, "dataset_computacion_grafica", "Wikipedia")
OOP_WIKIPEDIA_DIR        = os.path.join(DATASETS_DIR, "dataset_oop", "Wikipedia")
VINOS_WIKIPEDIA_DIR      = os.path.join(DATASETS_DIR, "dataset_vinos", "Wikipedia")
AUTOS_WIKIPEDIA_DIR      = os.path.join(DATASETS_DIR, "dataset_autos", "Wikipedia")
BEST_CONFIGS_PATH = os.path.join(DATASETS_DIR, "Datos_y_Configuracion", "best_configs.json")
CACHE_DIR         = os.path.join(CODIGO_DIR, ".cache")
SAVED_MAPS_DIR    = os.path.join(DATASETS_DIR, "Datos_y_Configuracion", "saved_maps_and_configuration")
UI_DIR            = os.path.join(CODIGO_DIR, "Visualizaciones e interfaz", "Interfaz_Web")
os.makedirs(SAVED_MAPS_DIR, exist_ok=True)


for _p in [CODIGO_DIR, CODIGO_PAPER_DIR]:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=BASE_DIR)

# ── Estado global ──────────────────────────────────────────────────────────────
state = {
    "ready":             False,
    "loading_msg":       "Iniciando...",
    "corpus":            None,
    "corpus_type":       "paper",    # "paper" | "wikipedia" | ...
    "corpus_switching":  False,      # True while a switch is in progress
    "engine":            None,
    "error":             None,
    "docs_per_category": {},
}

# ── Carga del backend en background ──────────────────────────────────────────
def _load_backend():
    try:
        state["loading_msg"] = "Importando módulos..."
        from extender_bertopic import EmbeddingEngine

        state["loading_msg"] = "Cargando modelo SBERT (primera vez descarga ~90 MB)..."
        engine = EmbeddingEngine()
        engine.fit()
        state["engine"] = engine

        os.makedirs(CACHE_DIR, exist_ok=True)

        # Cargar el corpus por defecto (paper) si está disponible
        if os.path.isdir(PAPER_DATASET_DIR):
            from corpus_loader import PaperDatasetCorpus
            state["loading_msg"] = "Cargando dataset del paper..."
            corpus = PaperDatasetCorpus(data_dir=PAPER_DATASET_DIR,
                                        max_docs_per_topic=150, verbose=False)
            state["corpus"]            = corpus
            state["corpus_type"]       = "paper"
            state["docs_per_category"] = Counter(e.category for e in corpus.entries)
            state["loading_msg"]       = f"Listo — {len(corpus.entries):,} documentos"
            # Disparar topic study en background para que los halos estén listos
            threading.Thread(target=_run_topic_study, args=("paper",), daemon=True).start()
        else:
            state["loading_msg"] = "Listo — sin corpus cargado (seleccioná uno desde la UI)"

        state["ready"] = True

    except Exception:
        state["error"]       = traceback.format_exc()
        state["loading_msg"] = "Error al cargar — ver consola"
        print(state["error"])


threading.Thread(target=_load_backend, daemon=True).start()


def _switch_corpus(corpus_type: str):
    """Swap the active corpus without reloading SBERT. Runs in a background thread."""
    if corpus_type == state["corpus_type"] and state["corpus"] is not None:
        return
    state["corpus_switching"] = True
    state["loading_msg"] = f"Cambiando corpus a '{corpus_type}'..."
    try:
        if corpus_type == "paper":
            if not os.path.isdir(PAPER_DATASET_DIR):
                raise FileNotFoundError(f"Dataset del paper no encontrado: {PAPER_DATASET_DIR}")
            from corpus_loader import PaperDatasetCorpus
            state["loading_msg"] = "Cargando dataset del paper..."
            corpus = PaperDatasetCorpus(data_dir=PAPER_DATASET_DIR,
                                        max_docs_per_topic=150, verbose=False)
        elif corpus_type == "wikipedia":
            articles_path = os.path.join(WIKIPEDIA_DIR, "articles.jsonl")
            if not os.path.isfile(articles_path):
                raise FileNotFoundError(
                    f"No se encontró articles.jsonl en {WIKIPEDIA_DIR}.\n"
                    "Corré wikipedia_crawler.py primero para descargar el dataset."
                )
            import sys as _sys
            _wiki_corpus_dir = WIKIPEDIA_DIR
            if _wiki_corpus_dir not in _sys.path:
                _sys.path.insert(0, _wiki_corpus_dir)
            from corpus_loader import WikipediaCorpus
            state["loading_msg"] = "Cargando corpus Wikipedia..."
            corpus = WikipediaCorpus(data_dir=WIKIPEDIA_DIR, include_keywords=None, verbose=False)
        elif corpus_type == "dataviz_wikipedia":
            articles_path = os.path.join(DATAVIZ_WIKIPEDIA_DIR, "articles.jsonl")
            if not os.path.isfile(articles_path):
                raise FileNotFoundError(
                    f"No se encontró articles.jsonl en {DATAVIZ_WIKIPEDIA_DIR}.\n"
                    "Corré wikipedia_crawler.py del dataset dataviz primero."
                )
            import sys as _sys
            _dataviz_corpus_dir = DATAVIZ_WIKIPEDIA_DIR
            if _dataviz_corpus_dir not in _sys.path:
                _sys.path.insert(0, _dataviz_corpus_dir)
            from corpus_loader import WikipediaCorpus
            state["loading_msg"] = "Cargando corpus Wikipedia — Visualización de Datos..."
            corpus = WikipediaCorpus(data_dir=DATAVIZ_WIKIPEDIA_DIR, include_keywords=None, verbose=False)
        elif corpus_type == "cg_wikipedia":
            articles_path = os.path.join(CG_WIKIPEDIA_DIR, "articles.jsonl")
            if not os.path.isfile(articles_path):
                raise FileNotFoundError(
                    f"No se encontró articles.jsonl en {CG_WIKIPEDIA_DIR}.\n"
                    "Corré wikipedia_crawler.py del dataset computacion_grafica primero."
                )
            import sys as _sys
            if CG_WIKIPEDIA_DIR not in _sys.path:
                _sys.path.insert(0, CG_WIKIPEDIA_DIR)
            from corpus_loader import WikipediaCorpus
            state["loading_msg"] = "Cargando corpus Wikipedia — Computación Gráfica..."
            corpus = WikipediaCorpus(data_dir=CG_WIKIPEDIA_DIR, include_keywords=None, verbose=False)
        elif corpus_type == "oop_wikipedia":
            articles_path = os.path.join(OOP_WIKIPEDIA_DIR, "articles.jsonl")
            if not os.path.isfile(articles_path):
                raise FileNotFoundError(
                    f"No se encontró articles.jsonl en {OOP_WIKIPEDIA_DIR}.\n"
                    "Corré wikipedia_crawler.py del dataset oop primero."
                )
            import sys as _sys
            if OOP_WIKIPEDIA_DIR not in _sys.path:
                _sys.path.insert(0, OOP_WIKIPEDIA_DIR)
            from corpus_loader import WikipediaCorpus
            state["loading_msg"] = "Cargando corpus Wikipedia — Programación Orientada a Objetos..."
            corpus = WikipediaCorpus(data_dir=OOP_WIKIPEDIA_DIR, include_keywords=None, verbose=False)
        elif corpus_type == "vinos_wikipedia":
            articles_path = os.path.join(VINOS_WIKIPEDIA_DIR, "articles.jsonl")
            if not os.path.isfile(articles_path):
                raise FileNotFoundError(
                    f"No se encontró articles.jsonl en {VINOS_WIKIPEDIA_DIR}.\n"
                    "Corré wikipedia_crawler.py del dataset vinos primero."
                )
            import sys as _sys
            if VINOS_WIKIPEDIA_DIR not in _sys.path:
                _sys.path.insert(0, VINOS_WIKIPEDIA_DIR)
            from corpus_loader import WikipediaCorpus
            state["loading_msg"] = "Cargando corpus Wikipedia — Vinos..."
            corpus = WikipediaCorpus(data_dir=VINOS_WIKIPEDIA_DIR, include_keywords=None, verbose=False)
        elif corpus_type == "autos_wikipedia":
            articles_path = os.path.join(AUTOS_WIKIPEDIA_DIR, "articles.jsonl")
            if not os.path.isfile(articles_path):
                raise FileNotFoundError(
                    f"No se encontró articles.jsonl en {AUTOS_WIKIPEDIA_DIR}.\n"
                    "Corré wikipedia_crawler.py del dataset autos primero."
                )
            import sys as _sys
            if AUTOS_WIKIPEDIA_DIR not in _sys.path:
                _sys.path.insert(0, AUTOS_WIKIPEDIA_DIR)
            from corpus_loader import WikipediaCorpus
            state["loading_msg"] = "Cargando corpus Wikipedia — Autos..."
            corpus = WikipediaCorpus(data_dir=AUTOS_WIKIPEDIA_DIR, include_keywords=None, verbose=False)
        else:
            raise ValueError(f"corpus_type desconocido: {corpus_type!r}")

        state["corpus"]            = corpus
        state["corpus_type"]       = corpus_type
        state["docs_per_category"] = Counter(e.category for e in corpus.entries)
        state["loading_msg"]       = f"Listo — {len(corpus.entries):,} documentos ({corpus_type})"
        print(state["loading_msg"])
        # Trigger topic study in background (only if not already done/running)
        _TOPIC_STUDY_CORPUS_TYPES = (
            "paper", "wikipedia", "dataviz_wikipedia",
            "cg_wikipedia", "oop_wikipedia", "vinos_wikipedia", "autos_wikipedia",
        )
        if corpus_type in _TOPIC_STUDY_CORPUS_TYPES:
            with _topic_study_lock:
                already = _topic_study_state.get(corpus_type)
            if already is None:
                threading.Thread(target=_run_topic_study, args=(corpus_type,), daemon=True).start()
    except Exception:
        err = traceback.format_exc()
        state["error"]       = err
        state["loading_msg"] = f"Error al cambiar corpus — ver consola"
        print(err)
    finally:
        state["corpus_switching"] = False


# ── Utilidades ─────────────────────────────────────────────────────────────────
def _load_best_configs():
    if os.path.exists(BEST_CONFIGS_PATH):
        with open(BEST_CONFIGS_PATH) as f:
            return json.load(f)
    return {}


def _compute_default_params(best_configs):
    """Valor más frecuente de cada parámetro entre todos los mapas/encoders conocidos."""
    counters = {}
    for map_data in best_configs.values():
        for enc_data in (map_data or {}).values():
            params = (enc_data or {}).get("params", {})
            for k, v in params.items():
                if k == "encoder":
                    continue
                counters.setdefault(k, Counter())[v] += 1
    return {k: c.most_common(1)[0][0] for k, c in counters.items()}


def _build_concept_map(cm_data):
    from extender_bertopic import ConceptMap
    cm = ConceptMap(title=cm_data.get("title", "Custom"))
    for c in cm_data.get("concepts", []):
        cm.add_concept(c)
    for r in cm_data.get("relations", []):
        if isinstance(r, (list, tuple)):
            cm.add_relation(str(r[0]), str(r[1]), str(r[2]))
        else:
            cm.add_relation(r["src"], r["link"], r["dst"])
    if cm_data.get("root"):
        cm.set_root(cm_data["root"])
    return cm


# ── Rutas ──────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    resp = send_from_directory(UI_DIR, "extender_ui.html")
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    return resp


@app.route("/api/status")
def api_status():
    corpus = state["corpus"]
    return jsonify({
        "ready":            state["ready"],
        "msg":              state["loading_msg"],
        "error":            state["error"],
        "corpus_type":      state["corpus_type"],
        "corpus_switching": state["corpus_switching"],
        "corpus_n_docs":    len(corpus.entries) if corpus else 0,
        "paper_available":              os.path.isdir(PAPER_DATASET_DIR),
        "wikipedia_available":          os.path.isfile(os.path.join(WIKIPEDIA_DIR, "articles.jsonl")),
        "wikipedia_n_docs":             sum(1 for _ in open(os.path.join(WIKIPEDIA_DIR, "articles.jsonl"), encoding="utf-8")) if os.path.isfile(os.path.join(WIKIPEDIA_DIR, "articles.jsonl")) else 0,
        "dataviz_wikipedia_available":  os.path.isfile(os.path.join(DATAVIZ_WIKIPEDIA_DIR, "articles.jsonl")),
        "dataviz_wikipedia_n_docs":     sum(1 for _ in open(os.path.join(DATAVIZ_WIKIPEDIA_DIR, "articles.jsonl"), encoding="utf-8")) if os.path.isfile(os.path.join(DATAVIZ_WIKIPEDIA_DIR, "articles.jsonl")) else 0,
        "cg_wikipedia_available":       os.path.isfile(os.path.join(CG_WIKIPEDIA_DIR, "articles.jsonl")),
        "cg_wikipedia_n_docs":          sum(1 for _ in open(os.path.join(CG_WIKIPEDIA_DIR, "articles.jsonl"), encoding="utf-8")) if os.path.isfile(os.path.join(CG_WIKIPEDIA_DIR, "articles.jsonl")) else 0,
        "oop_wikipedia_available":      os.path.isfile(os.path.join(OOP_WIKIPEDIA_DIR, "articles.jsonl")),
        "oop_wikipedia_n_docs":         sum(1 for _ in open(os.path.join(OOP_WIKIPEDIA_DIR, "articles.jsonl"), encoding="utf-8")) if os.path.isfile(os.path.join(OOP_WIKIPEDIA_DIR, "articles.jsonl")) else 0,
        "vinos_wikipedia_available":    os.path.isfile(os.path.join(VINOS_WIKIPEDIA_DIR, "articles.jsonl")),
        "vinos_wikipedia_n_docs":       sum(1 for _ in open(os.path.join(VINOS_WIKIPEDIA_DIR, "articles.jsonl"), encoding="utf-8")) if os.path.isfile(os.path.join(VINOS_WIKIPEDIA_DIR, "articles.jsonl")) else 0,
        "autos_wikipedia_available":    os.path.isfile(os.path.join(AUTOS_WIKIPEDIA_DIR, "articles.jsonl")),
        "autos_wikipedia_n_docs":       sum(1 for _ in open(os.path.join(AUTOS_WIKIPEDIA_DIR, "articles.jsonl"), encoding="utf-8")) if os.path.isfile(os.path.join(AUTOS_WIKIPEDIA_DIR, "articles.jsonl")) else 0,
    })


@app.route("/api/set_corpus", methods=["POST"])
def api_set_corpus():
    if state["corpus_switching"]:
        return jsonify({"error": "Ya hay un cambio de corpus en progreso"}), 409
    corpus_type = request.json.get("corpus_type", "paper")
    _VALID_CORPUS_TYPES = (
        "paper", "wikipedia", "dataviz_wikipedia",
        "cg_wikipedia", "oop_wikipedia", "vinos_wikipedia", "autos_wikipedia",
    )
    if corpus_type not in _VALID_CORPUS_TYPES:
        return jsonify({"error": f"corpus_type desconocido: '{corpus_type}'"}), 400
    threading.Thread(target=_switch_corpus, args=(corpus_type,), daemon=True).start()
    return jsonify({"ok": True, "msg": f"Cambiando a corpus '{corpus_type}'..."})


@app.route("/api/maps")
def api_maps():
    """Mapas conocidos con su definición, mejor config por encoder y defaults globales.
    Este endpoint es completamente independiente del backend ML — funciona
    aunque corpus/SBERT todavía estén cargando o hayan fallado."""
    try:
        from concept_maps import CONCEPT_MAPS_JSON, MAP_TO_TOPIC
    except Exception as e:
        return jsonify({"error": f"No se pudo importar concept_maps: {e}", "maps": {}, "default_params": {}}), 500

    best = _load_best_configs()
    default_params = _compute_default_params(best)

    maps = {}
    for name, spec in CONCEPT_MAPS_JSON.items():
        relations = [
            {"src": str(s), "link": str(l), "dst": str(d)}
            for s, l, d in spec["relations"]
        ]
        maps[name] = {
            "title":        spec["title"],
            "root":         spec.get("root", ""),
            "relations":    relations,
            "odp_path":     MAP_TO_TOPIC[name]["odp_path"],
            "best_configs": best.get(name, {}),
        }

    return jsonify({"maps": maps, "default_params": default_params})


@app.route("/encoder_dashboard")
def encoder_dashboard():
    return send_from_directory(UI_DIR, "encoder_dashboard.html")


@app.route("/api/concept_maps_dashboard")
def api_concept_maps_dashboard():
    """Mapas en formato edges [[src,link,dst]] + métricas desde best_configs.json.
    Usado por encoder_dashboard.html para evitar datos hardcodeados."""
    try:
        from concept_maps import CONCEPT_MAPS_JSON
    except Exception as e:
        return jsonify({"error": f"No se pudo importar concept_maps: {e}"}), 500

    # Mapas en formato edges
    concept_maps = {}
    for name, spec in CONCEPT_MAPS_JSON.items():
        edges = [[str(s), str(l), str(d)] for s, l, d in spec["relations"]]
        concept_maps[name] = {
            "root":  spec.get("root", ""),
            "edges": edges,
        }

    # Métricas desde best_configs.json
    best = _load_best_configs()
    metrics = []
    for map_name, enc_data in best.items():
        for enc_key, enc_info in (enc_data or {}).items():
            m = (enc_info or {}).get("metrics", {})
            metrics.append({
                "map":      map_name,
                "enc":      enc_key,
                "p10":      m.get("p10"),
                "rec_ret":  m.get("rec_ret"),
                "prec_fin": m.get("prec_fin"),
                "rec_fin":  m.get("rec_fin"),
                "n_cand":   m.get("n_cand"),
                "n_fin":    m.get("n_fin"),
            })

    return jsonify({"concept_maps": concept_maps, "metrics": metrics})


@app.route("/api/logs/stream")
def api_logs_stream():
    """Server-Sent Events stream: envía el buffer histórico y luego mensajes en tiempo real."""
    def generate():
        # ① Enviar buffer histórico acumulado hasta ahora
        with _log_lock:
            history = list(_log_buffer)
            q = queue.Queue()
            _log_subscribers.append(q)
        for line in history:
            yield f"data: {line}\n\n"

        # ② Streaming en tiempo real
        try:
            while True:
                try:
                    line = q.get(timeout=25)
                    yield f"data: {line}\n\n"
                except queue.Empty:
                    yield ": keepalive\n\n"   # SSE comment — evita timeout del proxy
        finally:
            with _log_lock:
                try:
                    _log_subscribers.remove(q)
                except ValueError:
                    pass

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",   # desactiva buffer de Nginx si hay proxy
        },
    )


@app.route("/api/logs/clear", methods=["POST"])
def api_logs_clear():
    with _log_lock:
        _log_buffer.clear()
    return jsonify({"ok": True})


@app.route("/api/preview_repr", methods=["POST"])
def api_preview_repr():
    """Devuelve los textos que serían embeddizados por el encoder elegido, sin correr SBERT."""
    data        = request.get_json(force=True)
    cm_data     = data.get("concept_map", {})
    encoder_key = data.get("encoder", "node_label")
    params      = data.get("params", {})

    try:
        from concept_map_encoder import (
            RelationAwareEncoder, PathAwareEncoder,
            all_root_paths, path_to_text,
            collapse_2cycles, all_root_paths_from_relations,
        )

        cm = _build_concept_map(cm_data)
        if not cm.nodes:
            return jsonify({"error": "El mapa está vacío."}), 400

        topo = cm.topological_weights()
        # BFS desde la raíz para profundidad real de cada nodo
        from collections import deque
        root = cm._root or next(iter(cm.nodes))
        bfs_depth = {root: 0}
        bfs_order = {root: 0}
        queue = deque([root])
        bfs_seq = 0
        while queue:
            cur = queue.popleft()
            for neighbor in (cm.nodes[cur].neighbors if cur in cm.nodes else []):
                if neighbor not in bfs_depth:
                    bfs_depth[neighbor] = bfs_depth[cur] + 1
                    bfs_seq += 1
                    bfs_order[neighbor] = bfs_seq
                    queue.append(neighbor)

        def _meta(node, depth_override=None):
            d = depth_override if depth_override is not None else bfs_depth.get(node, 999)
            return {
                "depth":      d,
                "bfs_order":  bfs_order.get(node, 999),
                "topo_weight": round(float(topo.get(node, 0.0)), 4),
            }

        items = []  # [{node, texts:[str], depth, bfs_order, topo_weight}]

        if encoder_key == "node_label":
            for node in cm.nodes:
                items.append({"node": node, "texts": [node], **_meta(node)})

        elif encoder_key == "relation_aware":
            for src, link, dst in cm.edges:
                text = f"{src} {link} {dst}".strip()
                d = max(bfs_depth.get(src, 0), bfs_depth.get(dst, 0))
                tw = (topo.get(src, 0.0) + topo.get(dst, 0.0)) / 2.0
                items.append({"node": f"{src} → {dst}", "texts": [text],
                               "depth": d, "bfs_order": bfs_order.get(dst, 999),
                               "topo_weight": round(float(tw), 4)})
            if not items:
                for node in cm.nodes:
                    items.append({"node": node, "texts": [node], **_meta(node)})

        elif encoder_key == "path_aware":
            max_depth = int(params.get("path_max_depth", 12))
            max_paths = int(params.get("path_max_paths_per_node", 200))
            paths_by_node = all_root_paths(cm, max_depth=max_depth, max_paths_per_node=max_paths)
            for node, paths in paths_by_node.items():
                texts = [path_to_text(p) for p in paths if p]
                if texts:
                    depth = min(len(p) for p in paths) if paths else bfs_depth.get(node, 0)
                    items.append({"node": node, "texts": texts, **_meta(node, depth)})
            if not items:
                for node in cm.nodes:
                    items.append({"node": node, "texts": [node], **_meta(node)})

        elif encoder_key == "hybrid":
            max_depth  = int(params.get("path_max_depth", 12))
            max_paths  = int(params.get("path_max_paths_per_node", 200))
            window_k   = int(params.get("hybrid_window_k", 3))
            max_used   = int(params.get("hybrid_max_paths_used_per_node", 3))
            root_label = cm._root or next(iter(cm.nodes))
            clean_edges = collapse_2cycles(cm.edges)
            paths_by_node = all_root_paths_from_relations(
                root_label, clean_edges,
                max_depth=max_depth, max_paths_per_node=max_paths,
            )
            for node, paths in paths_by_node.items():
                if not paths:
                    continue
                depth = min(len(p) for p in paths)
                used  = paths[:max_used]
                texts = []
                for path_edges in used:
                    if depth <= 1:
                        text = node
                    elif depth <= 3:
                        text = f"{root_label} :: " + path_to_text([path_edges[-1]])
                    else:
                        window = path_edges[-window_k:]
                        text   = f"{root_label} :: " + path_to_text(window)
                    texts.append(text)
                if texts:
                    items.append({"node": node, "texts": texts, **_meta(node, depth)})
            if not items:
                for node in cm.nodes:
                    items.append({"node": node, "texts": [node], **_meta(node)})

        total_texts = sum(len(it["texts"]) for it in items)
        return jsonify({"encoder": encoder_key, "items": items, "total": total_texts})

    except Exception:
        import traceback
        return jsonify({"error": traceback.format_exc()}), 500


@app.route("/api/browse_folder", methods=["GET"])
def api_browse_folder():
    """Abre un diálogo nativo de selección de carpeta y devuelve la ruta elegida."""
    try:
        import tkinter as _tk
        import tkinter.filedialog as _fd
        root = _tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        folder = _fd.askdirectory(title="Elegí la carpeta de destino para los timelines")
        root.destroy()
        if folder:
            return jsonify({"folder": folder})
        return jsonify({"folder": None})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/save_timeline", methods=["POST"])
def api_save_timeline():
    """
    Guarda el JSON del timeline en disco.
    Body: { folder_path: str, map_name: str, data: object }
    Escribe: {folder_path}/{map_name}_timeline.json
    """
    try:
        body       = request.get_json(force=True) or {}
        folder     = body.get("folder_path", "").strip()
        map_name   = body.get("map_name", "timeline").strip() or "timeline"
        data       = body.get("data")

        if not folder:
            return jsonify({"ok": False, "error": "folder_path vacío"}), 400
        if not os.path.isdir(folder):
            return jsonify({"ok": False, "error": f"Carpeta no encontrada: {folder}"}), 400

        # Nombre de archivo seguro
        safe_name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in map_name)
        filename  = "timeline_" + safe_name + ".json"
        filepath  = os.path.join(folder, filename)

        import json as _json_st
        with open(filepath, "w", encoding="utf-8") as fh:
            _json_st.dump(data, fh, ensure_ascii=False, indent=2)

        print(f"[save_timeline] Guardado: {filepath}")
        return jsonify({"ok": True, "path": filepath, "filename": filename})
    except Exception as e:
        import traceback as _tb_st
        print(f"[save_timeline] Error: {e}")
        _tb_st.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/export_repr_embeddings", methods=["POST"])
def api_export_repr_embeddings():
    """
    Construye el encoder real (con SBERT) y devuelve:
      - texts: lista de {text, node, weight, embedding}
      - aspects: lista de {index, embedding}  (aspect 0 = centroide)
      - map_vec: {text, embedding}
    Todo como JSON descargable.
    """
    if not state["ready"]:
        return jsonify({"error": "Backend todavía cargando."}), 503

    data        = request.get_json(force=True)
    cm_data     = data.get("concept_map", {})
    encoder_key = data.get("encoder", "node_label")
    params      = data.get("params", {})

    try:
        import numpy as np
        from concept_map_encoder import (
            ConceptMapEncoder, RelationAwareEncoder,
            PathAwareEncoder, HybridEncoder,
        )

        engine = state["engine"]
        cm     = _build_concept_map(cm_data)
        if not cm.nodes:
            return jsonify({"error": "El mapa está vacío."}), 400

        n_aspects  = int(params.get("n_aspects", 4))
        focus_mult = float(params.get("focus_multiplier", 3.0))

        enc_kwargs = dict(
            concept_map=cm, engine=engine,
            focus_concepts=[], focus_multiplier=focus_mult,
            n_aspects=n_aspects,
        )
        if encoder_key == "node_label":
            encoder = ConceptMapEncoder(**enc_kwargs)
        elif encoder_key == "relation_aware":
            encoder = RelationAwareEncoder(**enc_kwargs)
        elif encoder_key == "path_aware":
            encoder = PathAwareEncoder(
                **enc_kwargs,
                max_depth=int(params.get("path_max_depth", 12)),
                max_paths_per_node=int(params.get("path_max_paths_per_node", 200)),
            )
        elif encoder_key == "hybrid":
            encoder = HybridEncoder(
                **enc_kwargs,
                window_k=int(params.get("hybrid_window_k", 3)),
                max_depth=int(params.get("path_max_depth", 12)),
                max_paths_per_node=int(params.get("path_max_paths_per_node", 200)),
            )
        else:
            return jsonify({"error": f"Encoder desconocido: {encoder_key}"}), 400

        # Textos + pesos + embeddings (método varía por encoder)
        if encoder_key == "node_label":
            topo = cm.topological_weights()
            focus_mult_val = focus_mult
            texts_raw   = list(cm.nodes.keys())
            weights_raw = [
                topo.get(lbl, 0.1) * (focus_mult_val if lbl in [] else 1.0)
                for lbl in texts_raw
            ]
        elif encoder_key == "relation_aware":
            texts_raw, weights_raw = encoder._triple_texts_and_weights()
        elif encoder_key == "path_aware":
            texts_raw, weights_raw = encoder._path_texts_and_weights()
        else:
            texts_raw, weights_raw = encoder._get_texts_weights()

        embs = engine.embed(texts_raw)
        w_arr = np.array(weights_raw, dtype=float)
        texts_out = [
            {
                "index":     i,
                "text":      texts_raw[i],
                "weight":    round(float(w_arr[i]), 6),
                "embedding": embs[i].tolist(),
            }
            for i in range(len(texts_raw))
        ]

        # Aspects (aspect 0 = centroide)
        aspects_out = [
            {"index": i, "embedding": v.tolist()}
            for i, v in enumerate(encoder.aspects)
        ]

        # map_vec: top-30 términos por peso topológico concatenados y embeddizados
        map_text = " ".join(
            t for t, _ in sorted(
                cm.term_weights(focus_concepts=[], focus_multiplier=focus_mult).items(),
                key=lambda x: -x[1],
            )[:30]
        )
        map_vec_raw = engine.embed([map_text])[0]
        n = float(np.linalg.norm(map_vec_raw))
        map_vec_norm = (map_vec_raw / n).tolist() if n > 0 else map_vec_raw.tolist()

        payload = {
            "encoder":    encoder_key,
            "map_title":  cm.title,
            "timestamp":  __import__("datetime").datetime.now().isoformat(),
            "n_texts":    len(texts_out),
            "n_aspects":  len(aspects_out),
            "texts":      texts_out,
            "aspects":    aspects_out,
            "map_vec": {
                "text":      map_text,
                "embedding": map_vec_norm,
            },
        }
        return jsonify(payload)

    except Exception:
        import traceback
        return jsonify({"error": traceback.format_exc()}), 500


@app.route("/api/run", methods=["POST"])
def api_run():
    if not state["ready"]:
        return jsonify({"error": "Backend todavía cargando, esperá unos segundos."}), 503

    data          = request.get_json(force=True)
    cm_data       = data.get("concept_map", {})
    encoder_key   = data.get("encoder", "node_label")
    params        = data.get("params", {})
    known_map     = data.get("known_map_name")   # None → mapa nuevo

    try:
        from concept_maps import MAP_TO_TOPIC
        from concept_map_encoder import ConceptMapEncoder, RelationAwareEncoder, PathAwareEncoder, HybridEncoder
        from tuner import make_tuned_config, precision_recall_for_map
        ENCODER_CLASSES = {
            "node_label":     ConceptMapEncoder,
            "relation_aware": RelationAwareEncoder,
            "path_aware":     PathAwareEncoder,
            "hybrid":         HybridEncoder,
        }

        corpus = state["corpus"]

        # ── Concept map ──────────────────────────────────────────────────────
        cm = _build_concept_map(cm_data)
        if not cm.nodes:
            return jsonify({"error": "El mapa conceptual está vacío."}), 400

        # ── ODP path y keywords de dominio ──────────────────────────────────
        odp_path = ""
        if known_map and known_map in MAP_TO_TOPIC:
            odp_path = MAP_TO_TOPIC[known_map]["odp_path"]
        else:
            # Para mapas nuevos usamos el título como path ficticio
            title_words = cm.title.replace("-", "/").replace(" ", "/")
            odp_path    = f"Top/{title_words}" if title_words else "Top/Unknown"

        # domain_category_keywords: override desde UI si viene en params
        domain_kw = params.pop("domain_category_keywords", None)

        # Limpiar params — filtrar claves no-ExplorerConfig
        cfg_keys = {
            "top_k_per_aspect", "min_aspect_sim", "min_doc_relevance",
            "min_cluster_size", "min_samples", "umap_n_components", "umap_n_neighbors",
            "random_state", "focus_weight", "specificity_weight", "quality_weight",
            "min_nr_topics", "max_nr_topics",
            "domain_category_min_fraction", "domain_category_penalty",
            "exclude_category_max_fraction",
            "min_coherence", "max_topic_fraction",
            "min_output_focus", "n_output_topics",
        }
        clean_params = {k: v for k, v in params.items() if k in cfg_keys}
        # Tipos correctos
        int_keys = {"top_k_per_aspect", "min_cluster_size", "min_samples",
                    "umap_n_components", "umap_n_neighbors", "random_state",
                    "min_nr_topics", "max_nr_topics", "n_output_topics"}
        for k in int_keys:
            if k in clean_params:
                clean_params[k] = int(clean_params[k])
        float_keys = cfg_keys - int_keys
        for k in float_keys:
            if k in clean_params:
                clean_params[k] = float(clean_params[k])

        cfg = make_tuned_config(odp_path, cache_dir=CACHE_DIR, **clean_params)

        if domain_kw is not None:
            kw_tuple = tuple(w.strip() for w in domain_kw.split(",") if w.strip())
            cfg.domain_category_keywords = kw_tuple

        # ── Log de inicio de sesión ──────────────────────────────────────────
        import dataclasses, datetime
        enc_labels = {
            "node_label":     "Etiqueta de nodo",
            "relation_aware": "Relacional (tripletas)",
            "path_aware":     "Caminos (path-aware)",
            "hybrid":         "Híbrida",
        }
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("=" * 60)
        print(f"▶ NUEVA SESIÓN EXTENDER — {ts}")
        print(f"  Mapa      : {cm.title}")
        print(f"  Encoder   : {enc_labels.get(encoder_key, encoder_key)}")
        if domain_kw:
            print(f"  Keywords  : {domain_kw}")
        print("  Parámetros:")
        cfg_dict = dataclasses.asdict(cfg)
        for k, v in sorted(cfg_dict.items()):
            print(f"    {k:<38} = {v}")
        print("=" * 60)

        # ── Ejecutar pipeline ────────────────────────────────────────────────
        from topic_explorer import TopicExplorer
        enc_cls   = ENCODER_CLASSES.get(encoder_key, list(ENCODER_CLASSES.values())[0])
        exp       = TopicExplorer(corpus=corpus, concept_map=cm, config=cfg, focus_concepts=[], encoder_class=enc_cls)
        topics_raw = exp.run() or []

        # ── Serializar tópicos ───────────────────────────────────────────────
        topics_out = []
        for t in topics_raw:
            docs = [
                {
                    "url":      d.get("url", ""),
                    "title":    d.get("title", ""),
                    "category": d.get("category", ""),
                    "snippet":  d.get("snippet", "")[:250],
                }
                for d in (t.documents or [])
            ]
            # Centroide del tópico (ya calculado en TopicExplorer._build_explored_topics)
            _centroid = t.centroid if hasattr(t, "centroid") else None
            topics_out.append({
                "topic_id":   t.topic_id,
                "label":      t.label,
                "keywords":   t.keywords,
                "coherence":  round(float(t.coherence), 4),
                "focus_score": round(float(t.focus_score), 4),
                "doc_count":  len(docs),
                "documents":  docs,
                "centroid":   _centroid,
            })

        result = {
            "topics":       topics_out,
            "n_candidates": len(exp._candidate_indices),
            "is_known_map": bool(known_map),
        }

        # ── Métricas de recall/precisión solo para mapas conocidos ───────────
        if known_map and known_map in MAP_TO_TOPIC:
            target = MAP_TO_TOPIC[known_map]["odp_path"]
            total  = state["docs_per_category"].get(target, 0)
            if total > 0:
                m = precision_recall_for_map(exp, corpus, target, total)
                result["metrics"] = {
                    "n_candidatos":              m["n_candidatos"],
                    "precision_retrieval_at_10": m["precision_retrieval_at_10"],
                    "recall_retrieval":          m["recall_retrieval"],
                    "n_docs_topicos_finales":    m["n_docs_topicos_finales"],
                    "precision_topicos_finales": m["precision_topicos_finales"],
                    "recall_topicos_finales":    m["recall_topicos_finales"],
                }

        # Focus y coherencia promedio (siempre)
        if topics_out:
            result["avg_focus"]     = round(sum(t["focus_score"] for t in topics_out) / len(topics_out), 4)
            result["avg_coherence"] = round(sum(t["coherence"]   for t in topics_out) / len(topics_out), 4)
        else:
            result["avg_focus"]     = 0.0
            result["avg_coherence"] = 0.0

        # ── Aspect vectors + map_vec (para exportar en timeline) ────────────
        try:
            import numpy as _np2
            # exp.encoder.aspects ya está calculado: aspects[0] = centroide, aspects[1..] = aspects
            _aspects_out = [
                {"index": _i, "embedding": _v.tolist()}
                for _i, _v in enumerate(exp.encoder.aspects)
            ]
            # map_vec: usar el vector ya calculado por el explorer (mismo que usó el pipeline).
            # exp._map_vec ya está normalizado; reconstruimos el text para referencia
            # usando los mismos parámetros que _compute_map_vec_and_gate.
            _focus_mult = 3.0  # igual que en topic_explorer._compute_map_vec_and_gate
            _map_text = " ".join(
                _t for _t, _ in sorted(
                    cm.term_weights(focus_concepts=exp.focus, focus_multiplier=_focus_mult).items(),
                    key=lambda x: -x[1],
                )[:30]
            )
            _map_vec_norm = exp._map_vec.tolist() if exp._map_vec is not None else []
            # Per-node/per-text embeddings (the individual vectors aggregated
            # to build map_vec and aspects — useful for visualisation).
            _nodes_out = []
            try:
                if encoder_key == "node_label":
                    # ConceptMapEncoder: one vector per node label
                    _le = exp.encoder.label_embeddings  # already cached from aspect build
                    for _lbl, _vec in _le.items():
                        _nodes_out.append({"label": _lbl, "embedding": _vec.tolist()})
                elif encoder_key in ("relation_aware", "path_aware", "hybrid"):
                    # Each encoder exposes a method to get its raw texts+weights
                    if encoder_key == "relation_aware":
                        _texts, _wts = exp.encoder._triple_texts_and_weights()
                    elif encoder_key == "path_aware":
                        _texts, _wts = exp.encoder._path_texts_and_weights()
                    else:  # hybrid
                        _texts, _wts = exp.encoder._get_texts_weights()
                    _raw_embs = exp.encoder.engine.embed(_texts)
                    _norms = _np2.linalg.norm(_raw_embs, axis=1, keepdims=True)
                    _norms[_norms == 0] = 1.0
                    _embs_n = _raw_embs / _norms
                    for _t, _v in zip(_texts, _embs_n):
                        _nodes_out.append({"label": _t, "embedding": _v.tolist()})
            except Exception:
                pass

            result["repr_embeddings"] = {
                "aspects":    _aspects_out,
                "map_vec":    {"text": _map_text, "embedding": _map_vec_norm},
                "n_aspects":  len(_aspects_out),
                "encoder":    encoder_key,
                "nodes":      _nodes_out,
            }
        except Exception:
            pass  # No fallar el run si esto falla

        return jsonify(result)

    except Exception:
        tb = traceback.format_exc()
        print(tb)
        return jsonify({"error": tb}), 500


# ── Grid Search ────────────────────────────────────────────────────────────────

_gs_jobs: dict       = {}   # job_id → {status, progress, total, results, best, best_enc, error}
_gs_stop_flags: dict = {}   # job_id → bool  (True = cancelar)

GS_GRID_FAST = {
    "domain_category_min_fraction": [0.15, 0.30],
    "n_output_topics":              [6, 10],
    "min_cluster_size":             [3],
    "min_samples":                  [1],
}
GS_GRID_FULL = {
    "domain_category_min_fraction": [0.10, 0.20, 0.30],
    "n_output_topics":              [6, 8, 10],
    "min_cluster_size":             [2, 3, 5],
    "min_samples":                  [1],
}


_GS_INT_PARAMS = {"n_output_topics", "min_cluster_size", "min_samples",
                  "umap_n_neighbors", "umap_n_components",
                  "path_max_depth", "path_max_paths_per_node",
                  "hybrid_window_k", "hybrid_max_paths_used_per_node"}

# Defaults usados cuando un parámetro no está en el custom_grid
_GS_PARAM_DEFAULTS = {
    "domain_category_min_fraction": 0.15,
    "n_output_topics":              6,
    "min_cluster_size":             3,
    "min_samples":                  1,
}


def _run_gs_one(cm, enc_key, hp, odp_path):
    """Un run del pipeline optimizando (avg_focus + avg_coh) / 2."""
    from concept_map_encoder import ConceptMapEncoder, RelationAwareEncoder, PathAwareEncoder, HybridEncoder
    from tuner import make_tuned_config
    ENCODER_CLASSES = {
        "node_label":     ConceptMapEncoder,
        "relation_aware": RelationAwareEncoder,
        "path_aware":     PathAwareEncoder,
        "hybrid":         HybridEncoder,
    }
    try:
        overrides = {
            "domain_category_min_fraction": float(hp.get("domain_category_min_fraction", 0.15)),
            "n_output_topics":              int(hp.get("n_output_topics",              6)),
            "min_cluster_size":             int(hp.get("min_cluster_size",             3)),
            "min_samples":                  int(hp.get("min_samples",                  1)),
        }
        if "min_output_focus"  in hp: overrides["min_output_focus"]  = float(hp["min_output_focus"])
        if "umap_n_neighbors"  in hp: overrides["umap_n_neighbors"]  = int(hp["umap_n_neighbors"])
        if "umap_n_components" in hp: overrides["umap_n_components"] = int(hp["umap_n_components"])
        if "path_max_depth"                 in hp: overrides["path_max_depth"]                 = int(hp["path_max_depth"])
        if "path_max_paths_per_node"        in hp: overrides["path_max_paths_per_node"]        = int(hp["path_max_paths_per_node"])
        if "hybrid_window_k"                in hp: overrides["hybrid_window_k"]                = int(hp["hybrid_window_k"])
        if "hybrid_max_paths_used_per_node" in hp: overrides["hybrid_max_paths_used_per_node"] = int(hp["hybrid_max_paths_used_per_node"])
        cfg = make_tuned_config(odp_path, CACHE_DIR, **overrides)
        from topic_explorer import TopicExplorer
        enc_cls = ENCODER_CLASSES[enc_key]
        exp     = TopicExplorer(corpus=state["corpus"], concept_map=cm, config=cfg, focus_concepts=[], encoder_class=enc_cls)
        topics = exp.run() or []
        if topics:
            avg_focus = sum(t.focus_score for t in topics) / len(topics)
            avg_coh   = sum(t.coherence   for t in topics) / len(topics)
        else:
            avg_focus = avg_coh = 0.0
        score = round((avg_focus + avg_coh) / 2, 4)
        return {"score": score, "focus": round(avg_focus, 4), "coh": round(avg_coh, 4)}
    except Exception as e:
        return {"score": 0.0, "focus": 0.0, "coh": 0.0, "error": str(e)}


@app.route("/api/grid_search", methods=["POST"])
def api_grid_search():
    if not state["ready"]:
        return jsonify({"error": "Backend todavía cargando."}), 503

    data      = request.get_json(force=True)
    cm_data   = data.get("concept_map", {})
    known_map = data.get("known_map_name")
    fast_mode   = data.get("fast_mode", True)
    custom_grid = data.get("custom_grid")   # {param: [val, ...]} or None
    enc_keys    = data.get("encoder_keys") or ["node_label", "relation_aware", "path_aware", "hybrid"]

    job_id = f"gs_{int(time.time() * 1000)}"
    _gs_jobs[job_id] = {
        "status": "running", "progress": 0, "total": 0,
        "results": [], "best": {}, "best_enc": None, "error": None,
    }
    _gs_stop_flags[job_id] = False

    def _run():
        try:
            try:
                from concept_maps import MAP_TO_TOPIC
            except Exception:
                MAP_TO_TOPIC = {}

            cm = _build_concept_map(cm_data)
            if not cm.nodes:
                raise ValueError("Mapa vacío — agregá nodos primero.")

            if known_map and known_map in MAP_TO_TOPIC:
                odp_path = MAP_TO_TOPIC[known_map]["odp_path"]
            else:
                title_words = cm.title.replace("-", "/").replace(" ", "/")
                odp_path = f"Top/{title_words}" if title_words else "Top/Unknown"

            if custom_grid:
                hp_grid = {}
                for k, vals in custom_grid.items():
                    if not vals: continue
                    coerced = [int(v) if k in _GS_INT_PARAMS else float(v) for v in vals]
                    hp_grid[k] = coerced
                # Rellenar parámetros requeridos faltantes con el default de GS_GRID_FAST
                for k, default_list in GS_GRID_FAST.items():
                    if k not in hp_grid:
                        hp_grid[k] = default_list
                if not hp_grid:
                    hp_grid = GS_GRID_FAST
            else:
                hp_grid   = GS_GRID_FAST if fast_mode else GS_GRID_FULL
            hp_keys   = list(hp_grid.keys())
            hp_combos = list(itertools.product(*[hp_grid[k] for k in hp_keys]))
            total = len(hp_combos) * len(enc_keys)
            _gs_jobs[job_id]["total"] = total

            print(f"\n{'='*65}")
            print(f"Grid search: {cm.title}  |  {len(hp_combos)} HP combos × {len(enc_keys)} encoders = {total} runs")
            print(f"{'='*65}")

            best_by_enc = {enc: {"score": -1.0, "params": None, "focus": 0.0, "coh": 0.0}
                           for enc in enc_keys}
            run_i = 0
            _gs_jobs[job_id]["run_times"] = []  # elapsed seconds per run

            stopped = False
            for enc in enc_keys:
                if stopped: break
                for hp_combo in hp_combos:
                    if _gs_stop_flags.get(job_id):
                        stopped = True
                        break
                    hp    = dict(zip(hp_keys, hp_combo))
                    run_i += 1
                    t0  = time.time()
                    res = _run_gs_one(cm, enc, hp, odp_path)
                    elapsed = round(time.time() - t0, 2)
                    _gs_jobs[job_id]["run_times"].append(elapsed)

                    is_best = res["score"] > best_by_enc[enc]["score"]
                    if is_best:
                        best_by_enc[enc] = {
                            "score": res["score"], "focus": res["focus"],
                            "coh": res["coh"],     "params": hp,
                        }
                    _gs_jobs[job_id]["results"].append({
                        "run_i": run_i, "enc": enc, "params": hp,
                        "focus": res["focus"], "coh": res["coh"],
                        "score": res["score"], "is_best": is_best,
                        "error": res.get("error"),
                        "elapsed_s": elapsed,
                    })
                    _gs_jobs[job_id]["progress"] = run_i

                    flag = "★" if is_best else " "
                    print(f"  {flag}[{run_i:2d}/{total}]"
                          f" enc={enc[:5]:5s}"
                          f" | score={res['score']:.4f}"
                          f" focus={res['focus']:.3f} coh={res['coh']:.3f}"
                          f"  ({elapsed:.1f}s)")

            if stopped:
                _gs_jobs[job_id]["status"] = "stopped"
                print(f"\n⚑ Grid search cancelado en run {run_i}/{total}")
                return

            best_enc = max(best_by_enc.items(), key=lambda x: x[1]["score"])[0]
            best_summary = {
                enc: {"score": v["score"], "focus": v["focus"], "coh": v["coh"], "params": v["params"]}
                for enc, v in best_by_enc.items()
            }

            # ── Top 5 por encoder × métrica ───────────────────────────────
            all_results = _gs_jobs[job_id]["results"]
            top5_by_enc = {}
            for enc in enc_keys:
                enc_runs = [r for r in all_results if r["enc"] == enc and not r.get("error")]
                def _top5(runs, key):
                    return sorted(runs, key=lambda r: r.get(key, 0), reverse=True)[:5]
                top5_by_enc[enc] = {
                    "score": _top5(enc_runs, "score"),
                    "focus": _top5(enc_runs, "focus"),
                    "coh":   _top5(enc_runs, "coh"),
                }
            _gs_jobs[job_id]["best"]       = best_summary
            _gs_jobs[job_id]["best_enc"]   = best_enc
            _gs_jobs[job_id]["top5_by_enc"] = top5_by_enc
            _gs_jobs[job_id]["status"]     = "done"

            # ── Guardar en saved_maps_and_configuration/ ──────────────────
            import re as _re
            safe_title = _re.sub(r"[^\w\-]", "_", cm.title)[:60]
            save_path  = os.path.join(SAVED_MAPS_DIR, f"{safe_title}_gridsearch.json")
            save_data  = {
                "map_name":    cm.title,
                "known_map":   known_map,
                "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%S"),
                "fast_mode":   fast_mode,
                "best_enc":    best_enc,
                "best_configs": best_summary,
                "top5_by_enc": top5_by_enc,
                "concept_map": cm_data,
            }
            with open(save_path, "w", encoding="utf-8") as _f:
                json.dump(save_data, _f, indent=2, ensure_ascii=False)
            _gs_jobs[job_id]["saved_path"] = save_path
            print(f"\n✓ Grid search completo. Mejor encoder: {best_enc}"
                  f" (score={best_by_enc[best_enc]['score']:.4f})")
            print(f"  Guardado en: {save_path}")

        except Exception:
            tb = traceback.format_exc()
            print(tb)
            _gs_jobs[job_id]["status"] = "error"
            _gs_jobs[job_id]["error"]  = tb

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/grid_search/status/<job_id>")
def api_grid_search_status(job_id):
    job = _gs_jobs.get(job_id)
    if not job:
        return jsonify({"error": "job not found"}), 404
    return jsonify(job)


@app.route("/api/grid_search/stop/<job_id>", methods=["POST"])
def api_grid_search_stop(job_id):
    _gs_stop_flags[job_id] = True
    return jsonify({"ok": True})


@app.route("/api/saved_gridsearches")
def api_saved_gridsearches():
    """Lista los archivos de grid search guardados en SAVED_MAPS_DIR."""
    files = []
    if os.path.isdir(SAVED_MAPS_DIR):
        for fname in sorted(os.listdir(SAVED_MAPS_DIR)):
            if not fname.endswith("_gridsearch.json"):
                continue
            fpath = os.path.join(SAVED_MAPS_DIR, fname)
            try:
                with open(fpath, encoding="utf-8") as f:
                    data = json.load(f)
                files.append({
                    "filename":  fname,
                    "map_name":  data.get("map_name", fname),
                    "timestamp": data.get("timestamp", ""),
                    "best_enc":  data.get("best_enc", ""),
                    "fast_mode": data.get("fast_mode", True),
                    "best_configs": data.get("best_configs", {}),
                })
            except Exception:
                pass
    return jsonify({"files": files})


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC STUDY
# Análisis extensivo de tópicos del corpus activo.
# Se dispara automáticamente en background la primera vez que se carga cada
# corpus. Resultados disponibles en /api/topic_study.
# ══════════════════════════════════════════════════════════════════════════════

_topic_study_state:    dict = {}   # corpus_type → "computing" | list[dict]
_topic_study_centroids: dict = {}  # corpus_type → np.ndarray (n_topics × 384)
_topic_study_url_map:  dict = {}   # corpus_type → {url: result_idx}  (0-indexed into sorted results)
_topic_study_corpus_docs: dict = {}  # corpus_type → [{url,t,sn,tid,s:[...]}] — ALL corpus docs
_topic_study_corpus_embs: dict = {}  # corpus_type → np.ndarray (n_docs × 384) normalised
_topic_study_lock     = threading.Lock()

# ── Cargar centroides persistidos en disco (permite skip de recompute tras restart) ──
def _load_persisted_topic_study():
    import json as _json
    for _ct in ("paper", "wikipedia", "dataviz_wikipedia",
                "cg_wikipedia", "oop_wikipedia", "vinos_wikipedia", "autos_wikipedia"):
        _c_path = os.path.join(CACHE_DIR, f"topic_centroids_{_ct}.npy")
        _s_path = os.path.join(CACHE_DIR, f"topic_state_{_ct}.json")
        if os.path.isfile(_c_path) and os.path.isfile(_s_path):
            try:
                _topic_study_centroids[_ct] = np.load(_c_path).astype(np.float32)
                with open(_s_path, "r", encoding="utf-8") as _sf:
                    _topic_study_state[_ct] = _json.load(_sf)
                # Cargar url_map persistido si existe
                _u_path = os.path.join(CACHE_DIR, f"topic_url_map_{_ct}.json")
                if os.path.isfile(_u_path):
                    with open(_u_path, "r", encoding="utf-8") as _uf:
                        _topic_study_url_map[_ct] = _json.load(_uf)
                    print(f"[topic_study] Cargado desde disco: {_ct} ({len(_topic_study_state[_ct])} tópicos, {len(_topic_study_url_map[_ct])} URLs)")
                else:
                    print(f"[topic_study] Cargado desde disco: {_ct} ({len(_topic_study_state[_ct])} tópicos, sin url_map)")
                # Cargar corpus docs + embs si existen
                _d_path = os.path.join(CACHE_DIR, f"topic_corpus_docs_{_ct}.json")
                _e_path = os.path.join(CACHE_DIR, f"topic_corpus_embs_{_ct}.npy")
                if os.path.isfile(_d_path):
                    with open(_d_path, "r", encoding="utf-8") as _df:
                        _topic_study_corpus_docs[_ct] = _json.load(_df)
                    print(f"[topic_study] Corpus docs cargados: {_ct} ({len(_topic_study_corpus_docs[_ct])} docs)")
                if os.path.isfile(_e_path):
                    _topic_study_corpus_embs[_ct] = np.load(_e_path).astype(np.float32)
            except Exception as _le:
                print(f"[topic_study] WARN: no se pudo cargar caché de disco para {_ct}: {_le}")

_load_persisted_topic_study()


def _cosine_sim(a, B):
    a_n  = a / (float(np.linalg.norm(a)) + 1e-12)
    B_n  = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return B_n @ a_n


def _make_readable_name(keywords: list[str], top_categories: list | None = None) -> str:
    """
    Genera un nombre legible para un tópico.

    Estrategia (en orden de prioridad):
    1. Derivar nombre desde las categorías del corpus (más confiable y específico).
    2. Matching por patrones de keywords ponderados por longitud del término
       (términos multi-palabra son más específicos y pesan más).
    3. Fallback: combinar las keywords más informativas directamente.
    """
    import re as _re

    # ── Expansión de abreviaturas conocidas ───────────────────────────────────
    _ABBR = {
        "gis":   "Geographic Information Systems",
        "eda":   "Exploratory Data Analysis",
        "bi":    "Business Intelligence",
        "ai":    "Artificial Intelligence",
        "nlp":   "Natural Language Processing",
        "cad":   "CAD & Engineering",
        "pos":   "Point of Sale",
        "hci":   "Human-Computer Interaction",
        "ml":    "Machine Learning",
        "ui":    "User Interface",
        "ux":    "User Experience",
    }
    # Segmentos de categoría demasiado genéricos para usar solos como nombre
    _GENERIC = {
        "dataviz", "infographics", "computers", "software", "hardware",
        "tools", "data", "general", "misc", "other", "top", "multivariate",
        # También la forma colapsada (sin espacios) para _seg_is_generic
        "dataandinformationvisualization",
    }
    _STOP_WORDS = {
        "and","or","the","a","an","of","in","on","for","with","to","is",
        "are","was","were","used","using","based","system","systems",
        "method","methods","com","home","page","site",
    }

    def _clean_seg(seg: str) -> str:
        """Convierte un segmento de ruta de categoría en texto legible."""
        seg = seg.strip()
        low = seg.lower()
        if low in _ABBR:
            return _ABBR[low]
        seg = seg.replace("_", " ").replace("-", " ")
        # camelCase → separate words: "ComputerGraphics" → "Computer Graphics"
        seg = _re.sub(r"([a-z])([A-Z])", r"\1 \2", seg)
        seg = _re.sub(r"\s+", " ", seg).strip()
        # Preservar acrónimos (todo mayúsculas, ≤4 letras), capitalizar el resto
        parts = []
        for w in seg.split():
            wl = w.lower()
            if wl in _ABBR:
                parts.append(_ABBR[wl])
            elif w.isupper() and len(w) <= 4:
                parts.append(w)
            else:
                parts.append(w.capitalize())
        return " ".join(parts)

    def _seg_is_generic(seg: str) -> bool:
        return seg.lower().replace(" ", "").replace("_", "") in _GENERIC

    # Conjunciones/preposiciones que van en minúsculas dentro de un título
    _LOWER_IN_TITLE = {"and","or","of","the","a","an","in","on","for","with","to","at","by"}

    def _title(text: str) -> str:
        """Title-case preservando acrónimos y bajando conjunciones internas."""
        words = text.split()
        result = []
        for i, w in enumerate(words):
            if w.isupper() and len(w) <= 4:          # acrónimo: GIS, API…
                result.append(w)
            elif i > 0 and w.lower() in _LOWER_IN_TITLE:
                result.append(w.lower())
            else:
                result.append(w.capitalize())
        return " ".join(result)

    # Prefijos de leaf que deben eliminarse por ser poco descriptivos
    _STRIP_PREFIXES = (
        "discontinued ", "list of ", "lists of ", "history of ",
        "subfields of ", "overview of ", "types of ",
    )
    # Padres que aportan contexto útil como adjetivo (singular de la categoría)
    # Formato: f"{ctx} {leaf_word}" → ej. "Computers/Databases" → "Computer Databases"
    _CONTEXTUAL_PARENTS = {
        "computers": "Computer", "hardware": "Hardware",
        "software": "Software", "algorithms": "Algorithm",
        "distributed": "Distributed", "programming": "Programming",
        "operating_systems": "Operating System", "graphics": "Graphics",
    }
    # Padres de tipo "categoría contenedora" → formato invertido: f"{leaf_word} {ctx}"
    # Ej: "Charts/Statistical" → "Statistical Charts", "Maps/Cartography" → "Cartography Maps"
    _CONTEXTUAL_PARENTS_SUFFIX = {
        "charts": "Charts",
        "maps": "Maps",
    }

    def _name_from_path(path: str) -> str | None:
        """
        Extrae un nombre representativo de una ruta de categoría.

        Reglas (en orden):
        1. Si el leaf está en ABBR → expansión directa.
        2. Si el leaf es genérico → usar padre.
        3. Eliminar prefijos espurios ("Discontinued", "List of", etc.).
        4. Si leaf muy largo (>5 palabras útiles) → últimas 3 palabras informativas.
        5. Si leaf es 1 sola palabra → intentar enriquecer con el padre.
        6. Retornar leaf limpio.
        """
        parts = [p.strip() for p in path.split("/") if p.strip()]
        if not parts:
            return None

        leaf_raw = parts[-1]

        # 1. Abreviatura directa
        if leaf_raw.lower() in _ABBR:
            return _ABBR[leaf_raw.lower()]

        # 2. Leaf genérico → subir al padre
        if _seg_is_generic(leaf_raw) or len(leaf_raw) <= 2:
            if len(parts) >= 2 and not _seg_is_generic(parts[-2]):
                return _title(_clean_seg(parts[-2]))
            return None

        leaf = _clean_seg(leaf_raw)

        # 3. Eliminar prefijos poco informativos
        leaf_lower = leaf.lower()
        for pfx in _STRIP_PREFIXES:
            if leaf_lower.startswith(pfx):
                leaf = _title(leaf[len(pfx):])
                break

        # 4. Si el leaf empieza con el contenido del padre → el padre es más claro
        #    Ej: "Cluster analysis/Cluster analysis algorithms" → retornar padre "Cluster Analysis"
        #    EXCEPCIÓN: si el padre mismo es genérico (ej. "Data and information visualization")
        #    no retornar el padre — dejar pasar el leaf completo que es más específico.
        if len(parts) >= 2:
            parent_clean = _clean_seg(parts[-2]).lower()
            if (leaf.lower().startswith(parent_clean)
                    and len(parent_clean) >= 6
                    and not _seg_is_generic(parent_clean)):
                parent_name = _clean_seg(parts[-2])
                parent_words = [w for w in parent_name.split() if w.lower() not in _STOP_WORDS]
                if len(parent_words) >= 2:
                    return _title(parent_name)

        # 5. Leaf muy largo (>4 palabras de contenido) → truncar a las últimas 3
        leaf_words = [w for w in leaf.split() if w.lower() not in _STOP_WORDS]
        if len(leaf_words) > 4:
            return _title(" ".join(leaf_words[-3:]))

        # 6. Leaf de 1 sola palabra → enriquecer con el padre si aporta contexto
        if len(leaf_words) == 1 and len(parts) >= 2:
            parent_key = parts[-2].lower().replace("-", "_").replace(" ", "_")
            # 6a. Padre como prefijo: "Computers/Databases" → "Computer Databases"
            ctx = _CONTEXTUAL_PARENTS.get(parent_key)
            if ctx and not _seg_is_generic(parts[-2]) and leaf_words[0].lower() not in ctx.lower():
                return _title(f"{ctx} {leaf_words[0]}")
            # 6b. Padre como sufijo (categoría contenedora): "Charts/Statistical" → "Statistical Charts"
            ctx_sfx = _CONTEXTUAL_PARENTS_SUFFIX.get(parent_key)
            if ctx_sfx and not _seg_is_generic(parts[-2]) and leaf_words[0].lower() not in ctx_sfx.lower():
                return _title(f"{leaf_words[0]} {ctx_sfx}")

        return _title(leaf)

    # ── 1. Naming basado en categorías ────────────────────────────────────────
    if top_categories:
        # top_categories puede ser lista de tuplas (cat, n) o dicts {"cat":…, "n":…}
        normalized = []
        for item in top_categories[:4]:
            if isinstance(item, dict):
                normalized.append((item.get("cat", ""), item.get("n", 0)))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                normalized.append((str(item[0]), int(item[1])))

        # Buscar el mejor nombre desde las categorías más representadas.
        # Solo aceptar si el resultado tiene ≥ 2 palabras: evita leaves de 1 palabra
        # genérica que serían mejores resueltos por los patrones de keywords.
        for path, _n in normalized:
            name = _name_from_path(path)
            if name and not _seg_is_generic(name) and len(name.split()) >= 2:
                return name

    # ── 2. Pattern matching por keywords (ponderado por longitud del término) ──
    # Los términos multi-palabra son más específicos → pesan más que los de 1 palabra.
    # El score es la suma de len(term.split()) por cada término encontrado.
    patterns = [
        # DataViz / estadística
        (["chart", "charts", "plot", "plots", "diagram", "bar chart", "pie chart", "histogram"], "Charts & Plots"),
        (["cartography", "maps", "mapping", "geographic map", "map projection"], "Cartography & Maps"),
        (["gis", "geographic information", "spatial database", "geospatial", "spatial analysis"], "Geographic Information Systems"),
        (["graph drawing", "graph theory", "planar graph", "planar", "graph layout"], "Graph Theory & Drawing"),
        (["cluster analysis", "clustering", "k-means", "dbscan", "cluster"], "Cluster Analysis"),
        (["dimension reduction", "principal component", "pca", "manifold", "projection"], "Dimensionality Reduction"),
        (["multivariate", "analysis of variance", "anova", "covariance", "correlation"], "Multivariate Analysis"),
        (["statistics", "statistical", "regression", "hypothesis testing", "bayesian"], "Statistical Analysis"),
        (["data visualization", "information visualization", "visual analytics", "visualization software"], "Data Visualization"),
        (["data scientist", "professor", "academic", "researcher", "scientist"], "Data Scientists & Researchers"),
        # CS General
        (["machine learning", "deep learning", "neural network", "gradient", "backprop"], "Machine Learning & AI"),
        (["natural language processing", "nlp", "text mining", "sentiment", "language model", "bert", "gpt"], "NLP & Text Processing"),
        (["operating system", "kernel", "linux", "unix", "process scheduler"], "Operating Systems"),
        (["database", "sql", "query", "relational", "nosql", "mongodb", "postgresql"], "Databases"),
        (["network", "tcp", "ip", "router", "protocol", "ethernet", "wifi", "wireless"], "Computer Networks"),
        (["security", "encryption", "cryptography", "firewall", "vulnerability", "cyber"], "Cybersecurity"),
        (["distributed system", "hadoop", "spark", "mapreduce", "fault tolerance"], "Distributed Systems"),
        (["programming language", "compiler", "syntax", "interpreter", "runtime"], "Programming Languages"),
        (["software engineering", "design pattern", "agile", "testing", "devops"], "Software Engineering"),
        (["algorithm", "data structure", "sorting", "complexity", "dynamic programming"], "Algorithms & Data Structures"),
        (["computer vision", "object detection", "image segmentation", "cnn", "pixel"], "Computer Vision"),
        (["web", "http", "html", "browser", "frontend", "rest api", "css"], "Web Technologies"),
        (["cloud computing", "aws", "azure", "virtualization", "container", "docker", "kubernetes"], "Cloud Computing"),
        (["hardware", "cpu", "gpu", "processor", "embedded", "fpga", "scsi"], "Computer Hardware"),
        (["artificial intelligence", "expert system", "knowledge representation", "reasoning"], "Artificial Intelligence"),
        (["robotics", "automation", "control system", "sensor", "actuator", "autonomous"], "Robotics & Automation"),
        (["mobile", "android", "ios", "smartphone", "bluetooth", "wireless"], "Mobile Computing"),
        (["user interface", "human computer interaction", "ux", "accessibility", "hci"], "HCI & UX"),
        (["computer graphics", "3d", "rendering", "opengl", "shader", "texture", "ray tracing"], "Computer Graphics & 3D"),
        (["barcode", "bar code", "point of sale", "pos", "retail"], "POS & Barcode Systems"),
        (["storage", "tape", "backup", "raid", "scsi", "disk drive"], "Data Storage"),
        (["cad", "autocad", "3d modeling", "engineering design", "cam"], "CAD & Engineering"),
        (["training", "certification", "e-learning", "course", "tutorial"], "IT Training & Certification"),
        (["game engine", "game", "physics engine", "game development", "shader"], "Game Development"),
        (["data science", "big data", "analytics", "data mining", "data pipeline"], "Data Science & Analytics"),
    ]
    kw_lower = " ".join(keywords[:8]).lower()
    best_name, best_score = None, 0
    for terms, name in patterns:
        # Ponderar por longitud: "cluster analysis" (2 palabras) pesa más que "cluster" (1 palabra)
        score = sum(len(t.split()) for t in terms if t in kw_lower)
        if score > best_score:
            best_score, best_name = score, name
    if best_name and best_score >= 2:   # umbral 2: requiere al menos un término bi-grama o dos uni-grama
        return best_name

    # ── 3. Fallback: keywords más informativas ────────────────────────────────
    # Preferir keywords multi-palabra (más específicas) sobre uni-palabra
    selected = []
    for kw in keywords[:6]:
        if " " in kw:
            selected.append(kw.title())
    for kw in keywords[:4]:
        if " " not in kw and kw.lower() not in _STOP_WORDS:
            selected.append(kw.capitalize())
    unique = list(dict.fromkeys(selected))
    if unique:
        return " & ".join(unique[:2]) if len(unique) >= 2 else unique[0]
    return keywords[0].title() if keywords else "Unknown Topic"


def _run_topic_study(corpus_type: str):
    """Background thread: compute topic study for the given corpus."""
    import warnings
    import pandas as _pd_ts
    warnings.filterwarnings("ignore")
    _pd_ts.options.mode.chained_assignment = None  # silencia SettingWithCopyWarning de BERTopic

    with _topic_study_lock:
        if _topic_study_state.get(corpus_type) not in (None, "error"):
            return   # already running or done
        _topic_study_state[corpus_type] = "computing"

    print(f"[topic_study] Iniciando análisis para corpus '{corpus_type}'...")

    try:
        # ── 1. Determine embeddings cache + corpus metadata ───────────────────
        if corpus_type == "paper":
            npz_candidates = [
                os.path.join(CACHE_DIR, "curlie_embeddings_sbert_5781b0ace4.npz"),
            ]
            pkl_path    = os.path.join(PAPER_DATASET_DIR, ".paper_corpus_150_2000.pkl")
            cat_prefix  = "Top/Computers/"
            wiki_dir    = None
        elif corpus_type == "wikipedia":
            npz_candidates = [
                os.path.join(CACHE_DIR, "curlie_embeddings_sbert_646085e7b7.npz"),
            ]
            pkl_path    = os.path.join(WIKIPEDIA_DIR, ".wiki_corpus_all.pkl")
            cat_prefix  = None
            wiki_dir    = WIKIPEDIA_DIR
        elif corpus_type == "dataviz_wikipedia":
            npz_candidates = [
                os.path.join(CACHE_DIR, "curlie_embeddings_sbert_0bec156c44.npz"),
            ]
            pkl_path    = os.path.join(DATAVIZ_WIKIPEDIA_DIR, ".wiki_corpus_all.pkl")
            cat_prefix  = None
            wiki_dir    = DATAVIZ_WIKIPEDIA_DIR
        elif corpus_type == "cg_wikipedia":
            npz_candidates = [os.path.join(CACHE_DIR, "topic_embs_cg_wikipedia.npz")]
            pkl_path    = os.path.join(CG_WIKIPEDIA_DIR, ".wiki_corpus_all.pkl")
            cat_prefix  = None
            wiki_dir    = CG_WIKIPEDIA_DIR
        elif corpus_type == "oop_wikipedia":
            npz_candidates = [os.path.join(CACHE_DIR, "topic_embs_oop_wikipedia.npz")]
            pkl_path    = os.path.join(OOP_WIKIPEDIA_DIR, ".wiki_corpus_all.pkl")
            cat_prefix  = None
            wiki_dir    = OOP_WIKIPEDIA_DIR
        elif corpus_type == "vinos_wikipedia":
            npz_candidates = [os.path.join(CACHE_DIR, "topic_embs_vinos_wikipedia.npz")]
            pkl_path    = os.path.join(VINOS_WIKIPEDIA_DIR, ".wiki_corpus_all.pkl")
            cat_prefix  = None
            wiki_dir    = VINOS_WIKIPEDIA_DIR
        elif corpus_type == "autos_wikipedia":
            npz_candidates = [os.path.join(CACHE_DIR, "topic_embs_autos_wikipedia.npz")]
            pkl_path    = os.path.join(AUTOS_WIKIPEDIA_DIR, ".wiki_corpus_all.pkl")
            cat_prefix  = None
            wiki_dir    = AUTOS_WIKIPEDIA_DIR
        else:
            print(f"[topic_study] corpus '{corpus_type}' no soportado.")
            _topic_study_state[corpus_type] = []
            return

        # Find .npz (may need to compute on-the-fly for new corpora)
        npz_path = next((p for p in npz_candidates if os.path.isfile(p)), None)
        if not npz_path:
            # Fallback: compute embeddings from pkl corpus if engine is ready
            if wiki_dir and os.path.isfile(pkl_path):
                print(f"[topic_study] No se encontró .npz para '{corpus_type}'. Calculando embeddings...")
                try:
                    import pickle as _pickle
                    with open(pkl_path, "rb") as fh:
                        _entries = _pickle.load(fh)
                    _urls_raw  = [e.url for e in _entries]
                    _texts_raw = [
                        f"{getattr(e,'title','')}. {(getattr(e,'description','') or '')[:300]}"
                        for e in _entries
                    ]
                    _engine = state.get("engine")
                    if _engine is None:
                        raise RuntimeError("EmbeddingEngine no disponible aún.")
                    print(f"[topic_study] Codificando {len(_texts_raw)} textos con SBERT...")
                    _embs = _engine.embed(_texts_raw).astype(np.float32)
                    npz_path = npz_candidates[0]
                    np.savez_compressed(npz_path, embeddings=_embs, urls=np.array(_urls_raw))
                    print(f"[topic_study] .npz guardado: {npz_path}")
                except Exception as _emb_e:
                    print(f"[topic_study] Error calculando embeddings: {_emb_e}")
                    _topic_study_state[corpus_type] = []
                    return
            else:
                print(f"[topic_study] No se encontró .npz ni .pkl para '{corpus_type}'.")
                _topic_study_state[corpus_type] = []
                return

        # ── 2. Load embeddings ────────────────────────────────────────────────
        d    = np.load(npz_path, allow_pickle=True)
        embs = d["embeddings"].astype(np.float32)
        urls = list(d["urls"])
        print(f"[topic_study] Embeddings: {embs.shape}")

        # ── 3. Load metadata ──────────────────────────────────────────────────
        if wiki_dir and wiki_dir not in sys.path:
            sys.path.insert(0, wiki_dir)
        meta_map: dict = {}
        if os.path.isfile(pkl_path):
            import pickle as _pickle
            with open(pkl_path, "rb") as fh:
                entries = _pickle.load(fh)
            for e in entries:
                meta_map[e.url] = {
                    "url":      e.url,
                    "title":    getattr(e, "title", "") or "",
                    "snippet":  (getattr(e, "description", "") or "")[:200],
                    "category": getattr(e, "category", "") or "",
                }

        # ── 4. Filter (S1 → Top/Computers/) ──────────────────────────────────
        if cat_prefix:
            keep = [i for i, u in enumerate(urls)
                    if meta_map.get(u, {}).get("category", "").startswith(cat_prefix)]
            embs = embs[keep]
            urls = [urls[i] for i in keep]
            print(f"[topic_study] Filtrado a {cat_prefix}: {len(urls)} docs")

        # ── 5. Build text list for BERTopic ──────────────────────────────────
        texts = []
        for u in urls:
            m = meta_map.get(u, {})
            texts.append(f"{m.get('title','')}. {m.get('snippet','')}")

        # ── 6. BERTopic ───────────────────────────────────────────────────────
        from bertopic import BERTopic
        from umap import UMAP
        from hdbscan import HDBSCAN
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.cluster import KMeans
        from collections import Counter as _Counter

        n_docs = len(texts)
        # min_cluster_size proporcional al corpus para evitar micro-clusters
        # apunta a ~15-25 tópicos naturales de HDBSCAN
        min_cs = max(15, n_docs // 40)

        corpus_umap_params = {
            "paper":             dict(umap_n_neighbors=20, umap_n_components=5),
            "wikipedia":         dict(umap_n_neighbors=15, umap_n_components=5),
            "dataviz_wikipedia": dict(umap_n_neighbors=12, umap_n_components=5),
        }
        umap_p = corpus_umap_params.get(corpus_type, dict(umap_n_neighbors=15, umap_n_components=5))

        umap_model = UMAP(
            n_components = umap_p["umap_n_components"],
            n_neighbors  = umap_p["umap_n_neighbors"],
            min_dist     = 0.0,
            metric       = "cosine",
            random_state = 42,
        )
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_cs,
            min_samples=max(1, min_cs // 4),
            metric="euclidean", cluster_selection_method="eom",
            prediction_data=True,
        )
        vectorizer = CountVectorizer(stop_words="english", min_df=2,
                                     ngram_range=(1,2), max_features=10_000)
        topic_model = BERTopic(
            umap_model=umap_model, hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer,
            calculate_probabilities=False, verbose=False,
        )
        print(f"[topic_study] Corriendo BERTopic ({n_docs} docs, min_cluster_size={min_cs})...")
        topic_labels, _ = topic_model.fit_transform(texts, embeddings=embs)

        # ── Post-process 1: redistribuir outliers (-1) al tópico más cercano ──
        n_outliers = sum(1 for l in topic_labels if l == -1)
        if n_outliers > 0 and len(set(topic_labels) - {-1}) > 0:
            try:
                topic_labels = list(topic_model.reduce_outliers(
                    texts, topic_labels, strategy="embeddings", embeddings=embs))
                print(f"[topic_study] {n_outliers} outliers redistribuidos")
            except Exception as _oe:
                print(f"[topic_study] reduce_outliers falló: {_oe}")

        # ── Post-process 2: dividir tópicos sobredimensionados con K-Means ──
        valid = [l for l in topic_labels if l != -1]
        if valid:
            label_counts = _Counter(valid)
            avg_size     = sum(label_counts.values()) / len(label_counts)
            target_size  = max(avg_size, n_docs / 15)   # apunta a ~15 tópicos proporcionales
            max_label    = max(label_counts.keys())
            topic_labels = list(topic_labels)

            for tid, count in list(label_counts.items()):
                if count < target_size * 2.5:
                    continue
                n_sub = min(int(round(count / target_size)), 5)
                if n_sub < 2:
                    continue
                idx      = [i for i, l in enumerate(topic_labels) if l == tid]
                sub_embs = embs[idx]
                km       = KMeans(n_clusters=n_sub, random_state=42, n_init=10)
                sub_lbs  = km.fit_predict(sub_embs)
                for sub_id in range(1, n_sub):
                    max_label += 1
                    new_tid = max_label
                    for loc, sl in enumerate(sub_lbs):
                        if sl == sub_id:
                            topic_labels[idx[loc]] = new_tid
                print(f"[topic_study] Tópico {tid} ({count} docs) dividido en {n_sub} (target ~{int(target_size)})")

        # ── Derivar keywords para todos los tópicos (incluyendo subtópicos) ──
        topic_kws: dict = {}
        try:
            vocab    = np.array(vectorizer.get_feature_names_out())
            doc_term = vectorizer.transform(texts)
            for tid in set(t for t in topic_labels if t != -1):
                t_idx      = [i for i, l in enumerate(topic_labels) if l == tid]
                word_sums  = np.asarray(doc_term[t_idx].sum(axis=0)).flatten()
                top_i      = np.argsort(word_sums)[::-1][:10]
                topic_kws[tid] = [vocab[i] for i in top_i if word_sums[i] > 0]
        except Exception as _ke:
            print(f"[topic_study] Error derivando keywords: {_ke}")

        # ── 7. Build results ──────────────────────────────────────────────────
        topic_ids = sorted(set(topic_labels))
        if -1 in topic_ids:
            topic_ids.remove(-1)

        # Pre-build tid → all_urls mapping for fast URL lookups later
        tid_urls: dict = {}
        for i, (url, tlabel) in enumerate(zip(urls, topic_labels)):
            if tlabel != -1:
                tid_urls.setdefault(tlabel, []).append(url)

        results = []
        for tid in topic_ids:
            doc_idx = [i for i, t in enumerate(topic_labels) if t == tid]
            if not doc_idx:
                continue
            keywords = topic_kws.get(tid, [])

            doc_embs = embs[doc_idx]
            centroid = doc_embs.mean(axis=0)
            sims     = _cosine_sim(centroid, doc_embs)
            top_k    = min(5, len(doc_idx))
            rep_local = np.argsort(sims)[::-1][:top_k].tolist()
            rep_idx   = [doc_idx[i] for i in rep_local]

            # Category distribution
            cat_counts: dict = {}
            for i in doc_idx:
                cat = meta_map.get(urls[i], {}).get("category", "")
                parts = cat.split("/")
                short = "/".join(parts[-2:]) if len(parts) >= 2 else cat
                cat_counts[short] = cat_counts.get(short, 0) + 1
            top_cats = sorted(cat_counts.items(), key=lambda x: -x[1])[:5]

            rep_docs = []
            for i in rep_idx:
                m = meta_map.get(urls[i], {"url": urls[i], "title": "", "snippet": ""})
                rep_docs.append({"url": m["url"], "title": m["title"], "snippet": m.get("snippet","")})

            results.append({
                "name":           _make_readable_name(keywords, top_cats),
                "keywords":       keywords[:8],
                "n_docs":         len(doc_idx),
                "rep_docs":       rep_docs,
                "top_categories": [{"cat": c, "n": n} for c, n in top_cats],
                "_centroid":      centroid,   # numpy array — removed before JSON response
                "_tid":           tid,        # BERTopic label — removed before JSON response
            })

        results.sort(key=lambda r: -r["n_docs"])
        for i, r in enumerate(results, 1):
            r["rank"] = i

        # Build URL → result_idx map (0-indexed into sorted results list)
        url_map: dict = {}
        for ridx, r in enumerate(results):
            tid = r.get("_tid")
            for url in tid_urls.get(tid, []):
                url_map[url] = ridx
        _topic_study_url_map[corpus_type] = url_map

        print(f"[topic_study] Listo: {len(results)} tópicos para '{corpus_type}'. URL map: {len(url_map)} docs.")

        # Guardar los centroides separadamente para el endpoint de halos semánticos
        with _topic_study_lock:
            _topic_study_state[corpus_type] = results

        _topic_study_centroids[corpus_type] = np.stack(
            [r["_centroid"] for r in results]
        ).astype(np.float32)
        # Persistir centroides y resultados en disco (permite recarga rápida tras restart)
        try:
            _c_path = os.path.join(CACHE_DIR, f"topic_centroids_{corpus_type}.npy")
            _s_path = os.path.join(CACHE_DIR, f"topic_state_{corpus_type}.json")
            np.save(_c_path, _topic_study_centroids[corpus_type])
            # Guardar state sin _centroid/_tid (ya limpios después del pop abajo)
            import copy as _copy
            _results_clean = _copy.deepcopy(results)
            for _r in _results_clean:
                _r.pop("_centroid", None)
                _r.pop("_tid", None)
            with open(_s_path, "w", encoding="utf-8") as _sf:
                import json as _json
                _json.dump(_results_clean, _sf, ensure_ascii=False)
            # Persistir url_map (necesario para corpus_counts en proto viz y para no-devueltos pie)
            _u_path = os.path.join(CACHE_DIR, f"topic_url_map_{corpus_type}.json")
            with open(_u_path, "w", encoding="utf-8") as _uf:
                _json.dump(url_map, _uf, ensure_ascii=False)
            print(f"[topic_study] Centroides + url_map persistidos en {_c_path}")
            # Pre-computar similitudes de TODOS los docs del corpus (para proto viz con corpus completo)
            _norms_full = np.linalg.norm(embs, axis=1, keepdims=True)
            _embs_norm_full = (embs / np.where(_norms_full > 0, _norms_full, 1.0)).astype(np.float32)
            _cents = _topic_study_centroids[corpus_type]
            _cnorms = np.linalg.norm(_cents, axis=1, keepdims=True)
            _cents_n = (_cents / np.where(_cnorms > 0, _cnorms, 1.0)).astype(np.float32)
            _all_sims = (_embs_norm_full @ _cents_n.T).clip(-1, 1)
            _n_t = len(results)
            _corpus_docs_list = []
            for _ci, _curl in enumerate(urls):
                _cm = meta_map.get(_curl, {})
                _cridx = url_map.get(_curl, int(np.argmax(_all_sims[_ci])))
                _corpus_docs_list.append({
                    "url": _curl,
                    "t":   _cm.get("title", ""),
                    "sn":  _cm.get("snippet", "")[:200],
                    "tid": _cridx,
                    "s":   [round(float(_all_sims[_ci, _cj]), 4) for _cj in range(_n_t)]
                })
            _d_path2 = os.path.join(CACHE_DIR, f"topic_corpus_docs_{corpus_type}.json")
            with open(_d_path2, "w", encoding="utf-8") as _df2:
                _json.dump(_corpus_docs_list, _df2, ensure_ascii=False)
            _e_path2 = os.path.join(CACHE_DIR, f"topic_corpus_embs_{corpus_type}.npy")
            np.save(_e_path2, _embs_norm_full)
            _topic_study_corpus_docs[corpus_type] = _corpus_docs_list
            _topic_study_corpus_embs[corpus_type] = _embs_norm_full
            print(f"[topic_study] Corpus docs ({len(_corpus_docs_list)}) + embs persistidos.")
        except Exception as _pe:
            print(f"[topic_study] Warning: no se pudo persistir en disco: {_pe}")
        # Eliminar centroide y _tid del JSON público (evita respuesta enorme)
        for r in results:
            r.pop("_centroid", None)
            r.pop("_tid", None)
        return  # lock ya liberado arriba

    except Exception as exc:
        import traceback
        print(f"[topic_study] ERROR para '{corpus_type}': {exc}")
        traceback.print_exc()
        _topic_study_state[corpus_type] = "error"


@app.route("/api/topic_study")
def api_topic_study():
    """
    GET /api/topic_study?corpus_type=paper|wikipedia
    Returns:
      { status: "computing" | "ready" | "unavailable", topics: [...] }
    """
    corpus_type = request.args.get("corpus_type", state.get("corpus_type", ""))
    with _topic_study_lock:
        val = _topic_study_state.get(corpus_type)

    if val is None:
        # First request — trigger background computation
        _TS_TYPES = ("paper", "wikipedia", "dataviz_wikipedia",
                     "cg_wikipedia", "oop_wikipedia", "vinos_wikipedia", "autos_wikipedia")
        if corpus_type in _TS_TYPES:
            threading.Thread(target=_run_topic_study, args=(corpus_type,), daemon=True).start()
        return jsonify({"status": "computing", "topics": []})
    if val == "computing":
        return jsonify({"status": "computing", "topics": []})
    if val == "error":
        return jsonify({"status": "unavailable", "topics": []})
    return jsonify({"status": "ready", "topics": val})


def _compute_node_embeddings_for_topics(labels, edges, root, mode, engine):
    """
    Compute per-node embeddings for semantic topic assignment.

    Modes
    -----
    "label"          — SBERT(node_label)
    "neighbors"      — mean SBERT over [node] + direct neighbours
    "relation_aware" — mean SBERT over all triples involving the node
    "path_aware"     — mean SBERT over all root→node path texts
    "hybrid"         — mean SBERT over label + triples + paths
    """
    label_set = set(labels)

    adj:     dict = {lbl: [] for lbl in labels}
    triples: dict = {lbl: [] for lbl in labels}
    fwd:     dict = {}

    for e in edges:
        src  = e.get("src",  "")
        dst  = e.get("dst",  "")
        link = (e.get("link", "") or "").strip()
        if src in label_set and dst not in adj.get(src, []):
            adj[src].append(dst)
        if dst in label_set and src not in adj.get(dst, []):
            adj[dst].append(src)
        triple = f"{src} {link} {dst}" if link else f"{src} {dst}"
        if src in triples: triples[src].append(triple)
        if dst in triples: triples[dst].append(triple)
        fwd.setdefault(src, []).append((link, dst))

    paths: dict = {lbl: [] for lbl in labels}
    if root in label_set:
        visited: set = {root}
        queue = [(root, [root])]
        while queue:
            cur, ptoks = queue.pop(0)
            for lk, nxt in fwd.get(cur, []):
                if nxt in label_set and nxt not in visited:
                    visited.add(nxt)
                    new_toks = ptoks + ([lk] if lk else []) + [nxt]
                    paths[nxt].append(" → ".join(new_toks))
                    queue.append((nxt, new_toks))

    node_texts: dict = {}
    for lbl in labels:
        if mode == "label":
            node_texts[lbl] = [lbl]
        elif mode == "neighbors":
            node_texts[lbl] = [lbl] + adj.get(lbl, [])
        elif mode == "relation_aware":
            node_texts[lbl] = triples.get(lbl) or [lbl]
        elif mode == "path_aware":
            node_texts[lbl] = paths.get(lbl) or [lbl]
        elif mode == "hybrid":
            ts = [lbl] + triples.get(lbl, []) + paths.get(lbl, [])
            node_texts[lbl] = ts or [lbl]
        else:
            node_texts[lbl] = [lbl]

    all_texts = list(dict.fromkeys(t for ts in node_texts.values() for t in ts))
    if not all_texts:
        return {lbl: np.zeros(384, dtype=np.float32) for lbl in labels}

    all_embs = engine.embed(all_texts).astype(np.float32)
    text2emb  = dict(zip(all_texts, all_embs))

    result: dict = {}
    for lbl in labels:
        ts = [t for t in node_texts.get(lbl, [lbl]) if t in text2emb]
        result[lbl] = (np.stack([text2emb[t] for t in ts]).mean(axis=0)
                       if ts else text2emb.get(lbl, np.zeros(384, dtype=np.float32)))
    return result


@app.route("/api/node_topic_colors", methods=["POST"])
def api_node_topic_colors():
    """
    POST /api/node_topic_colors
    Body: {
      "labels":       [str, ...],
      "corpus_type":  "paper" | "wikipedia",
      "mode":         "label" | "neighbors" | "relation_aware" | "path_aware" | "hybrid",
      "edges":        [{"src": str, "link": str, "dst": str}, ...],
      "root":         str
    }
    """
    data        = request.get_json(force=True) or {}
    labels      = data.get("labels",      [])
    corpus_type = data.get("corpus_type", "")
    mode        = data.get("mode",        "label")
    edges       = data.get("edges",       [])
    root        = data.get("root",        "")

    if not labels or corpus_type not in ("paper", "wikipedia", "dataviz_wikipedia", "cg_wikipedia", "oop_wikipedia", "vinos_wikipedia", "autos_wikipedia"):
        return jsonify({"status": "unavailable", "reason": f"corpus '{corpus_type}' no soportado", "results": [], "topics": []})

    with _topic_study_lock:
        topics = _topic_study_state.get(corpus_type)

    if topics == "computing":
        return jsonify({"status": "computing", "results": [], "topics": []})
    if not isinstance(topics, list) or not topics:
        return jsonify({"status": "unavailable", "reason": "topic_study no ejecutado aún (reinicia el servidor y espera)", "results": [], "topics": []})

    centroids = _topic_study_centroids.get(corpus_type)
    if centroids is None or len(centroids) == 0:
        return jsonify({"status": "unavailable", "reason": "centroides no disponibles (el estudio de tópicos se ejecutó con versión antigua — reinicia el servidor)", "results": [], "topics": []})

    engine = state.get("engine")
    if engine is None or not getattr(engine, "_is_fitted", False):
        return jsonify({"status": "unavailable", "reason": "engine SBERT no inicializado", "results": [], "topics": []})

    try:
        node_embs = _compute_node_embeddings_for_topics(labels, edges, root, mode, engine)
        embs      = np.stack([node_embs[lbl] for lbl in labels]).astype(np.float32)

        embs_n  = embs      / (np.linalg.norm(embs,      axis=1, keepdims=True) + 1e-12)
        cent_n  = centroids / (np.linalg.norm(centroids,  axis=1, keepdims=True) + 1e-12)
        sims    = embs_n @ cent_n.T   # (N, T)

        results = []
        for i, label in enumerate(labels):
            top3_idx = np.argsort(sims[i])[::-1][:3].tolist()
            top3 = [
                {
                    "topic_idx":  int(tidx),
                    "topic_name": topics[int(tidx)]["name"] if int(tidx) < len(topics) else f"Tópico {tidx}",
                    "similarity": round(float(sims[i, int(tidx)]), 4),
                }
                for tidx in top3_idx
            ]
            results.append({
                "label":      label,
                "topic_idx":  top3[0]["topic_idx"],   # backward compat: halo usa el primero
                "topic_name": top3[0]["topic_name"],
                "similarity": top3[0]["similarity"],
                "top3":       top3,
            })

        topic_summaries = [
            {"name": t["name"], "keywords": t.get("keywords", [])[:4]}
            for t in topics
        ]
        return jsonify({"status": "ready", "results": results, "topics": topic_summaries})

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "results": [], "topics": [], "error": str(exc)})


@app.route("/api/topic_colors_for_centroids", methods=["POST"])
def topic_colors_for_centroids():
    """Given a list of centroid vectors, returns the closest BERTopic corpus topic for each."""
    data = request.get_json(force=True)
    centroids   = data.get("centroids", [])
    corpus_type = data.get("corpus_type", "")

    if corpus_type not in ("paper", "wikipedia", "dataviz_wikipedia", "cg_wikipedia", "oop_wikipedia", "vinos_wikipedia", "autos_wikipedia"):
        return jsonify({"status": "unavailable", "reason": f"corpus '{corpus_type}' no soportado", "results": []})

    topics          = _topic_study_state.get(corpus_type)
    corpus_centroids = _topic_study_centroids.get(corpus_type)

    if not isinstance(topics, list) or not topics:
        return jsonify({"status": "unavailable", "reason": "topic_study no ejecutado aún", "results": []})

    if corpus_centroids is None or len(corpus_centroids) == 0:
        return jsonify({"status": "unavailable", "reason": "centroides no disponibles", "results": []})

    corpus_norm = corpus_centroids / (np.linalg.norm(corpus_centroids, axis=1, keepdims=True) + 1e-10)

    results = []
    for centroid in centroids:
        if centroid is None:
            results.append(None)
            continue
        vec = np.array(centroid, dtype=np.float32)
        vec_norm = vec / (np.linalg.norm(vec) + 1e-10)
        sims = corpus_norm @ vec_norm
        best_idx  = int(np.argmax(sims))
        best_sim  = float(sims[best_idx])
        t_info    = topics[best_idx] if best_idx < len(topics) else {}
        results.append({
            "topic_idx":  best_idx,
            "topic_name": t_info.get("label", f"Tópico {best_idx}"),
            "similarity": round(best_sim, 4),
        })

    return jsonify({"status": "ready", "results": results})


@app.route("/api/all_topic_sims", methods=["POST"])
def all_topic_sims():
    """
    Para cada vector de centroide enviado, devuelve la similitud coseno con TODOS
    los tópicos del corpus (no solo el mejor). Útil para construir matrices de similitud.

    Body:  { corpus_type: str, centroids: [[float, ...], ...], labels: [str, ...] }
    Returns: {
        status: "ready",
        topics: [{name, keywords, n_docs, rank}, ...],  # ordenados por rank
        results: [
            {label: str, sims: [float, ...], best_idx: int, best_sim: float},
            ...
        ]
    }
    """
    data        = request.get_json(force=True)
    centroids   = data.get("centroids", [])
    corpus_type = data.get("corpus_type", "")
    labels      = data.get("labels", [f"c{i}" for i in range(len(centroids))])

    if corpus_type not in ("paper", "wikipedia", "dataviz_wikipedia", "cg_wikipedia", "oop_wikipedia", "vinos_wikipedia", "autos_wikipedia"):
        return jsonify({"status": "unavailable", "reason": f"corpus '{corpus_type}' no soportado", "results": []})

    topics         = _topic_study_state.get(corpus_type)
    corpus_centroids = _topic_study_centroids.get(corpus_type)

    if not isinstance(topics, list) or not topics:
        return jsonify({"status": "unavailable", "reason": "topic_study no ejecutado aún", "results": []})

    if corpus_centroids is None or len(corpus_centroids) == 0:
        return jsonify({"status": "unavailable", "reason": "centroides no disponibles", "results": []})

    corpus_norm = corpus_centroids / (np.linalg.norm(corpus_centroids, axis=1, keepdims=True) + 1e-10)
    n_topics    = len(corpus_norm)

    results = []
    for i, centroid in enumerate(centroids):
        label = labels[i] if i < len(labels) else f"c{i}"
        if centroid is None:
            results.append({"label": label, "sims": [0.0] * n_topics, "best_idx": 0, "best_sim": 0.0})
            continue
        vec      = np.array(centroid, dtype=np.float32)
        vec_norm = vec / (np.linalg.norm(vec) + 1e-10)
        sims     = (corpus_norm @ vec_norm).tolist()
        best_idx = int(np.argmax(sims))
        results.append({
            "label":    label,
            "sims":     [round(float(s), 4) for s in sims],
            "best_idx": best_idx,
            "best_sim": round(float(sims[best_idx]), 4),
        })

    topic_info = [
        {"name": t.get("name",""), "keywords": t.get("keywords",[])[:5],
         "n_docs": t.get("n_docs", 0), "rank": t.get("rank", 0)}
        for t in topics
    ]

    return jsonify({"status": "ready", "topics": topic_info, "results": results})


def _ensure_url_map(corpus_type: str) -> bool:
    """
    Garantiza que _topic_study_url_map[corpus_type] esté disponible.
    Si falta pero hay centroides, reconstruye desde .npz y guarda en disco.
    Retorna True si url_map queda disponible, False si no.
    """
    if _topic_study_url_map.get(corpus_type) is not None:
        return True  # ya disponible

    topics  = _topic_study_state.get(corpus_type)
    cents   = _topic_study_centroids.get(corpus_type)
    if not isinstance(topics, list) or cents is None:
        return False  # sin datos suficientes

    _TC_NPZ2 = {
        "paper":            (os.path.join(CACHE_DIR, "curlie_embeddings_sbert_5781b0ace4.npz"),
                             os.path.join(PAPER_DATASET_DIR, ".paper_corpus_150_2000.pkl"), "Top/Computers/"),
        "wikipedia":        (os.path.join(CACHE_DIR, "curlie_embeddings_sbert_646085e7b7.npz"),
                             os.path.join(WIKIPEDIA_DIR, ".wiki_corpus_all.pkl"), None),
        "dataviz_wikipedia":(os.path.join(CACHE_DIR, "curlie_embeddings_sbert_0bec156c44.npz"),
                             os.path.join(DATAVIZ_WIKIPEDIA_DIR, ".wiki_corpus_all.pkl"), None),
        "cg_wikipedia":     (os.path.join(CACHE_DIR, "topic_embs_cg_wikipedia.npz"),
                             os.path.join(CG_WIKIPEDIA_DIR, ".wiki_corpus_all.pkl"), None),
        "oop_wikipedia":    (os.path.join(CACHE_DIR, "topic_embs_oop_wikipedia.npz"),
                             os.path.join(OOP_WIKIPEDIA_DIR, ".wiki_corpus_all.pkl"), None),
        "vinos_wikipedia":  (os.path.join(CACHE_DIR, "topic_embs_vinos_wikipedia.npz"),
                             os.path.join(VINOS_WIKIPEDIA_DIR, ".wiki_corpus_all.pkl"), None),
        "autos_wikipedia":  (os.path.join(CACHE_DIR, "topic_embs_autos_wikipedia.npz"),
                             os.path.join(AUTOS_WIKIPEDIA_DIR, ".wiki_corpus_all.pkl"), None),
    }
    ct_info = _TC_NPZ2.get(corpus_type)
    if not ct_info:
        return False
    npz_p, pkl_p, cat_prefix = ct_info
    if not os.path.isfile(npz_p):
        print(f"[ensure_url_map] .npz no encontrado: {npz_p}")
        return False
    try:
        import json as _json_um
        nd    = np.load(npz_p, allow_pickle=True)
        embs  = nd["embeddings"].astype(np.float32)
        urls  = list(nd["urls"])
        if cat_prefix:
            import pickle as _pkl_um
            meta_map: dict = {}
            if os.path.isfile(pkl_p):
                with open(pkl_p, "rb") as fh:
                    for e in _pkl_um.load(fh):
                        meta_map[e.url] = getattr(e, "category", "") or ""
            keep  = [i for i, u in enumerate(urls) if meta_map.get(u, "").startswith(cat_prefix)]
            embs  = embs[keep]
            urls  = [urls[i] for i in keep]
        norms  = np.linalg.norm(embs, axis=1, keepdims=True)
        embs_n = (embs / np.where(norms > 0, norms, 1.0)).astype(np.float32)
        c      = cents.astype(np.float32)
        cn     = np.linalg.norm(c, axis=1, keepdims=True)
        c_n    = (c / np.where(cn > 0, cn, 1.0)).astype(np.float32)
        sims   = (embs_n @ c_n.T).clip(-1, 1)
        url_map = {u: int(np.argmax(sims[i])) for i, u in enumerate(urls)}
        _topic_study_url_map[corpus_type] = url_map
        # Guardar para futuras recargas
        u_path = os.path.join(CACHE_DIR, f"topic_url_map_{corpus_type}.json")
        with open(u_path, "w", encoding="utf-8") as uf:
            _json_um.dump(url_map, uf, ensure_ascii=False)
        print(f"[ensure_url_map] url_map reconstruido y guardado: {len(url_map)} docs para {corpus_type!r}")
        return True
    except Exception as e:
        import traceback as _tb_um
        print(f"[ensure_url_map] Error: {e}\n{_tb_um.format_exc()}")
        return False


@app.route("/api/topic_colors_for_docs", methods=["POST"])
def topic_colors_for_docs():
    """
    Given a list of URL lists (one per result topic), returns for each result topic
    ALL corpus topics present in its documents (not just the closest centroid).

    Body:  { corpus_type: str, doc_url_lists: [[url, ...], [url, ...], ...] }
    Returns: { status: "ready"|"unavailable", results: [[{topic_idx, topic_name, n_docs}, ...], ...] }
    Each inner list is sorted by n_docs desc (most represented corpus topic first).
    """
    data          = request.get_json(force=True)
    doc_url_lists = data.get("doc_url_lists", [])
    corpus_type   = data.get("corpus_type", "")

    url_map = _topic_study_url_map.get(corpus_type)
    topics  = _topic_study_state.get(corpus_type)

    print(f"[topic_colors] corpus={corpus_type!r} url_map={len(url_map) if url_map else 'None'} topics={len(topics) if isinstance(topics,list) else repr(topics)} n_lists={len(doc_url_lists)}")

    # ── Si topics no es lista, intentar disparar el topic study ──
    if not isinstance(topics, list):
        _TOPIC_STUDY_CORPUS_TYPES_TC = (
            "paper", "wikipedia", "dataviz_wikipedia",
            "cg_wikipedia", "oop_wikipedia", "vinos_wikipedia", "autos_wikipedia",
        )
        if corpus_type in _TOPIC_STUDY_CORPUS_TYPES_TC:
            with _topic_study_lock:
                _already_tc = _topic_study_state.get(corpus_type)
            if _already_tc is None:
                print(f"[topic_colors] Disparando topic study para {corpus_type!r}...")
                import threading as _thr_tc
                _thr_tc.Thread(target=_run_topic_study, args=(corpus_type,), daemon=True).start()
        return jsonify({"status": "unavailable", "results": []})

    # ── Fallback: si url_map falta, reconstruir desde .npz + centroides ──
    if url_map is None:
        _ensure_url_map(corpus_type)
        url_map = _topic_study_url_map.get(corpus_type)

    if url_map is None:
        return jsonify({"status": "unavailable", "results": []})

    results = []
    for url_list in doc_url_lists:
        seen: dict = {}  # result_idx → count
        for url in (url_list or []):
            ridx = url_map.get(url)
            if ridx is not None:
                seen[ridx] = seen.get(ridx, 0) + 1
        # Sort by count desc, then by result_idx for stability
        sorted_topics = sorted(seen.items(), key=lambda x: (-x[1], x[0]))
        topic_list = []
        for ridx, cnt in sorted_topics:
            t_info = topics[ridx] if ridx < len(topics) else {}
            topic_list.append({
                "topic_idx":  ridx,
                "topic_name": t_info.get("name", f"Tópico {ridx}"),
                "n_docs":     cnt,
            })
        results.append(topic_list)

    return jsonify({"status": "ready", "results": results})


@app.route("/api/url_topic", methods=["POST"])
def url_topic():
    """
    Para cada URL individual devuelve el tópico del corpus al que pertenece.

    Body:    { corpus_type: str, urls: [str, ...] }
    Returns: { status: "ready"|"unavailable",
               results: { url: { topic_idx: int, topic_name: str } } }
    """
    data        = request.get_json(force=True)
    urls        = data.get("urls", [])
    corpus_type = data.get("corpus_type", "")

    url_map = _topic_study_url_map.get(corpus_type)
    topics  = _topic_study_state.get(corpus_type)

    if url_map is None and isinstance(topics, list):
        _ensure_url_map(corpus_type)
        url_map = _topic_study_url_map.get(corpus_type)

    if url_map is None or not isinstance(topics, list):
        return jsonify({"status": "unavailable", "results": {}})

    out = {}
    for url in urls:
        ridx = url_map.get(url)
        if ridx is not None and ridx < len(topics):
            t_info = topics[ridx]
            out[url] = {"topic_idx": ridx, "topic_name": t_info.get("name", f"Tópico {ridx}")}

    return jsonify({"status": "ready", "results": out})


@app.route("/api/proto_viz_data", methods=["POST"])
def api_proto_viz_data():
    """
    Para la pestaña Prototipos: dado un conjunto de documentos (de la última corrida del extender),
    calcula la similitud coseno de cada doc con TODOS los tópicos del corpus y devuelve la estructura
    necesaria para el gráfico de ejes paralelos (al estilo POLAR_DOCS_FULL_DV7).

    Body: {
        corpus_type: str,
        docs: [{ url, title, snippet, result_topic_idx: int, result_topic_label: str }, ...],
        map_vec: [float, ...] | null,      # optional: concept map centroid vector
        aspects: [[float, ...], ...]  | null  # optional: aspect vectors
    }
    Returns: {
        status: "ready" | "unavailable",
        docs:   [{ tid: int, s: [float, ...], t: title, url, sn: snippet, rt: int, rt_label: str,
                   s_map: float | null, s_aspects: [float, ...] }],
        topics: [{ name: str, keywords: [str, ...] }],
        topic_order: [int, ...],
        edges:  [{ src: int, dst: int, sim: float }],
        has_map_sim: bool,
        n_aspects: int
    }
    """
    import numpy as _np_pv
    data           = request.get_json(force=True) or {}
    corpus_type    = data.get("corpus_type", "")
    docs_in        = data.get("docs", [])
    returned_urls  = data.get("returned_urls", None)  # list[str] | None — usa corpus completo
    map_vec_raw    = data.get("map_vec", None)   # list[float] or null
    aspects_raw    = data.get("aspects", None)   # list[list[float]] or null

    print(f"[proto_viz] corpus={corpus_type!r} returned_urls={len(returned_urls) if returned_urls is not None else 'None'} docs_in={len(docs_in)} map_vec={map_vec_raw is not None} aspects={len(aspects_raw) if aspects_raw else 0}")

    with _topic_study_lock:
        topics    = _topic_study_state.get(corpus_type)
    centroids     = _topic_study_centroids.get(corpus_type)
    url_map       = _topic_study_url_map.get(corpus_type)

    print(f"[proto_viz] topics={'list('+str(len(topics))+')' if isinstance(topics,list) else repr(topics)} centroids={'shape'+str(centroids.shape) if centroids is not None else 'None'} corpus_docs={len(_topic_study_corpus_docs.get(corpus_type,[]))}")

    if not isinstance(topics, list) or not topics:
        reason = "computing" if topics == "computing" else "Ejecutá el topic study primero (disponible para corpus paper / wikipedia / dataviz_wikipedia)"
        print(f"[proto_viz] -> unavailable: {reason}")
        return jsonify({"status": "unavailable", "reason": reason, "docs": [], "topics": [], "topic_order": [], "edges": []})

    if centroids is None or len(centroids) == 0:
        print(f"[proto_viz] -> unavailable: centroides no disponibles")
        return jsonify({"status": "unavailable", "reason": "centroides no disponibles", "docs": [], "topics": [], "topic_order": [], "edges": []})

    engine = state.get("engine")

    # ── Modo corpus completo (returned_urls provisto) ─────────────────────────
    # Si el cache no está en memoria, intentar cargar desde .npz on-the-fly
    if returned_urls is not None and not _topic_study_corpus_docs.get(corpus_type) and centroids is not None:
        print(f"[proto_viz] corpus cache vacío para {corpus_type!r}, intentando cargar desde .npz on-the-fly...")
        try:
            _CT_NPZ = {
                "paper":             (os.path.join(CACHE_DIR, "curlie_embeddings_sbert_5781b0ace4.npz"),
                                      os.path.join(PAPER_DATASET_DIR, ".paper_corpus_150_2000.pkl"), "Top/Computers/"),
                "wikipedia":         (os.path.join(CACHE_DIR, "curlie_embeddings_sbert_646085e7b7.npz"),
                                      os.path.join(WIKIPEDIA_DIR, ".wiki_corpus_all.pkl"), None),
                "dataviz_wikipedia": (os.path.join(CACHE_DIR, "curlie_embeddings_sbert_0bec156c44.npz"),
                                      os.path.join(DATAVIZ_WIKIPEDIA_DIR, ".wiki_corpus_all.pkl"), None),
                "cg_wikipedia":      (os.path.join(CACHE_DIR, "topic_embs_cg_wikipedia.npz"),
                                      os.path.join(CG_WIKIPEDIA_DIR, ".wiki_corpus_all.pkl"), None),
                "oop_wikipedia":     (os.path.join(CACHE_DIR, "topic_embs_oop_wikipedia.npz"),
                                      os.path.join(OOP_WIKIPEDIA_DIR, ".wiki_corpus_all.pkl"), None),
                "vinos_wikipedia":   (os.path.join(CACHE_DIR, "topic_embs_vinos_wikipedia.npz"),
                                      os.path.join(VINOS_WIKIPEDIA_DIR, ".wiki_corpus_all.pkl"), None),
                "autos_wikipedia":   (os.path.join(CACHE_DIR, "topic_embs_autos_wikipedia.npz"),
                                      os.path.join(AUTOS_WIKIPEDIA_DIR, ".wiki_corpus_all.pkl"), None),
            }
            _ct_info = _CT_NPZ.get(corpus_type)
            if _ct_info:
                _npz_p, _pkl_p, _cat_prefix = _ct_info
                if os.path.isfile(_npz_p):
                    import time as _time_pv
                    _t0_pv = _time_pv.time()
                    print(f"[proto_viz] Cargando .npz desde {_npz_p}...")
                    _nd   = _np_pv.load(_npz_p, allow_pickle=True)
                    _embs = _nd["embeddings"].astype(_np_pv.float32)
                    _urls_list = list(_nd["urls"])
                    # Cargar metadata del corpus
                    _meta_map = {}
                    if os.path.isfile(_pkl_p):
                        import pickle as _pkl_mod
                        with open(_pkl_p, "rb") as _fh:
                            _entries_raw = _pkl_mod.load(_fh)
                        for _e in _entries_raw:
                            _meta_map[_e.url] = {
                                "title":    getattr(_e, "title",       "") or "",
                                "snippet":  (getattr(_e, "description","") or "")[:200],
                                "category": getattr(_e, "category",    "") or "",
                            }
                    # Filtrar por prefijo de categoría si aplica (ej: corpus "paper")
                    if _cat_prefix:
                        _keep = [_i for _i, _u in enumerate(_urls_list)
                                 if _meta_map.get(_u, {}).get("category", "").startswith(_cat_prefix)]
                        _embs       = _embs[_keep]
                        _urls_list  = [_urls_list[_i] for _i in _keep]
                        print(f"[proto_viz] Filtrado a {_cat_prefix}: {len(_urls_list)} docs")
                    # Normalizar embeddings
                    _norms_f = _np_pv.linalg.norm(_embs, axis=1, keepdims=True)
                    _embs_n  = (_embs / _np_pv.where(_norms_f > 0, _norms_f, 1.0)).astype(_np_pv.float32)
                    # Calcular similitudes a todos los centroides
                    _cents   = centroids.astype(_np_pv.float32)
                    _cnorms  = _np_pv.linalg.norm(_cents, axis=1, keepdims=True)
                    _cents_n = (_cents / _np_pv.where(_cnorms > 0, _cnorms, 1.0)).astype(_np_pv.float32)
                    _all_sims = (_embs_n @ _cents_n.T).clip(-1, 1)
                    _n_t_local = len(topics)
                    _url_map_local = _topic_study_url_map.get(corpus_type) or {}
                    _corpus_docs_list = []
                    for _ci, _curl in enumerate(_urls_list):
                        _cm    = _meta_map.get(_curl, {})
                        _cridx = _url_map_local.get(_curl, int(_np_pv.argmax(_all_sims[_ci])))
                        _corpus_docs_list.append({
                            "url": _curl,
                            "t":   _cm.get("title",   ""),
                            "sn":  _cm.get("snippet",  ""),
                            "tid": _cridx,
                            "s":   [round(float(_all_sims[_ci, _cj]), 4) for _cj in range(_n_t_local)]
                        })
                    _topic_study_corpus_docs[corpus_type] = _corpus_docs_list
                    _topic_study_corpus_embs[corpus_type] = _embs_n
                    print(f"[proto_viz] Corpus on-the-fly: {len(_corpus_docs_list)} docs listos en {_time_pv.time()-_t0_pv:.1f}s.")
                    # Persistir a disco para evitar recompute en próximos reinicios
                    try:
                        import json as _json_pv
                        _d_path_pv = os.path.join(CACHE_DIR, f"topic_corpus_docs_{corpus_type}.json")
                        _e_path_pv = os.path.join(CACHE_DIR, f"topic_corpus_embs_{corpus_type}.npy")
                        with open(_d_path_pv, "w", encoding="utf-8") as _dfpv:
                            _json_pv.dump(_corpus_docs_list, _dfpv, ensure_ascii=False)
                        np.save(_e_path_pv, _embs_n)
                        print(f"[proto_viz] Corpus docs persistidos en disco: {_d_path_pv}")
                    except Exception as _save_e:
                        print(f"[proto_viz] Warning: no se pudo persistir corpus docs: {_save_e}")
                else:
                    print(f"[proto_viz] .npz no encontrado: {_npz_p}")
            else:
                print(f"[proto_viz] corpus_type {corpus_type!r} no tiene .npz registrado")
        except Exception as _load_e:
            import traceback as _tb
            print(f"[proto_viz] Error cargando corpus on-the-fly: {_load_e}")
            _tb.print_exc()

    use_full_corpus = returned_urls is not None and bool(_topic_study_corpus_docs.get(corpus_type))
    print(f"[proto_viz] use_full_corpus={use_full_corpus} corpus_docs={len(_topic_study_corpus_docs.get(corpus_type,[]))}")

    if returned_urls is not None and not use_full_corpus:
        # Aún sin corpus on-the-fly → fallback a docs_in (solo devueltos)
        print(f"[proto_viz] Sin corpus cache. Fallback a docs_in ({len(docs_in)} docs)")
        if not docs_in:
            print(f"[proto_viz] -> unavailable: sin docs_in y sin corpus cache")
            return jsonify({"status": "unavailable", "reason": "Corpus completo no pre-computado. Re-ejecutá el estudio de tópicos para habilitar esta vista.", "docs": [], "topics": [], "topic_order": [], "edges": []})

    if not docs_in and not use_full_corpus:
        topics_out = [{"name": t.get("name", t.get("label", f"T{i}")), "keywords": t.get("keywords", [])[:5]} for i, t in enumerate(topics)]
        return jsonify({"status": "ready", "docs": [], "topics": topics_out, "topic_order": list(range(len(topics))), "edges": []})

    try:
        n_topics    = len(topics)
        map_sims    = None   # (N,) float or None
        aspect_sims = None   # (N, A) float or None

        if use_full_corpus:
            # ── Camino rápido: usar similitudes pre-computadas del corpus completo ──
            corpus_docs = _topic_study_corpus_docs[corpus_type]
            corpus_embs = _topic_study_corpus_embs.get(corpus_type)  # may be None
            ret_set     = set(returned_urls)

            # Compute right-axis sims if embs cached and vectors provided
            if corpus_embs is not None and (map_vec_raw or (aspects_raw and len(aspects_raw) > 0)):
                if map_vec_raw:
                    try:
                        mv = _np_pv.array(map_vec_raw, dtype=_np_pv.float32)
                        mv /= (_np_pv.linalg.norm(mv) + 1e-12)
                        map_sims = (corpus_embs @ mv).clip(-1, 1)
                    except Exception:
                        map_sims = None
                if aspects_raw and len(aspects_raw) > 0:
                    try:
                        av = _np_pv.array(aspects_raw, dtype=_np_pv.float32)
                        norms_a = _np_pv.linalg.norm(av, axis=1, keepdims=True)
                        av /= _np_pv.where(norms_a > 0, norms_a, 1.0)
                        aspect_sims = (corpus_embs @ av.T).clip(-1, 1)  # (N_corpus, A)
                    except Exception:
                        aspect_sims = None

            out_docs = []
            for i, cd in enumerate(corpus_docs):
                is_ret = cd["url"] in ret_set
                out_docs.append({
                    "tid":       cd["tid"],
                    "s":         cd["s"],
                    "t":         cd["t"],
                    "url":       cd["url"],
                    "sn":        cd["sn"],
                    "rt":        0 if is_ret else -1,
                    "rt_label":  "",
                    "s_map":     round(float(map_sims[i]), 4) if map_sims is not None else None,
                    "s_aspects": [round(float(aspect_sims[i, j]), 4) for j in range(aspect_sims.shape[1])] if aspect_sims is not None else [],
                })
        else:
            # ── Camino clásico: re-embeder los docs provistos por el frontend ──
            if engine is None or not getattr(engine, "_is_fitted", False):
                return jsonify({"status": "unavailable", "reason": "engine SBERT no inicializado", "docs": [], "topics": [], "topic_order": [], "edges": []})

            texts    = [(d.get("title","")+" "+d.get("snippet","")).strip() or d.get("url","doc") for d in docs_in]
            doc_embs = engine.embed(texts)  # (N, 384), L2-normalised

            if map_vec_raw:
                try:
                    mv = _np_pv.array(map_vec_raw, dtype=_np_pv.float32)
                    mv /= (_np_pv.linalg.norm(mv) + 1e-12)
                    map_sims = (doc_embs.astype(_np_pv.float32) @ mv).clip(-1, 1)
                except Exception:
                    map_sims = None
            if aspects_raw and len(aspects_raw) > 0:
                try:
                    av = _np_pv.array(aspects_raw, dtype=_np_pv.float32)
                    norms_a = _np_pv.linalg.norm(av, axis=1, keepdims=True)
                    av /= _np_pv.where(norms_a > 0, norms_a, 1.0)
                    aspect_sims = (doc_embs.astype(_np_pv.float32) @ av.T).clip(-1, 1)
                except Exception:
                    aspect_sims = None

            cent_arr = _np_pv.array(centroids, dtype=_np_pv.float32)
            norms    = _np_pv.linalg.norm(cent_arr, axis=1, keepdims=True)
            cent_n   = cent_arr / (_np_pv.where(norms > 0, norms, 1.0))
            sims     = doc_embs.astype(_np_pv.float32) @ cent_n.T  # (N, T)

            out_docs = []
            for i, d in enumerate(docs_in):
                sims_row = [round(float(sims[i, j]), 4) for j in range(n_topics)]
                tid = url_map.get(d.get("url",""), int(_np_pv.argmax(sims[i]))) if url_map else int(_np_pv.argmax(sims[i]))
                out_docs.append({
                    "tid":       tid,
                    "s":         sims_row,
                    "t":         d.get("title",""),
                    "url":       d.get("url",""),
                    "sn":        d.get("snippet","")[:200],
                    "rt":        d.get("result_topic_idx", 0),
                    "rt_label":  d.get("result_topic_label",""),
                    "s_map":     round(float(map_sims[i]), 4) if map_sims is not None else None,
                    "s_aspects": [round(float(aspect_sims[i, j]), 4) for j in range(aspect_sims.shape[1])] if aspect_sims is not None else [],
                })

        topics_out = [
            {"name": t.get("name", t.get("label", f"T{i}")), "keywords": t.get("keywords", [])[:5]}
            for i, t in enumerate(topics)
        ]

        # Topic order: most represented first (by number of corpus docs)
        freq = [0] * n_topics
        for d in out_docs:
            if 0 <= d["tid"] < n_topics:
                freq[d["tid"]] += 1
        topic_order = sorted(range(n_topics), key=lambda x: -freq[x])

        # Inter-topic edges (shared between both code paths)
        _cent_arr2 = _np_pv.array(centroids, dtype=_np_pv.float32)
        _cnorms2   = _np_pv.linalg.norm(_cent_arr2, axis=1, keepdims=True)
        _cent_n2   = _cent_arr2 / _np_pv.where(_cnorms2 > 0, _cnorms2, 1.0)
        sim_matrix = _cent_n2 @ _cent_n2.T
        edges = []
        for ii in range(n_topics):
            for jj in range(ii + 1, n_topics):
                edges.append({"src": ii, "dst": jj, "sim": round(float(sim_matrix[ii, jj]), 4)})
        edges.sort(key=lambda e: -e["sim"])
        edges = edges[: min(n_topics * 2, len(edges))]

        # Corpus counts per topic
        # En modo full-corpus out_docs ya tiene TODOS los docs del corpus → contamos directo
        corpus_counts = {}
        if use_full_corpus:
            for _d in out_docs:
                _k = str(_d["tid"])
                corpus_counts[_k] = corpus_counts.get(_k, 0) + 1
        elif url_map:
            for _url, _tid in url_map.items():
                corpus_counts[str(_tid)] = corpus_counts.get(str(_tid), 0) + 1
        else:
            for _d in out_docs:
                _k = str(_d["tid"])
                corpus_counts[_k] = corpus_counts.get(_k, 0) + 1

        ret_count  = sum(1 for d in out_docs if d.get("rt", -1) >= 0)
        nret_count = len(out_docs) - ret_count
        print(f"[proto_viz] -> ready: {len(out_docs)} docs (ret={ret_count} noret={nret_count}), {len(topics_out)} topics, corpus_counts keys={len(corpus_counts)}, has_map_sim={map_sims is not None}, n_aspects={int(aspect_sims.shape[1]) if aspect_sims is not None else 0}")
        return jsonify({
            "status":        "ready",
            "docs":          out_docs,
            "topics":        topics_out,
            "topic_order":   topic_order,
            "edges":         edges,
            "corpus_counts": corpus_counts,
            "has_map_sim":   map_sims is not None,
            "n_aspects":     int(aspect_sims.shape[1]) if aspect_sims is not None else 0,
        })

    except Exception:
        tb = traceback.format_exc()
        print(tb)
        return jsonify({"status": "error", "reason": tb, "docs": [], "topics": [], "topic_order": [], "edges": [], "corpus_counts": {}})


if __name__ == "__main__":
    print("=" * 60)
    print("  EXTENDER UI — http://localhost:5000")
    print("  Corpus y modelo SBERT cargando en background...")
    print("=" * 60)
    app.run(debug=False, port=5000, threaded=True)
