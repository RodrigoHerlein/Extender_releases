

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ===========================================================================
# Domain seed texts
# ===========================================================================

_DOMAIN_SEEDS: dict[str, list[str]] = {
    "space": [
        "planetary science geology astrobiology exobiology",
        "spacecraft orbit trajectory mission launch rocket propulsion",
        "Mars Venus Jupiter Saturn planetary exploration probe",
        "NASA ESA JAXA space agency astronaut cosmonaut crew",
        "telescope observatory spectroscopy remote sensing instrument",
        "satellite orbital mechanics rendezvous docking manoeuvre",
        "interplanetary transfer Hohmann spacecraft delta-v",
        "rover lander orbiter flyby probe descent ascent",
        "sample collection curation analysis laboratory geology",
        "surface terrain crater basin volcano regolith dust",
        "atmosphere climate weather dust storm pressure temperature",
        "radiation magnetic field solar wind plasma environment",
        "landing site selection hazard terrain mapping reconnaissance",
        "entry descent landing parachute retrorocket airbag",
        "biosignature organic chemistry life detection abiotic",
        "mineralogy petrology stratigraphy rock geochemistry",
        "photovoltaic solar panel power radioisotope thermoelectric",
        "communication antenna relay uplink downlink bandwidth",
        "navigation guidance control autonomous software system",
        "human spaceflight life support habitat crew deep space",
        "technology development prototype test validation engineering",
        "planetary protection contamination protocol sterilisation",
        "subsurface drilling borehole coring ice permafrost",
        "airborne balloon drone aerial platform reconnaissance",
        "sample return capsule re-entry Earth retrieval curation",
    ],
    "moon": [
        "lunar surface regolith soil impact crater ejecta",
        "Apollo program astronaut moonwalk EVA sample return geology",
        "mare basalt highland anorthosite mineralogy petrology",
        "tidal locking synchronous rotation libration orbit",
        "water ice permanently shadowed region polar crater volatile",
        "selenology lunar science topography selenographic",
        "ISRU resource utilisation helium-3 oxygen hydrogen",
        "Lunar Reconnaissance Orbiter LADEE GRAIL gravity",
        "moonquake seismometer internal core mantle crust",
        "solar wind implantation volatile regolith gardening",
        "Artemis Gateway cislunar crewed landing exploration",
        "telescope lunar farside radio quiet astronomy",
    ],
    "mars": [
        "Mars surface geology geochemistry mineralogy basalt",
        "Curiosity Perseverance rover instrument MOXIE sample tube",
        "Mars Reconnaissance Orbiter MAVEN atmosphere climate",
        "Ingenuity helicopter drone aerial flight altitude",
        "ancient lake river delta sediment habitability",
        "methane atmospheric chemistry photochemistry oxidant",
        "dust storm global regional opacity solar panel",
        "InSight seismometer marsquake interior structure",
        "Phobos Deimos moon orbit tidal evolution",
        "terraforming colonisation human habitat pressure",
    ],
    "biology": [
        "organism species taxonomy evolution genetics mutation",
        "cell membrane protein DNA RNA chromosome gene expression",
        "ecosystem ecology biodiversity conservation habitat niche",
        "photosynthesis respiration metabolism enzyme catalyst",
        "virus bacteria pathogen infection immune system antibody",
        "anatomy physiology neural brain neuroscience synapse",
        "fossil paleontology evolutionary adaptation selection",
        "bioinformatics genomics proteomics sequencing phylogeny",
        "microbiology microbiome gut flora culture medium",
        "clinical trial drug therapy diagnosis treatment",
    ],
    "computer": [
        "algorithm data structure complexity sorting search",
        "machine learning neural network deep learning training",
        "database query SQL NoSQL retrieval information system",
        "network protocol TCP internet distributed computing cloud",
        "software engineering architecture design pattern testing",
        "artificial intelligence knowledge representation reasoning",
        "natural language processing text mining semantic parsing",
        "computer vision image recognition object detection",
        "cybersecurity cryptography encryption vulnerability",
        "operating system kernel scheduler memory management",
    ],
    "earth": [
        "geophysics seismology earthquake fault tectonic plate",
        "climate change carbon dioxide greenhouse warming",
        "ocean current thermohaline circulation salinity",
        "atmospheric science troposphere stratosphere ozone",
        "remote sensing satellite land cover vegetation index",
        "hydrology watershed river basin flood precipitation",
        "soil erosion sediment geomorphology landscape",
        "mineralogy crystallography ore deposit mining",
        "geodesy GPS reference frame coordinate survey",
        "environmental impact pollution remediation ecology",
    ],
}


def _get_domain_seeds(concept_map_title: str, concept_map_root: str = "") -> list[str]:
    """
    Return domain seed sentences matching the map's subject domain.

    Searches the title and root for keywords from _DOMAIN_SEEDS.
    Multiple domains may match (e.g. 'Earth Science and Space').
    Always returns at least the 'space' seeds as a default.
    """
    combined = (concept_map_title + " " + concept_map_root).lower()
    seeds: list[str] = []
    matched: set[str] = set()
    for key, texts in _DOMAIN_SEEDS.items():
        if key in combined and key not in matched:
            seeds.extend(texts)
            matched.add(key)
    # Default fallback: always include space seeds for academic maps
    if "space" not in matched:
        seeds.extend(_DOMAIN_SEEDS["space"])
    return seeds

# ---------------------------------------------------------------------------
# Stopwords
# ---------------------------------------------------------------------------

_BASE_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "this", "that",
    "these", "those", "it", "its", "as", "up", "out", "about", "into",
    "through", "during", "before", "after", "above", "below", "between",
    "each", "than", "so", "if", "not", "no", "nor", "yet", "both", "either",
    "also", "just", "more", "most", "such", "only", "own", "same", "then",
    "when", "where", "which", "while", "who", "how", "all", "any", "few",
    "other", "some", "what", "their", "they", "them", "we", "our", "you",
    "your", "he", "she", "his", "her", "my", "me", "him", "us", "new",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "http", "www", "com", "org", "net", "edu", "gov",
    "including", "provides", "offers", "based", "information", "services",
    "company", "products", "manufacturer", "manufacturers", "supplier",
    "suppliers", "located", "includes", "featuring", "featuring", "various",
    "general", "website", "site", "page", "online",
})

_NUMERIC_PATTERN = re.compile(r"^\d+$")


def tokenise(text: str, stopwords: frozenset[str] = _BASE_STOPWORDS) -> list[str]:
    """Tokeniza un texto en palabras alfanuméricas de 3+ caracteres, filtrando stopwords y números.

    Args:
        text: Texto de entrada a tokenizar.
        stopwords: Conjunto de palabras a ignorar.

    Returns:
        Lista de tokens limpios en minúsculas.
    """
    tokens = re.findall(r"[a-zA-Z0-9]{3,}", text.lower())
    return [
        t for t in tokens
        if t not in stopwords
        and not _NUMERIC_PATTERN.match(t)   # strip numeric IDs
    ]


def clean_text(text: str) -> str:
    """Elimina tokens puramente numéricos de un texto.

    Args:
        text: Texto de entrada.

    Returns:
        Texto con los tokens numéricos eliminados.
    """
    tokens = text.split()
    clean = [t for t in tokens if not _NUMERIC_PATTERN.match(t.strip(".,;:()"))]
    return " ".join(clean)


# ===========================================================================
# 1. ConceptMap
# ===========================================================================


@dataclass
class ConceptNode:
    """Nodo del mapa conceptual con sus vecinos y etiquetas de arcos.

    Attributes:
        label: Etiqueta del concepto.
        neighbors: Lista de etiquetas de nodos adyacentes.
        link_labels: Lista de etiquetas de los arcos hacia los vecinos.
    """
    label: str
    neighbors: list[str] = field(default_factory=list)
    link_labels: list[str] = field(default_factory=list)


class ConceptMap:
    """Grafo de mapa conceptual con nodos (conceptos) y arcos (relaciones etiquetadas).

    Attributes:
        title: Título del mapa.
        nodes: Diccionario de etiqueta → ConceptNode.
        edges: Lista de tripletas (src, link, dst).
    """

    def __init__(self, title: str = ""):
        """Inicializa un mapa conceptual vacío.

        Args:
            title: Título descriptivo del mapa.
        """
        self.title = title
        self.nodes: dict[str, ConceptNode] = {}
        self.edges: list[tuple[str, str, str]] = []   # (src, link, dst)
        self._root: Optional[str] = None

    # ------------------------------------------------------------------
    # Graph construction

    def add_concept(self, label: str) -> None:
        """Agrega un concepto al mapa si no existe todavía.

        Args:
            label: Etiqueta del concepto a agregar.
        """
        label = label.strip()
        if label not in self.nodes:
            self.nodes[label] = ConceptNode(label=label)

    def add_relation(self, src: str, link: str, dst: str) -> None:
        """Agrega una relación dirigida entre dos conceptos.

        Crea automáticamente los nodos src y dst si no existen y
        registra la adyacencia bidireccional en ambos nodos.

        Args:
            src: Etiqueta del concepto origen.
            link: Etiqueta del arco (verbo o frase de relación).
            dst: Etiqueta del concepto destino.
        """
        for concept in (src, dst):
            self.add_concept(concept)
        self.nodes[src].neighbors.append(dst)
        self.nodes[src].link_labels.append(link)
        self.nodes[dst].neighbors.append(src)
        self.edges.append((src, link, dst))

    def set_root(self, label: str) -> None:
        """Mark label as the topical root of the map."""
        self.add_concept(label)
        self._root = label

    # ------------------------------------------------------------------
    # Topological analysis

    def topological_weights(self) -> dict[str, float]:
        """Calcula pesos topológicos para cada nodo combinando grado y distancia a la raíz.

        El peso final es el promedio de:
          - deg_score: grado normalizado del nodo.
          - dist_score: proximidad a la raíz (1 − distancia normalizada).

        Returns:
            Diccionario {etiqueta: peso} con valores en [0, 1].
        """
        if not self.nodes:
            return {}

        degree: dict[str, int] = defaultdict(int)
        for src, _, dst in self.edges:
            degree[src] += 1
            degree[dst] += 1
        max_deg = max(degree.values()) if degree else 1

        root = self._root or next(iter(self.nodes))
        dist: dict[str, float] = {root: 0.0}
        queue = [root]
        while queue:
            cur = queue.pop(0)
            for nb in self.nodes[cur].neighbors:
                if nb not in dist:
                    dist[nb] = dist[cur] + 1.0
                    queue.append(nb)
        max_dist = max(dist.values(), default=1.0) or 1.0

        weights: dict[str, float] = {}
        for label in self.nodes:
            deg_score  = degree.get(label, 0) / max_deg
            dist_score = 1.0 - dist.get(label, max_dist) / (max_dist + 1)
            weights[label] = 0.5 * deg_score + 0.5 * dist_score
        return weights

    def term_weights(
        self,
        focus_concepts: Optional[list[str]] = None,
        focus_multiplier: float = 2.0,
    ) -> dict[str, float]:
        """Calcula pesos normalizados para cada token del mapa, basados en topología.

        Acumula el peso topológico del nodo en sus tokens de etiqueta (peso completo)
        y en los tokens de sus arcos (30 % del peso). Los conceptos en focus_concepts
        reciben un multiplicador adicional.

        Args:
            focus_concepts: Conceptos a enfatizar.
            focus_multiplier: Factor multiplicador para conceptos en foco.

        Returns:
            Diccionario {token: peso_normalizado}.
        """
        topo      = self.topological_weights()
        weights:  dict[str, float] = defaultdict(float)
        focus_set = set(focus_concepts or [])

        for label, node in self.nodes.items():
            base = topo.get(label, 0.1)
            if label in focus_set:
                base *= focus_multiplier
            for tok in tokenise(label):
                weights[tok] += base
            for link in node.link_labels:
                for tok in tokenise(link):
                    weights[tok] += base * 0.3

        total = sum(weights.values()) or 1.0
        return {t: w / total for t, w in weights.items()}

    def as_pseudo_documents(self) -> list[str]:
        """Genera una lista de pseudo-documentos a partir del mapa para BERTopic.

        Incluye el título, cada etiqueta de nodo, cada tripleta "src link dst"
        y las frases semilla de dominio correspondientes al mapa.

        Returns:
            Lista de strings que representan el contenido del mapa.
        """
        docs: list[str] = []
        if self.title:
            docs.append(self.title)
        for label in self.nodes:
            docs.append(label)
        for src, link, dst in self.edges:
            docs.append(f"{src} {link} {dst}")
        root = self._root or ""
        docs.extend(_get_domain_seeds(self.title, root))
        return docs

    def domain_context_tokens(self, top_n: int = 6) -> list[str]:
        """Retorna los tokens más representativos del mapa según sus pesos.

        Args:
            top_n: Cantidad de tokens a retornar.

        Returns:
            Lista de tokens ordenados por peso descendente.
        """
        weights = self.term_weights()
        ranked = sorted(weights.items(), key=lambda x: -x[1])
        return [t for t, _ in ranked[:top_n]]

    def concept_bigrams(self, top_n: int = 6) -> list[str]:
        """Retorna los conceptos multi-palabra mejor rankeados topológicamente.

        Solo considera etiquetas que contengan 2+ tokens al tokenizar.

        Args:
            top_n: Número máximo de conceptos a retornar.

        Returns:
            Lista de etiquetas multi-palabra ordenadas por peso topológico.
        """
        topo   = self.topological_weights()
        scored = []
        for label in self.nodes:
            parts = tokenise(label)
            if len(parts) >= 2:           # only multi-word concepts
                w = topo.get(label, 0.0)
                scored.append((w, label))
        scored.sort(reverse=True)
        return [label for _, label in scored[:top_n]]

    # ------------------------------------------------------------------
    # Serialisation

    @classmethod
    def from_dict(cls, data: dict) -> "ConceptMap":
        """Construye un ConceptMap a partir de un diccionario serializado.

        Args:
            data: Diccionario con claves opcionales "title", "concepts",
                  "relations" (lista de {src, link, dst}) y "root".

        Returns:
            Instancia de ConceptMap poblada.
        """
        cm = cls(title=data.get("title", ""))
        for c in data.get("concepts", []):
            cm.add_concept(c)
        for r in data.get("relations", []):
            cm.add_relation(r["src"], r["link"], r["dst"])
        if "root" in data:
            cm.set_root(data["root"])
        return cm

    @classmethod
    def from_json_file(cls, path: str) -> "ConceptMap":
        """Carga un ConceptMap desde un archivo JSON en disco.

        Args:
            path: Ruta al archivo JSON serializado.

        Returns:
            Instancia de ConceptMap cargada desde el archivo.
        """
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


# ===========================================================================
# 2. EmbeddingEngine
# ===========================================================================


class EmbeddingEngine:
    """Motor de embeddings basado en SBERT (all-MiniLM-L6-v2, 384 dim).

    Attributes:
        random_state: Semilla para reproducibilidad.
    """

    def __init__(self, random_state: int = 42):
        """Inicializa el motor de embeddings.

        Args:
            random_state: Semilla aleatoria para reproducibilidad.
        """
        self.random_state = random_state
        self._sbert       = None
        self._is_fitted   = False

    # ------------------------------------------------------------------
    # Public API

    def fit(self, texts: list[str] = ()) -> "EmbeddingEngine":
        """Carga el modelo SBERT en memoria. Debe llamarse antes de embed().

        Args:
            texts: Ignorado; solo presente por compatibilidad de interfaz.

        Returns:
            La instancia actual (para encadenamiento).
        """
        self._fit_sbert()
        self._is_fitted = True
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        """Genera embeddings normalizados para una lista de textos.

        Args:
            texts: Lista de strings a embeddear.

        Returns:
            Array de shape (N, 384) con embeddings normalizados en float64.

        Raises:
            RuntimeError: Si fit() no fue llamado previamente.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before embed().")
        if not texts:
            return np.zeros((0, self.dim))
        return self._embed_sbert(texts)

    def fit_embed(self, texts: list[str]) -> np.ndarray:
        """Atajo que llama a fit() y luego embed() en una sola llamada.

        Args:
            texts: Lista de strings a embeddear.

        Returns:
            Array de shape (N, 384) con embeddings normalizados.
        """
        return self.fit(texts).embed(texts)

    def refit(self, all_texts: list[str] = ()) -> "EmbeddingEngine":
        """No-op: los pesos de SBERT son fijos, no requiere re-entrenamiento.

        Args:
            all_texts: Ignorado.

        Returns:
            La instancia actual.
        """
        return self

    @property
    def dim(self) -> int:
        return 384

    # ------------------------------------------------------------------
    # Private

    def _fit_sbert(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            self._sbert = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as exc:
            raise RuntimeError(
                "sentence-transformers unavailable or model download failed."
            ) from exc

    def _embed_sbert(self, texts: list[str]) -> np.ndarray:
        return self._sbert.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).astype(np.float64)
