# EXTENDER — Pipeline de descubrimiento de tópicos

---

## Cómo ejecutar

Ejecutá `abrir_extender.bat`. Instala Flask si no está, levanta el servidor y abre el navegador en `http://localhost:5000` automáticamente.

La primera vez que se trabaja con un corpus debe calcular los embeddings del mismo y tardará mucho más en ejecutarse.


---

## Archivos principales

| Archivo | Qué hace |
|---|---|
| `extender_bertopic.py` | Bloques base compartidos: `ConceptMap`, `ConceptNode`, `EmbeddingEngine` (SBERT 384-dim) y utilidades de texto. |
| `concept_map_encoder.py` | 4 estrategias para convertir un mapa conceptual en vectores de búsqueda: `ConceptMapEncoder` (nodos), `RelationAwareEncoder` (tripletas), `PathAwareEncoder` (caminos), `HybridEncoder` (híbrido por profundidad). |
| `topic_explorer.py` | Pipeline completo en una sola pasada: embeddings del corpus → filtrado por similitud coseno → BERTopic → selección de nivel jerárquico → tópicos rankeados. Acepta cualquier encoder via el parámetro `encoder_class`. |
| `corpus_loader.py` | Loaders de corpus: `PaperDatasetCorpus` (dataset ODP/Curlie etiquetado) y `WikipediaCorpus` (artículos de Wikipedia desde `articles.jsonl`). |
| `abrir_extender.bat` | Script de arranque para Windows. |

---

## Carpetas

### `Datasets/`
Datasets listos para usar con el pipeline. Cada subcarpeta corresponde a un dominio:

- `dataset_paper/` — dataset principal ODP/Curlie etiquetado (papers).
- `dataset_autos/`, `dataset_vinos/`, `dataset_oop/`, `dataset_dataviz/`, `dataset_computacion_grafica/` — datasets temáticos adicionales.
- Dentro de cada dataset hay una subcarpeta `Wikipedia/` con el corpus de Wikipedia del dominio (generado con `wikipedia_crawler.py`).
- `Datos_y_Configuracion/` — configuraciones y archivos de soporte compartidos.

### `Documentación/`
Documentación técnica interactiva del pipeline. `Documentación.html` muestra la arquitectura, el código fuente de cada módulo y la referencia de clases y métodos.

### `Pruebas_y_mapas_conceptuales_generados/`
Mapas conceptuales de prueba y resultados de experimentos anteriores en formato JSON.

### `Visualizaciones e interfaz/`
Visualizaciones auxiliares e interfaz web alternativa (`Interfaz_Web/`). También contiene `explorer_extensions.py` (versión archivada, no se usa en el pipeline activo).

---

## Dependencias

```
flask
bertopic
umap-learn
hdbscan
sentence-transformers
scikit-learn
numpy
```

Instalación:
```bash
pip install flask bertopic umap-learn hdbscan sentence-transformers scikit-learn numpy
```
