#!/usr/bin/env python3
"""
generate_viz_data.py — genera JS desde datos reales.
"""
import json, math, re, subprocess, tempfile, os
import numpy as np
from pathlib import Path

BASE    = Path('/sessions/stoic-gracious-galileo/mnt/Proyecto_mineria_de_texto')
RESULTS = BASE / 'Pruebas_y_mapas_conceptuales_generados'
OUT     = BASE

# ─────────── helpers ────────────────────────────────────────
def cos_sim(a, b):
    a, b = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 1e-10 and nb > 1e-10 else 0.0

def to_angle(s):
    return round(math.acos(max(-1.0, min(1.0, float(s)))) * 180 / math.pi, 1)

def write_js_vars(path, var_dict):
    with open(path, 'w', encoding='utf-8') as f:
        for name, data in var_dict.items():
            f.write(f'window.{name}={json.dumps(data, separators=(",",":"), ensure_ascii=False)};\n')
    sz = path.stat().st_size
    print(f'  -> {path.name}  ({sz//1024} KB,  {len(var_dict)} vars)')

def eval_js_to_json(js_snippet, timeout=60):
    """Evaluate a JS expression via node, return parsed JSON."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False, encoding='utf-8') as tf:
        tf.write(f'process.stdout.write(JSON.stringify({js_snippet}));')
        tname = tf.name
    try:
        r = subprocess.run(['node', tname], capture_output=True, text=True, timeout=timeout)
        if r.returncode == 0:
            return json.loads(r.stdout)
        else:
            raise RuntimeError(r.stderr[:200])
    finally:
        try: os.unlink(tname)
        except: pass

def extract_js_var(content, var_pattern):
    """Extract a JS object/array value from file content matching 'var_pattern =' ."""
    m = re.search(rf'{var_pattern}\s*=\s*(\[|\{{)', content)
    if not m: return None
    start = m.start(1)
    opener = content[start]; closer = ']' if opener == '[' else '}'
    depth = 0
    for i, ch in enumerate(content[start:], start=start):
        if ch == opener: depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0: return content[start:i+1]
    return None

# ─────────── read helpers ────────────────────────────────────
def read_emb_dv7(ep):
    f = RESULTS/'DataViz'/('embeddings.json' if ep==1 else f'embeddings_{ep-1}.json')
    return json.load(open(f))

def read_tl_dv7(ep):
    f = RESULTS/'DataViz'/('timeline.json' if ep==1 else f'timeline_{ep-1}.json')
    return json.load(open(f))

def read_emb_s1(ep):
    return json.load(open(RESULTS/'Software'/f'embeddings_{ep}.json'))

def read_tl_s1(ep):
    return json.load(open(RESULTS/'Software'/f'software_timeline_{ep}.json'))

def read_emb_s2(ep):
    f = RESULTS/'Software2'/('embeddings.json' if ep==1 else f'embeddings_{ep-1}.json')
    return json.load(open(f))

def read_tl_s2(ep):
    f = RESULTS/'Software2'/('timeline.json' if ep==1 else f'timeline_{ep-1}.json')
    return json.load(open(f))

def parse_returned_topics(tl):
    return [{'id': t.get('topic_id'), 'label': t.get('label',''),
             'kw': (t.get('keywords') or [])[:6], 'n_docs': t.get('doc_count') or 0}
            for t in (tl.get('topics') or [])]

def get_cm(tl):
    cm = tl.get('concept_map') or {}
    concepts = cm.get('concepts') or []
    edges = [{'src': r.get('src',''), 'link': r.get('link',''), 'dst': r.get('dst','')}
             for r in (cm.get('relations') or [])]
    return concepts, edges

def dest_from_text(text, concepts):
    return max((c for c in concepts if text.endswith(c)), key=len, default='')

def compute_node_sims(emb_data, concepts, centroids_list):
    """
    centroids_list: list of {id, centroid (np.array)} in order.
    Returns nodes with sims to each centroid.
    """
    text_emb = {t['text']: np.array(t['embedding'], dtype=np.float32)
                for t in emb_data.get('texts', [])}
    nodes = []
    for c in concepts:
        ev = text_emb.get(c)
        if ev is None:
            for txt, ev2 in text_emb.items():
                if txt.endswith(c): ev = ev2; break
        if ev is None:
            nodes.append({'node': c, 'topic_idx': -1, 'topic_id': -1,
                          'best_sim': 0, 'sims': []})
            continue
        sims = [round(cos_sim(ev, ct['centroid']), 4) for ct in centroids_list]
        if not sims:
            nodes.append({'node': c, 'topic_idx': -1, 'topic_id': -1,
                          'best_sim': 0, 'sims': []})
            continue
        best_i = int(np.argmax(sims))
        nodes.append({'node': c, 'topic_idx': best_i,
                      'topic_id': centroids_list[best_i]['id'],
                      'best_sim': sims[best_i], 'sims': sims})
    return nodes

def compute_ang(emb_data, concepts, ep):
    map_vec = np.array(emb_data['map_vec']['embedding'], dtype=np.float32)
    return [{'e': ep, 't': t['text'],
             'a': to_angle(cos_sim(np.array(t['embedding'], dtype=np.float32), map_vec)),
             'w': round(float(t.get('weight', 1.0)), 4),
             'n': t.get('index', 0),
             'd': dest_from_text(t['text'], concepts)}
            for t in emb_data.get('texts', [])]

def compute_asp(emb_data, ep):
    map_vec = np.array(emb_data['map_vec']['embedding'], dtype=np.float32)
    return [{'e': ep, 't': f'Aspecto {i}',
             'a': to_angle(cos_sim(np.array(a['embedding'], dtype=np.float32), map_vec))}
            for i, a in enumerate(emb_data.get('aspects', []))]

def compute_aspect_nodes(emb_data, concepts, encoder, ep):
    text_emb = {t['text']: np.array(t['embedding'], dtype=np.float32)
                for t in emb_data.get('texts', [])}
    aspects_data = []
    for ai, a in enumerate(emb_data.get('aspects', [])):
        av = np.array(a['embedding'], dtype=np.float32)
        node_sims = []
        for c in concepts:
            ev = text_emb.get(c)
            if ev is None:
                for txt, ev2 in text_emb.items():
                    if txt.endswith(c): ev = ev2; break
            if ev is not None:
                node_sims.append({'node': c, 'sim': round(cos_sim(ev, av), 4)})
        node_sims.sort(key=lambda x: -x['sim'])
        aspects_data.append({'idx': ai, 'top_nodes': node_sims[:5]})
    return {'e': ep, 'encoder': encoder, 'aspects': aspects_data}

def compute_topic_sims_entry(ep, emb_data, topics_with_cents, encoder):
    """topics_with_cents: list of {id,label,kw,n_docs,centroid(np array or None)}"""
    map_vec = np.array(emb_data['map_vec']['embedding'], dtype=np.float32)
    map_text = emb_data.get('map_vec', {}).get('text') or ''
    n_asp = emb_data.get('n_aspects', len(emb_data.get('aspects', [])))
    asp_vecs = [np.array(a['embedding'], dtype=np.float32)
                for a in emb_data.get('aspects', [])]
    topics_out = []
    for t in topics_with_cents:
        cent = t.get('centroid')
        if cent is not None:
            sm = round(cos_sim(cent, map_vec), 4)
            sa = [round(cos_sim(cent, av), 4) for av in asp_vecs[1:]]
        else:
            sm, sa = 0.0, []
        topics_out.append({'id': t['id'], 'label': t['label'], 'kw': t['kw'],
                           'n_docs': t['n_docs'], 'sim_map': sm, 'sim_aspects': sa})
    return {'e': ep, 'map_text': map_text, 'n_aspects': n_asp,
            'encoder': encoder, 'topics': topics_out}

# ─────────── Load centroids ──────────────────────────────────
print('Loading DV7 corpus centroids (.npy)...')
dv7_npy = np.load(BASE / 'Pipeline_EXTENDER' / '.cache' / 'topic_centroids_dataviz_wikipedia.npy')
dv7_topic_state = json.load(open(BASE / 'Pipeline_EXTENDER' / '.cache' / 'topic_state_dataviz_wikipedia.json'))
# DV7 corpus topics: indexed 0-9, each with {id, label, kw, n_docs, centroid}
dv7_corpus_topics = []
for i, t in enumerate(dv7_topic_state):
    kws = (t.get('keywords') or [])[:4]
    label = t.get('name', '_'.join(kws[:2]))
    dv7_corpus_topics.append({'id': i, 'label': label, 'kw': kws,
                               'n_docs': t.get('n_docs', 0),
                               'centroid': dv7_npy[i]})
# For node_sims: centroids_list format
dv7_centroids_list = [{'id': t['id'], 'centroid': t['centroid']} for t in dv7_corpus_topics]

print('Loading S2 centroids from epoch_vectors.js...')
with open(BASE / 'epoch_vectors.js') as f:
    ev_content = f.read()
s2_cent_js = extract_js_var(ev_content, r'window\.EPOCH_VECTORS\.S2_TOPIC_CENTROIDS')
s2_centroids_raw = json.loads(s2_cent_js) if s2_cent_js else {}
# keyed by BERTopic ID string: '6','14','22','87','164','166','174','179','182','188'
s2_centroids_by_id = {int(k): np.array(v, dtype=np.float32) for k, v in s2_centroids_raw.items()}

print('Loading topic_study_s1/s2...')
topic_study_s1 = json.load(open(BASE / 'topic_study_s1.json'))
topic_study_s2 = json.load(open(BASE / 'topic_study_s2.json'))

# ─────────── DV7 (8 epochs) ─────────────────────────────────
print('\n=== DV7 (8 epochs) ===')
topic_map_dv7, topic_sims_dv7 = [], []
# DV7 corpus topics constant (same every epoch):
dv7_const_topics = [{'id': t['id'], 'label': t['label'], 'kw': t['kw'], 'n_docs': t['n_docs']}
                    for t in dv7_corpus_topics]

for ep in range(1, 9):
    print(f'  ep{ep}...')
    emb = read_emb_dv7(ep)
    tl  = read_tl_dv7(ep)
    enc = emb.get('encoder') or tl.get('encoder') or ''
    concepts, edges = get_cm(tl)

    nodes = compute_node_sims(emb, concepts, dv7_centroids_list)
    topic_map_dv7.append({'e': ep, 'encoder': enc, 'topics': dv7_const_topics,
                          'nodes': nodes, 'concepts': concepts, 'edges': edges})

    # TOPIC_SIMS: all 10 corpus topics + their sims to map_vec/aspects
    twc = [dict(t) for t in dv7_corpus_topics]  # includes centroid
    topic_sims_dv7.append(compute_topic_sims_entry(ep, emb, twc, enc))

write_js_vars(OUT / 'viz_data_dv7.js', {
    'VIZ_TOPIC_MAP_DV7': topic_map_dv7,
    'VIZ_TOPIC_SIMS_DV7': topic_sims_dv7,
})

# ─────────── S1 (12 epochs) ─────────────────────────────────
print('\n=== S1 (12 epochs) ===')
topic_map_s1, topic_sims_s1 = [], []
ang_s1, asp_s1, anodes_s1 = [], [], []

for ep in range(1, 13):
    print(f'  ep{ep}...')
    emb = read_emb_s1(ep)
    tl  = read_tl_s1(ep)
    enc = emb.get('encoder') or tl.get('encoder') or ''
    returned = parse_returned_topics(tl)
    concepts, edges = get_cm(tl)

    # S1 nodes: no centroids (lost run), empty sims
    nodes = [{'node': c, 'topic_idx': -1, 'topic_id': -1, 'best_sim': 0, 'sims': []}
             for c in concepts]
    topic_map_s1.append({'e': ep, 'encoder': enc, 'topics': returned,
                         'nodes': nodes, 'concepts': concepts, 'edges': edges})

    # TOPIC_SIMS_S1: also no centroids for returned S1 topics
    twc = [{**t, 'centroid': None} for t in returned]
    topic_sims_s1.append(compute_topic_sims_entry(ep, emb, twc, enc))

    ang_s1.extend(compute_ang(emb, concepts, ep))
    asp_s1.extend(compute_asp(emb, ep))
    anodes_s1.append(compute_aspect_nodes(emb, concepts, enc, ep))

ctopics_s1 = [{'rank': t.get('rank'), 'name': t.get('name'),
               'keywords': (t.get('keywords') or [])[:8], 'n_docs': t.get('n_docs'),
               'rep_docs': (t.get('rep_docs') or [])[:3],
               'top_categories': (t.get('top_categories') or [])[:5]}
              for t in topic_study_s1]

write_js_vars(OUT / 'viz_data_s1.js', {
    'VIZ_TOPIC_MAP_S1': topic_map_s1,
    'VIZ_TOPIC_SIMS_S1': topic_sims_s1,
    'VIZ_ANG_S1': ang_s1,
    'VIZ_ASP_S1': asp_s1,
    'VIZ_ASPECT_NODES_S1': anodes_s1,
    'VIZ_CTOPICS_S1': ctopics_s1,
})

# ─────────── S2 (7 epochs) ──────────────────────────────────
print('\n=== S2 (7 epochs) ===')
topic_map_s2, topic_sims_s2 = [], []
ang_s2, asp_s2, anodes_s2 = [], [], []

for ep in range(1, 8):
    print(f'  ep{ep}...')
    emb = read_emb_s2(ep)
    tl  = read_tl_s2(ep)
    enc = emb.get('encoder') or tl.get('encoder') or ''
    returned = parse_returned_topics(tl)
    concepts, edges = get_cm(tl)

    # S2 nodes: use S2_TOPIC_CENTROIDS keyed by returned topic BERTopic ID
    ret_cents_list = [{'id': t['id'], 'centroid': s2_centroids_by_id.get(t['id'])}
                      for t in returned]
    # filter out None centroids for sims computation
    valid_cents = [c for c in ret_cents_list if c['centroid'] is not None]
    nodes = compute_node_sims(emb, concepts, valid_cents)
    topic_map_s2.append({'e': ep, 'encoder': enc, 'topics': returned,
                         'nodes': nodes, 'concepts': concepts, 'edges': edges})

    # TOPIC_SIMS_S2: returned topics + sims from S2_TOPIC_CENTROIDS
    twc = [{**t, 'centroid': s2_centroids_by_id.get(t['id'])} for t in returned]
    topic_sims_s2.append(compute_topic_sims_entry(ep, emb, twc, enc))

    ang_s2.extend(compute_ang(emb, concepts, ep))
    asp_s2.extend(compute_asp(emb, ep))
    anodes_s2.append(compute_aspect_nodes(emb, concepts, enc, ep))

ctopics_s2 = [{'rank': t.get('rank'), 'name': t.get('name'),
               'keywords': (t.get('keywords') or [])[:8], 'n_docs': t.get('n_docs'),
               'rep_docs': (t.get('rep_docs') or [])[:3],
               'top_categories': (t.get('top_categories') or [])[:5]}
              for t in topic_study_s2]

write_js_vars(OUT / 'viz_data_s2.js', {
    'VIZ_TOPIC_MAP_S2': topic_map_s2,
    'VIZ_TOPIC_SIMS_S2': topic_sims_s2,
    'VIZ_ANG_S2': ang_s2,
    'VIZ_ASP_S2': asp_s2,
    'VIZ_ASPECT_NODES_S2': anodes_s2,
    'VIZ_CTOPICS_S2': ctopics_s2,
})

# ─────────── Static extraction from HTML ────────────────────
print('\n=== Extracting static data from HTML ===')
with open(BASE / 'Pipeline_EXTENDER' / 'Visualizaciones e interfaz' / 'extender_visualizaciones.html', encoding='utf-8') as f:
    html = f.read()

def extract_const_via_node(html, const_name):
    js_val = extract_js_var(html, f'const {const_name}')
    if not js_val:
        print(f'  WARNING: {const_name} not found')
        return None
    try:
        data = eval_js_to_json(js_val)
        print(f'  OK {const_name}: {len(data)} items')
        return data
    except Exception as e:
        print(f'  ERROR {const_name}: {e}')
        return None

static = {}
for name in ['GLOBAL_CORPUS_S1', 'GLOBAL_CORPUS_S2', 'CORPUS_TREE_S1', 'CORPUS_TREE_S2',
             'TOPIC_ASPECT_SIMS_S1']:
    d = extract_const_via_node(html, name)
    if d is not None:
        static[f'VIZ_{name}'] = d

with open(OUT / 'viz_static_data.js', 'w', encoding='utf-8') as f:
    f.write('// Datos extraidos del HTML — corridas originales no replicables.\n\n')
    for name, data in static.items():
        f.write(f'window.{name}={json.dumps(data, separators=(",",":"), ensure_ascii=False)};\n')
sz = (OUT / 'viz_static_data.js').stat().st_size
print(f'  -> viz_static_data.js  ({sz//1024} KB,  {len(static)} vars)')

print('\n=== Done ===')
