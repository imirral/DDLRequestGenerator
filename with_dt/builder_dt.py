import re
import networkx as nx

from razdel import sentenize


def segment_text_into_edus(text):
    sentences = list(sentenize(text))

    edus = []
    edu_id = 0

    for sent in sentences:
        sent_text = text[sent.start:sent.stop].strip()
        if not sent_text:
            continue

        chunks = re.split(r'(,|—|:)', sent_text)
        for chunk in chunks:
            c = chunk.strip()
            if c and c not in [',', '—', ':']:
                edus.append((edu_id, c))
                edu_id += 1

    return edus


def detect_relations(edus):
    markers_map = {
        'однако': 'Contrast',
        'потому что': 'Cause',
        'например': 'Elaboration',
        'то есть': 'Explanation'
    }

    relations = []

    for i in range(len(edus) - 1):
        edu_id1, text1 = edus[i]
        edu_id2, text2 = edus[i + 1]

        for marker, r_type in markers_map.items():
            if marker in text1.lower() or marker in text2.lower():
                relations.append((edu_id1, edu_id2, r_type))

    return relations


def build_discourse_tree(text):
    edus = segment_text_into_edus(text)
    rels = detect_relations(edus)

    graph = nx.DiGraph()

    for edu_id, edu_text in edus:
        graph.add_node(edu_id, text=edu_text)

    for r in rels:
        edu1, edu2, relation_type = r
        graph.add_edge(edu1, edu2, relation=relation_type)

    return graph
