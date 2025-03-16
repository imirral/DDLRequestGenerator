import json


def load_jsonl():
    file_path = "D:/Magistracy/FQW/DDLRequestGenerator/data/imirral.jsonl"
    data = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    return data


def preprocess_data():
    raw_data = load_jsonl()
    preprocessed_data = []

    for item in raw_data:
        text = item['text']
        entities = sorted(item['entities'], key=lambda x: x['start_offset'])
        relations = item.get('relations', [])
        entity_labels = []
        attribute_labels = []
        type_labels = []
        relations_list = []

        id_to_entity = {}

        for entity in entities:
            entity_id = entity['id']
            start_offset = entity['start_offset']
            end_offset = entity['end_offset']
            label = entity['label']

            id_to_entity[entity_id] = (entity_id, start_offset, end_offset, label)

            if label == 'ENTITY':
                entity_labels.append((entity_id, start_offset, end_offset, 'ENTITY'))
            elif label == 'ATTRIBUTE':
                attribute_labels.append((entity_id, start_offset, end_offset, 'ATTRIBUTE'))
            elif label in ['VARCHAR', 'INT', 'DATE', 'DECIMAL', 'BOOLEAN']:
                type_labels.append((entity_id, start_offset, end_offset, label))

        for relation in relations:
            from_id = relation['from_id']
            to_id = relation['to_id']
            rel_type = relation['type'].lower()
            relations_list.append((from_id, to_id, rel_type))

        preprocessed_data.append({
            'text': text,
            'entities': entity_labels,
            'attributes': attribute_labels,
            'types': type_labels,
            'relations': relations_list,
            'id_to_entity': id_to_entity
        })

    return preprocessed_data
