import json


class DataPreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.raw_data = self.load_jsonl()
        self.preprocessed_data = self.preprocess_data()

    def load_jsonl(self):
        data = []
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def preprocess_data(self):
        preprocessed_data = []
        for item in self.raw_data:
            text = item['text']
            entities = sorted(item['entities'], key=lambda x: x['start_offset'])

            entity_labels = []
            attribute_labels = []
            type_labels = []

            for entity in entities:
                start_offset = entity['start_offset']
                end_offset = entity['end_offset']
                label = entity['label']

                if label == 'ENTITY':
                    entity_labels.append((start_offset, end_offset, 'ENTITY'))
                elif label == 'ATTRIBUTE':
                    attribute_labels.append((start_offset, end_offset, 'ATTRIBUTE'))
                elif label in ['VARCHAR', 'INT', 'DATE']:
                    type_labels.append((start_offset, end_offset, label))

            preprocessed_data.append({
                'text': text,
                'entities': entity_labels,
                'attributes': attribute_labels,
                'types': type_labels
            })

        return preprocessed_data
