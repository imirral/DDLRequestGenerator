import re
import json
import torch
import networkx as nx

from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForTokenClassification
from sklearn.model_selection import train_test_split
from seqeval.metrics import classification_report as seq_classification_report
from razdel import sentenize


class Token:

    def __init__(self, text, start, stop):
        self.text = text
        self.start = start
        self.stop = stop


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
    G = nx.DiGraph()

    for edu_id, edu_text in edus:
        G.add_node(edu_id, text=edu_text)

    for r in rels:
        edu1, edu2, relation_type = r
        G.add_edge(edu1, edu2, relation=relation_type)
    return G


def tokenize(text):
    tokens = []
    for match in re.finditer(r"\w+|\S", text):
        token_text = match.group()
        token_start = match.start()
        token_end = match.end()
        tokens.append(Token(text=token_text, start=token_start, stop=token_end))
    return tokens


class DataPreprocessor:
    def __init__(self):
        self.file_path = "D:/Magistracy/FQW/DDLRequestGenerator/data/imirral.jsonl"
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

            discourse_graph = build_discourse_tree(text)

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

            for rel in relations:
                from_id = rel['from_id']
                to_id = rel['to_id']
                rel_type = rel['type'].lower()
                relations_list.append((from_id, to_id, rel_type))

            preprocessed_data.append({
                'text': text,
                'discourse_graph': discourse_graph,
                'entities': entity_labels,
                'attributes': attribute_labels,
                'types': type_labels,
                'relations': relations_list,
                'id_to_entity': id_to_entity
            })
        return preprocessed_data


def prepare_data_for_entity_extraction(preprocessed_data):
    prepared_data = []
    for item in preprocessed_data:
        text = item['text']
        tokens = tokenize(text)
        words = [t.text for t in tokens]
        word_labels = ['O'] * len(words)

        char_to_word = {}
        for idx, t in enumerate(tokens):
            for char_pos in range(t.start, t.stop):
                char_to_word[char_pos] = idx

        all_entities = item['entities'] + item['attributes'] + item['types']
        for entity in all_entities:
            entity_id, start, end, label = entity
            for char_pos in range(start, end):
                word_idx = char_to_word.get(char_pos)
                if word_idx is not None:
                    current_label = word_labels[word_idx]
                    if current_label == 'O':
                        prefix = 'B'
                    elif current_label[2:] != label:
                        prefix = 'B'
                    else:
                        prefix = 'I'
                    word_labels[word_idx] = f'{prefix}-{label}'

        prepared_data.append({
            'words': words,
            'labels': word_labels
        })

    return prepared_data


class EntityDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_map = {
            'O': 0,
            'B-ENTITY': 1, 'I-ENTITY': 2,
            'B-ATTRIBUTE': 3, 'I-ATTRIBUTE': 4,
            'B-VARCHAR': 5, 'I-VARCHAR': 6,
            'B-INT': 7, 'I-INT': 8,
            'B-DATE': 9, 'I-DATE': 10,
            'B-DECIMAL': 11, 'I-DECIMAL': 12,
            'B-BOOLEAN': 13, 'I-BOOLEAN': 14
        }
        self.id2label = {id_val: label for label, id_val in self.label_map.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        words = item['words']
        labels = item['labels']

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_special_tokens_mask=True
        )
        aligned_labels = self.align_labels(labels, encoding)

        return {
            'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(aligned_labels, dtype=torch.long)
        }

    def align_labels(self, labels, encoding):
        aligned_labels = []
        word_ids = encoding.word_ids()
        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)
            else:
                aligned_labels.append(self.label_map[labels[word_idx]])
        return aligned_labels


class EntityExtractionModel(nn.Module):
    def __init__(self, num_labels):
        super(EntityExtractionModel, self).__init__()
        self.model = BertForTokenClassification.from_pretrained(
            "DeepPavlov/rubert-base-cased", num_labels=num_labels
        )
        self.model.config.id2label = {
            id_val: label for id_val, label in enumerate(self.model.config.id2label)
        }
        self.model.config.label2id = {
            label: id_val for id_val, label in self.model.config.id2label.items()
        }

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs.loss, outputs.logits


def train_entity_extraction_model(model, train_dataloader, val_dataloader, num_epochs, device, label_map):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    model.to(device)

    id2label = {id_val: label for label, id_val in label_map.items()}

    for epoch in range(num_epochs):
        total_loss = 0
        model.train()
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            loss, logits = model(input_ids, attention_mask, labels=labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_loss:.4f}")
        evaluate(model, val_dataloader, device, id2label)


def evaluate(model, dataloader, device, id2label):
    model.eval()
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']

            outputs = model.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu()

            for i in range(len(labels)):
                seq_true = []
                seq_pred = []
                for j in range(len(labels[i])):
                    if labels[i][j] != -100:
                        seq_true.append(id2label[labels[i][j].item()])
                        seq_pred.append(id2label[preds[i][j].item()])
                true_labels.append(seq_true)
                pred_labels.append(seq_pred)

    report = seq_classification_report(true_labels, pred_labels, digits=4, zero_division=0)
    print("Validation Metrics:")
    print(report)
    model.train()


def save_model_and_tokenizer(model, tokenizer, output_dir, label_map):
    model.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    with open(f'{output_dir}/label_map.json', 'w', encoding='utf-8') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    print(f"Model and tokenizer saved to {output_dir}")


def load_model_and_tokenizer(output_dir, num_labels):
    tokenizer = BertTokenizerFast.from_pretrained(output_dir)
    loaded_model = BertForTokenClassification.from_pretrained(output_dir, num_labels=num_labels)
    model = EntityExtractionModel(num_labels=num_labels)
    model.model = loaded_model

    with open(f'{output_dir}/label_map.json', 'r', encoding='utf-8') as f:
        label_map = json.load(f)

    return model, tokenizer, label_map


def predict(text, model, tokenizer, label_map, device):
    model.to(device)
    model.eval()

    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model.model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1).cpu().numpy()[0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    offset_mappings = encoding['offset_mapping'][0].cpu().numpy()

    id2label = {id_val: label for label, id_val in label_map.items()}

    entities = []
    current_entity = None

    for idx, pred in enumerate(predictions):
        label = id2label.get(pred, 'O')
        start, end = offset_mappings[idx]
        word = tokens[idx]

        if label.startswith('B-'):
            if current_entity is not None:
                entities.append(current_entity)
            current_entity = {
                'label': label[2:],
                'text': word,
                'start': int(start),
                'end': int(end)
            }
        elif label.startswith('I-') and current_entity is not None and current_entity['label'] == label[2:]:
            current_entity['text'] += " " + word
            current_entity['end'] = int(end)
        else:
            if current_entity is not None:
                entities.append(current_entity)
                current_entity = None

    if current_entity:
        entities.append(current_entity)

    return entities


if __name__ == '__main__':
    preprocessor = DataPreprocessor()
    preprocessed_data = preprocessor.preprocessed_data

    entity_data = prepare_data_for_entity_extraction(preprocessed_data)

    train_data, val_data = train_test_split(entity_data, test_size=0.2, random_state=42)

    tokenizer = BertTokenizerFast.from_pretrained("DeepPavlov/rubert-base-cased")

    train_dataset = EntityDataset(train_data, tokenizer)
    val_dataset = EntityDataset(val_data, tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_map = train_dataset.label_map

    model = EntityExtractionModel(num_labels=len(label_map))
    train_entity_extraction_model(
        model,
        train_dataloader,
        val_dataloader,
        num_epochs=30,
        device=device,
        label_map=label_map
    )

    output_dir = 'D:/Magistracy/FQW/DDLRequestGenerator/saved_models/entity_model_with_dt'
    save_model_and_tokenizer(model, tokenizer, output_dir, label_map)
