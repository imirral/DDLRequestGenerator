import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
import torch
import torch.nn as nn

from razdel import tokenize
from transformers import BertForTokenClassification, BertTokenizerFast
from torch.utils.data import Dataset, DataLoader
from without_dt.preprocessing import preprocess_data
from sklearn.model_selection import train_test_split
from seqeval.metrics import classification_report as seq_classification_report


def prepare_data_for_entity_extraction(preprocessed_data):
    prepared_data = []

    for item in preprocessed_data:
        text = item['text']
        tokens = list(tokenize(text))
        words = [token.text for token in tokens]

        word_labels = ['O'] * len(words)

        # Словарь {индекс символа: индекс токена}
        # К какому слову (токену) относится символ?
        char_to_word = {}
        for idx, token in enumerate(tokens):
            for char_pos in range(token.start, token.stop):
                char_to_word[char_pos] = idx

        # Присвоение метки каждому токену
        # B - beginning
        # I - inside
        # O - outside
        for entity in item['entities'] + item['attributes'] + item['types']:
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


# Класс-обертка для DataLoader
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
        self.id2label = {id: label for label, id in self.label_map.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        words = item['words']
        labels = item['labels']

        # Подготовка входных данных для модели BERT
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_special_tokens_mask=True
        )

        labels = self.align_labels(labels, encoding)

        return {
            'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

    # Разметка subword-токенов
    def align_labels(self, labels, encoding):
        aligned_labels = []
        word_ids = encoding.word_ids()
        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)  # Позиция для игнорирования
            else:
                aligned_labels.append(self.label_map[labels[word_idx]])
        return aligned_labels


class EntityModel(nn.Module):
    def __init__(self, num_labels):
        super(EntityModel, self).__init__()

        # num_labels - общее число классов (включая B/I для каждого типа и "O")
        # Инициализация модели rubert-base-cased
        self.model = BertForTokenClassification.from_pretrained(
            "DeepPavlov/rubert-base-cased", num_labels=num_labels)

        self.model.config.id2label = {id: label for label, id in enumerate(self.model.config.id2label)}
        self.model.config.label2id = {label: id for id, label in self.model.config.id2label.items()}

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs.loss, outputs.logits


def train_entity_model(model, train_dataloader, val_dataloader, num_epochs, device, label_map):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)  # lr - скорость обучения
    model.to(device)

    id2label = {id: label for label, id in label_map.items()}

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
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            outputs = model.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1).cpu()

            for i in range(len(labels)):
                true_label = []
                pred_label = []

                for j in range(len(labels[i])):
                    if labels[i][j] != -100:
                        true_label.append(id2label[labels[i][j].item()])
                        pred_label.append(id2label[predictions[i][j].item()])

                true_labels.append(true_label)
                pred_labels.append(pred_label)

    report = seq_classification_report(true_labels, pred_labels, digits=4, zero_division=0)
    print("Validation Metrics:")
    print(report)
    model.train()


def save_model_and_tokenizer(model, tokenizer, output_dir, label_map):
    model.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    with open(f'{output_dir}/label_map.json', 'w') as f:
        json.dump(label_map, f)
    print(f"Model and tokenizer saved to {output_dir}")


def load_model_and_tokenizer(output_dir, num_labels):
    tokenizer = BertTokenizerFast.from_pretrained(output_dir)
    loaded_model = BertForTokenClassification.from_pretrained(output_dir, num_labels=num_labels)
    model = EntityModel(num_labels=num_labels)
    model.model = loaded_model
    with open(f'{output_dir}/label_map.json', 'r') as f:
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

    id2label = {id: label for label, id in label_map.items()}

    entities = []
    current_entity = None
    for idx, pred in enumerate(predictions):
        label = id2label.get(pred, 'O')
        if label != 'O':
            word = tokens[idx]
            start, end = offset_mappings[idx]
            if label.startswith('I-'):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {'label': label[2:], 'text': word, 'start': int(start), 'end': int(end)}
            elif label.startswith('B-') and current_entity:
                current_entity['text'] += tokenizer.convert_tokens_to_string([word])
                current_entity['end'] = int(end)
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = None
    if current_entity:
        entities.append(current_entity)
    return entities


if __name__ == '__main__':
    preprocessed_data = preprocess_data()
    entity_data = prepare_data_for_entity_extraction(preprocessed_data)

    train_data, val_data = train_test_split(entity_data, test_size=0.2, random_state=42)

    tokenizer = BertTokenizerFast.from_pretrained("DeepPavlov/rubert-base-cased")

    train_dataset = EntityDataset(train_data, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)

    val_dataset = EntityDataset(val_data, tokenizer)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_map = train_dataset.label_map
    model = EntityModel(num_labels=len(label_map))
    train_entity_model(model, train_dataloader, val_dataloader, num_epochs=20, device=device,
                       label_map=label_map)

    output_dir = 'D:/Magistracy/FQW/DDLRequestGenerator/saved_models/entity_model_without_dt'
    save_model_and_tokenizer(model, tokenizer, output_dir, label_map)

    # test_text = ("""
    #     Пациент заключает договор на лечение в заданном отделении больницы. У каждого пациента есть лечащий врач.
    #     Пациент может оплачивать свой договор за счёт страховой компании. Обследование пациента проводит не только лечащий врач.
    #     О пациенте должна содержаться следующая информация: фамилия пациента, его категория, номер паспорта, номер страхового полиса, дата поступления, гражданство.
    #     О враче должна содержаться следующая информация: фамилия врача, категория, специальность, оклад, контактный телефон.
    #     Об обследовании должна содержаться следующая информация: название обследования, вид обследования, дата проведения обследования, стоимость обследования.
    #     О страховой компании должна содержаться следующая информация: название страховой компании, номер лицензии, фамилия руководителя, контактный телефон.""")
    #
    # loaded_model, loaded_tokenizer, loaded_label_map = load_model_and_tokenizer(output_dir, num_labels=len(label_map))
    #
    # predicted_entities = predict(test_text,
    #                              loaded_model,
    #                              loaded_tokenizer,
    #                              loaded_label_map,
    #                              device)
    #
    # print("Predicted Entities:")
    # print(predicted_entities)
    #
    # processed_entities = postprocess_entities(predicted_entities)
    #
    # print("Processed Entities:")
    # print(processed_entities)
