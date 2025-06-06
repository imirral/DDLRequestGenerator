import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
import torch
import torch.nn as nn

from transformers import BertModel, BertTokenizerFast
from torch.utils.data import Dataset, DataLoader
from without_dt.preprocessing import preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def prepare_data_for_relation_extraction(preprocessed_data):
    relation_data = []

    for item in preprocessed_data:
        text = item['text']
        id_to_entity = item['id_to_entity']
        relations = item['relations']

        for from_id, to_id, rel_type in relations:
            entity1 = id_to_entity[from_id]
            entity2 = id_to_entity[to_id]

            start1, end1 = entity1[1], entity1[2]
            start2, end2 = entity2[1], entity2[2]

            start = max(0, min(start1, start2) - 50)
            end = min(len(text), max(end1, end2) + 50)
            context = text[start:end]

            relation_data.append({
                'text': context,
                'entity1': text[start1:end1],
                'entity2': text[start2:end2],
                'entity1_start': start1 - start,
                'entity1_end': end1 - start,
                'entity2_start': start2 - start,
                'entity2_end': end2 - start,
                'relation': rel_type.lower()
            })

    return relation_data


def determine_relation(entity1, entity2, relation_dict):
    entity1_id = entity1[0]
    entity2_id = entity2[0]

    if (entity1_id, entity2_id) in relation_dict:
        rel_type = relation_dict[(entity1_id, entity2_id)].lower()
        return rel_type
    elif (entity2_id, entity1_id) in relation_dict:
        rel_type = relation_dict[(entity2_id, entity1_id)].lower()
        return rel_type
    else:
        return 'unknown'


class RelationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.relation_map = {'has_attribute': 0, 'one_to_one': 1, 'one_to_many': 2, 'many_to_many': 3}
        self.id2relation = {v: k for k, v in self.relation_map.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        entity1_span = (item['entity1_start'], item['entity1_end'])
        entity2_span = (item['entity2_start'], item['entity2_end'])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_tensors='pt'
        )

        labels = torch.tensor(self.relation_map.get(item['relation'], 0), dtype=torch.long)

        entity_positions = self.get_entity_positions(encoding['offset_mapping'][0], entity1_span, entity2_span)

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'entity1_mask': entity_positions['entity1_mask'],
            'entity2_mask': entity_positions['entity2_mask'],
            'labels': labels
        }

    def get_entity_positions(self, offset_mapping, entity1_span, entity2_span):
        entity1_mask = torch.zeros(len(offset_mapping), dtype=torch.bool)
        entity2_mask = torch.zeros(len(offset_mapping), dtype=torch.bool)

        for idx, (start, end) in enumerate(offset_mapping):
            if start == end:
                continue
            if (start <= entity1_span[0] < end or start < entity1_span[1] <= end
                    or (entity1_span[0] <= start and end <= entity1_span[1])):
                entity1_mask[idx] = True
            if (start <= entity2_span[0] < end or start < entity2_span[1] <= end
                    or (entity2_span[0] <= start and end <= entity2_span[1])):
                entity2_mask[idx] = True

        return {'entity1_mask': entity1_mask, 'entity2_mask': entity2_mask}


class RelationModel(nn.Module):
    def __init__(self, num_labels, freeze_bert_layers=False):
        super(RelationModel, self).__init__()
        self.bert = BertModel.from_pretrained("DeepPavlov/rubert-base-cased")

        if freeze_bert_layers:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size * 2, num_labels)

    def forward(self, input_ids, attention_mask, entity1_mask, entity2_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        entity1_rep = self.get_entity_representation(sequence_output, entity1_mask)
        entity2_rep = self.get_entity_representation(sequence_output, entity2_mask)

        concat_output = torch.cat((entity1_rep, entity2_rep), dim=1)
        concat_output = self.dropout(concat_output)
        logits = self.classifier(concat_output)
        return logits

    def get_entity_representation(self, sequence_output, entity_mask):
        batch_size, seq_len, hidden_size = sequence_output.size()
        entity_rep = torch.zeros(batch_size, hidden_size, device=sequence_output.device)
        has_entity = entity_mask.any(dim=1)

        if has_entity.any():
            first_token_indices = entity_mask[has_entity].float().argmax(dim=1)
            batch_indices = torch.nonzero(has_entity, as_tuple=False).squeeze()
            entity_rep[has_entity] = sequence_output[batch_indices, first_token_indices, :]

        return entity_rep


def train_relation_model(model, train_dataloader, val_dataloader, num_epochs, device, relation_map):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    use_amp = device.type == 'cuda'
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()

    model.to(device)

    for epoch in range(num_epochs):
        total_loss = 0.0
        model.train()

        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            entity1_mask = batch['entity1_mask'].to(device)
            entity2_mask = batch['entity2_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(input_ids, attention_mask, entity1_mask, entity2_mask)
                    loss = criterion(logits, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(input_ids, attention_mask, entity1_mask, entity2_mask)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_loss:.4f}")

        evaluate(model, val_dataloader, device, relation_map)


def evaluate(model, dataloader, device, relation_map):
    model.eval()
    true_labels = []
    pred_labels = []

    id2relation = {v: k for k, v in relation_map.items()}

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            entity1_mask = batch['entity1_mask'].to(device)
            entity2_mask = batch['entity2_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask, entity1_mask, entity2_mask)
            predictions = torch.argmax(logits, dim=1)

            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predictions.cpu().numpy())

    report = classification_report(true_labels, pred_labels, target_names=list(id2relation.values()), digits=4)
    print("Validation Metrics:")
    print(report)
    model.train()


def save_model_and_tokenizer(model, tokenizer, output_dir, relation_map):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))

    model_to_save.bert.config.to_json_file(os.path.join(output_dir, 'config.json'))

    tokenizer.save_pretrained(output_dir)

    with open(os.path.join(output_dir, 'relation_map.json'), 'w') as f:
        json.dump(relation_map, f)

    print(f"Model and tokenizer saved to {output_dir}")


def load_model_and_tokenizer(output_dir, num_labels):
    tokenizer = BertTokenizerFast.from_pretrained(output_dir)
    model = RelationModel(num_labels=num_labels)
    model.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin'), map_location=torch.device('cpu')))

    with open(os.path.join(output_dir, 'relation_map.json'), 'r') as f:
        relation_map = json.load(f)

    return model, tokenizer, relation_map


def predict(model, tokenizer, text, entity1_span, entity2_span, device, relation_map, threshold=0.5):
    model.to(device)
    model.eval()

    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_offsets_mapping=True,
        return_tensors='pt'
    )

    entity_positions = get_entity_positions(encoding['offset_mapping'][0], entity1_span, entity2_span)
    entity1_mask = entity_positions['entity1_mask'].unsqueeze(0).to(device)
    entity2_mask = entity_positions['entity2_mask'].unsqueeze(0).to(device)

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask, entity1_mask, entity2_mask)
        probs = torch.softmax(logits, dim=1)  # Размер [batch_size, num_labels]

        # Для batch_size=1 извлекаем значения из тензоров
        max_prob, prediction = torch.max(probs, dim=1)
        max_prob = max_prob.item()  # Скаляр
        prediction = prediction.item()  # Индекс класса

    id2relation = {v: k for k, v in relation_map.items()}

    # Если самая большая вероятность меньше порога - отказываемся
    if max_prob < threshold:
        return 'unknown'
    else:
        return id2relation.get(prediction, 'unknown')


def get_entity_positions(offset_mapping, entity1_span, entity2_span):
    entity1_mask = torch.zeros(len(offset_mapping), dtype=torch.bool)
    entity2_mask = torch.zeros(len(offset_mapping), dtype=torch.bool)

    for idx, (start, end) in enumerate(offset_mapping):
        if start == end:
            continue
        if (start <= entity1_span[0] < end or start < entity1_span[1] <= end
                or (entity1_span[0] <= start and end <= entity1_span[1])):
            entity1_mask[idx] = True
        if (start <= entity2_span[0] < end or start < entity2_span[1] <= end
                or (entity2_span[0] <= start and end <= entity2_span[1])):
            entity2_mask[idx] = True

    return {'entity1_mask': entity1_mask, 'entity2_mask': entity2_mask}


if __name__ == '__main__':
    preprocessed_data = preprocess_data()
    relation_data = prepare_data_for_relation_extraction(preprocessed_data)

    train_data, val_data = train_test_split(relation_data, test_size=0.2, random_state=42)

    tokenizer = BertTokenizerFast.from_pretrained("DeepPavlov/rubert-base-cased")

    train_dataset = RelationDataset(train_data, tokenizer)
    val_dataset = RelationDataset(val_data, tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    relation_map = train_dataset.relation_map
    num_labels = len(relation_map)
    model = RelationModel(num_labels=num_labels, freeze_bert_layers=False)

    train_relation_model(model, train_dataloader, val_dataloader, num_epochs=10, device=device, relation_map=relation_map)

    output_dir = 'D:/Magistracy/FQW/DDLRequestGenerator/saved_models/relation_model_without_dt'
    save_model_and_tokenizer(model, tokenizer, output_dir, relation_map)

    # loaded_model, loaded_tokenizer, loaded_relation_map = load_model_and_tokenizer(output_dir, num_labels=num_labels)
    #
    # example_text = "Библиотека имеет множество книг, которые написаны разными авторами."
    # entity1_text = "Библиотека"
    # entity2_text = "книг"
    #
    # entity1_start = example_text.find(entity1_text)
    # entity1_end = entity1_start + len(entity1_text)
    # entity2_start = example_text.find(entity2_text)
    # entity2_end = entity2_start + len(entity2_text)
    #
    # predicted_relation = predict(
    #     loaded_model,
    #     loaded_tokenizer,
    #     example_text,
    #     (entity1_start, entity1_end),
    #     (entity2_start, entity2_end),
    #     device,
    #     loaded_relation_map
    # )
    #
    # print(f"Predicted Relation between '{entity1_text}' and '{entity2_text}': {predicted_relation}")
