import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader
from data_preprocessor import DataPreprocessor


def prepare_data_for_relation_extraction(preprocessed_data):
    relation_data = []
    for item in preprocessed_data:
        text = item['text']
        entities = item['entities']
        attributes = item['attributes']

        # Генерируем пары сущностей
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i + 1:]):
                start1, end1, _ = entity1
                start2, end2, _ = entity2
                context = text[max(0, start1 - 50):min(len(text), end2 + 50)]
                relation_data.append({
                    'text': context,
                    'entity1': text[start1:end1],
                    'entity2': text[start2:end2],
                    'entity1_start': start1 - max(0, start1 - 50),
                    'entity1_end': end1 - max(0, start1 - 50),
                    'entity2_start': start2 - max(0, start1 - 50),
                    'entity2_end': end2 - max(0, start1 - 50),
                    'relation': 'unknown'  # Здесь нужно добавить правильную метку связи, если она есть
                })

        # Генерируем пары сущность-атрибут
        for entity in entities:
            for attribute in attributes:
                start_e, end_e, _ = entity
                start_a, end_a, _ = attribute
                context = text[max(0, start_e - 50):min(len(text), max(end_e, end_a) + 50)]
                relation_data.append({
                    'text': context,
                    'entity1': text[start_e:end_e],
                    'entity2': text[start_a:end_a],
                    'entity1_start': start_e - max(0, start_e - 50),
                    'entity1_end': end_e - max(0, start_e - 50),
                    'entity2_start': start_a - max(0, start_e - 50),
                    'entity2_end': end_a - max(0, start_e - 50),
                    'relation': 'has_attribute'  # Предполагаем, что все атрибуты принадлежат сущностям
                })

    return relation_data


# Подготовка данных для извлечения отношений
preprocessor = DataPreprocessor('path_to_your_jsonl_file.jsonl')
preprocessed_data = preprocessor.preprocessed_data
relation_data = prepare_data_for_relation_extraction(preprocessed_data)


class RelationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        entity1 = item['entity1']
        entity2 = item['entity2']

        encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        # Находим позиции сущностей в токенизированном тексте
        entity1_start = encoded.char_to_token(item['entity1_start'])
        entity1_end = encoded.char_to_token(item['entity1_end'] - 1)
        entity2_start = encoded.char_to_token(item['entity2_start'])
        entity2_end = encoded.char_to_token(item['entity2_end'] - 1)

        return {
            'input_ids': encoded['input_ids'].flatten(),
            'attention_mask': encoded['attention_mask'].flatten(),
            'entity1_pos': torch.tensor([entity1_start, entity1_end]),
            'entity2_pos': torch.tensor([entity2_start, entity2_end]),
            'label': torch.tensor(self.relation_to_id(item['relation']))
        }

    def relation_to_id(self, relation):
        relation_map = {'unknown': 0, 'has_attribute': 1, 'one_to_many': 2, 'many_to_one': 3, 'many_to_many': 4}
        return relation_map.get(relation, 0)


class RelationExtractionModel(nn.Module):
    def __init__(self, num_labels):
        super(RelationExtractionModel, self).__init__()
        self.bert = BertModel.from_pretrained("DeepPavlov/rubert-base-cased")
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size * 2, num_labels)

    def forward(self, input_ids, attention_mask, entity1_pos, entity2_pos):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]

        entity1_output = torch.mean(sequence_output[torch.arange(sequence_output.size(0)).unsqueeze(1), entity1_pos],
                                    dim=1)
        entity2_output = torch.mean(sequence_output[torch.arange(sequence_output.size(0)).unsqueeze(1), entity2_pos],
                                    dim=1)

        concat_output = torch.cat((entity1_output, entity2_output), dim=1)
        concat_output = self.dropout(concat_output)
        logits = self.classifier(concat_output)
        return logits


def train_relation_extraction_model(model, train_dataloader, num_epochs, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0

        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            entity1_pos = batch['entity1_pos'].to(device)
            entity2_pos = batch['entity2_pos'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()

            logits = model(input_ids, attention_mask, entity1_pos, entity2_pos)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")


# Создаем tokenizer и dataset
tokenizer = BertTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
relation_dataset = RelationDataset(relation_data, tokenizer)
relation_dataloader = DataLoader(relation_dataset, batch_size=8, shuffle=True)

# Инициализация и обучение модели
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
relation_model = RelationExtractionModel(num_labels=5)  # 5 меток: unknown, has_attribute, и т.д.
train_relation_extraction_model(relation_model, relation_dataloader, num_epochs=5, device=device)
