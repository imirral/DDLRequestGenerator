import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader
from data_preprocessor import DataPreprocessor


def prepare_data_for_entity_extraction(preprocessed_data):
    tokenizer = BertTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")

    prepared_data = []
    for item in preprocessed_data:
        tokens = tokenizer.tokenize(item['text'])
        token_spans = tokenizer.get_offset_mapping(item['text'])

        entity_labels = ['O'] * len(tokens)
        attribute_labels = ['O'] * len(tokens)
        type_labels = ['O'] * len(tokens)

        for start, end, label in item['entities'] + item['attributes'] + item['types']:
            for i, (token_start, token_end) in enumerate(token_spans):
                if token_start >= start and token_end <= end:
                    if label == 'ENTITY':
                        entity_labels[i] = 'B-ENTITY' if entity_labels[i - 1] == 'O' else 'I-ENTITY'
                    elif label == 'ATTRIBUTE':
                        attribute_labels[i] = 'B-ATTRIBUTE' if attribute_labels[i - 1] == 'O' else 'I-ATTRIBUTE'
                    else:  # Type labels
                        type_labels[i] = f'B-{label}' if type_labels[i - 1] == 'O' else f'I-{label}'

        prepared_data.append({
            'tokens': tokens,
            'entity_labels': entity_labels,
            'attribute_labels': attribute_labels,
            'type_labels': type_labels
        })

    return prepared_data


# Подготовка данных для извлечения сущностей
preprocessor = DataPreprocessor('path_to_your_jsonl_file.jsonl')
preprocessed_data = preprocessor.preprocessed_data
ner_data = prepare_data_for_entity_extraction(preprocessed_data)


class EntityDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(item['tokens'],
                                  is_split_into_words=True,
                                  return_offsets_mapping=True,
                                  padding='max_length',
                                  truncation=True,
                                  max_length=self.max_length)

        entity_labels = self.align_labels(item['entity_labels'], encoding)
        attribute_labels = self.align_labels(item['attribute_labels'], encoding)
        type_labels = self.align_labels(item['type_labels'], encoding)

        return {
            'input_ids': torch.tensor(encoding['input_ids']),
            'attention_mask': torch.tensor(encoding['attention_mask']),
            'entity_labels': torch.tensor(entity_labels),
            'attribute_labels': torch.tensor(attribute_labels),
            'type_labels': torch.tensor(type_labels)
        }

    def align_labels(self, labels, encoding):
        aligned_labels = []
        for word_idx in encoding.word_ids():
            if word_idx is None:
                aligned_labels.append(-100)
            else:
                aligned_labels.append(self.label_to_id(labels[word_idx]))
        return aligned_labels

    def label_to_id(self, label):
        label_map = {'O': 0, 'B-ENTITY': 1, 'I-ENTITY': 2, 'B-ATTRIBUTE': 3, 'I-ATTRIBUTE': 4,
                     'B-VARCHAR': 5, 'I-VARCHAR': 6, 'B-INT': 7, 'I-INT': 8, 'B-DATE': 9, 'I-DATE': 10}
        return label_map.get(label, 0)


class EntityExtractionModel(nn.Module):
    def __init__(self, num_labels):
        super(EntityExtractionModel, self).__init__()
        self.bert = BertModel.from_pretrained("DeepPavlov/rubert-base-cased")
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits


def train_entity_extraction_model(model, train_dataloader, num_epochs, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            entity_labels = batch['entity_labels'].to(device)
            attribute_labels = batch['attribute_labels'].to(device)
            type_labels = batch['type_labels'].to(device)

            optimizer.zero_grad()

            logits = model(input_ids, attention_mask)

            entity_loss = criterion(logits[:, :, :3].view(-1, 3), entity_labels.view(-1))
            attribute_loss = criterion(logits[:, :, 3:5].view(-1, 2), attribute_labels.view(-1))
            type_loss = criterion(logits[:, :, 5:].view(-1, 6), type_labels.view(-1))

            total_loss = entity_loss + attribute_loss + type_loss
            total_loss.backward()

            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss.item()}")


# Подготовка данных для обучения
tokenizer = BertTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
dataset = EntityDataset(ner_data, tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Обучение модели
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EntityExtractionModel(
    num_labels=11)
# 11 labels: O, B-ENTITY, I-ENTITY, B-ATTRIBUTE, I-ATTRIBUTE,
# B-VARCHAR, I-VARCHAR, B-INT, I-INT, B-DATE, I-DATE
train_entity_extraction_model(model, dataloader, num_epochs=5, device=device)
