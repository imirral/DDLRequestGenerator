import os
import json
# import torch
# import torch.nn as nn

from transformers import BertModel, BertTokenizer
# from torch.utils.data import Dataset, DataLoader


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
                    else:
                        type_labels[i] = f'B-{label}' if type_labels[i - 1] == 'O' else f'I-{label}'

        prepared_data.append({
            'tokens': tokens,
            'entity_labels': entity_labels,
            'attribute_labels': attribute_labels,
            'type_labels': type_labels
        })

    return prepared_data


path_to_data = os.path.abspath("data/imirral.jsonl")

preprocessor = DataPreprocessor(path_to_data)
preprocessed_data = preprocessor.preprocessed_data
entity_data = prepare_data_for_entity_extraction(preprocessed_data)
print()


# class EntityDataset(Dataset):
#     def __init__(self, data, tokenizer, max_length=512):
#         self.data = data
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         item = self.data[idx]
#
#         encoding = self.tokenizer(item['tokens'],
#                                   is_split_into_words=True,
#                                   return_offsets_mapping=True,
#                                   padding='max_length',
#                                   truncation=True,
#                                   max_length=self.max_length)
#
#         entity_labels = self.align_labels(item['entity_labels'], encoding)
#         attribute_labels = self.align_labels(item['attribute_labels'], encoding)
#         type_labels = self.align_labels(item['type_labels'], encoding)
#
#         return {
#             'input_ids': torch.tensor(encoding['input_ids']),
#             'attention_mask': torch.tensor(encoding['attention_mask']),
#             'entity_labels': torch.tensor(entity_labels),
#             'attribute_labels': torch.tensor(attribute_labels),
#             'type_labels': torch.tensor(type_labels)
#         }
#
#     def align_labels(self, labels, encoding):
#         aligned_labels = []
#
#         for word_idx in encoding.word_ids():
#             if word_idx is None:
#                 aligned_labels.append(-100)
#             else:
#                 aligned_labels.append(self.label_to_id(labels[word_idx]))
#         return aligned_labels
#
#     def label_to_id(self, label):
#         label_map = {'O': 0, 'B-ENTITY': 1, 'I-ENTITY': 2, 'B-ATTRIBUTE': 3, 'I-ATTRIBUTE': 4,
#                      'B-VARCHAR': 5, 'I-VARCHAR': 6, 'B-INT': 7, 'I-INT': 8, 'B-DATE': 9, 'I-DATE': 10}
#         return label_map.get(label, 0)
#
#
# class EntityExtractionModel(nn.Module):
#     def __init__(self, num_labels):
#         super(EntityExtractionModel, self).__init__()
#         self.bert = BertModel.from_pretrained("DeepPavlov/rubert-base-cased")
#         self.dropout = nn.Dropout(0.1)
#         self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
#
#     def forward(self, input_ids, attention_mask):
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         sequence_output = outputs[0]
#         sequence_output = self.dropout(sequence_output)
#         logits = self.classifier(sequence_output)
#
#         return logits
#
#
# def train_entity_extraction_model(model, train_dataloader, num_epochs, device):
#     optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
#     criterion = nn.CrossEntropyLoss()
#
#     model.to(device)
#     model.train()
#
#     for epoch in range(num_epochs):
#         for batch in train_dataloader:
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             entity_labels = batch['entity_labels'].to(device)
#             attribute_labels = batch['attribute_labels'].to(device)
#             type_labels = batch['type_labels'].to(device)
#
#             optimizer.zero_grad()
#
#             logits = model(input_ids, attention_mask)
#
#             entity_loss = criterion(logits[:, :, :3].view(-1, 3), entity_labels.view(-1))
#             attribute_loss = criterion(logits[:, :, 3:5].view(-1, 2), attribute_labels.view(-1))
#             type_loss = criterion(logits[:, :, 5:].view(-1, 6), type_labels.view(-1))
#
#             total_loss = entity_loss + attribute_loss + type_loss
#             total_loss.backward()
#
#             optimizer.step()
#
#         print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss.item()}")
#
#
# tokenizer = BertTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
# dataset = EntityDataset(entity_data, tokenizer)
# dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# entity_model = EntityExtractionModel(num_labels=11)
# train_entity_extraction_model(entity_model, dataloader, num_epochs=5, device=device)
#
#
# def prepare_data_for_relation_extraction(preprocessed_data):
#     relation_data = []
#
#     for item in preprocessed_data:
#         text = item['text']
#         entities = item['entities']
#         attributes = item['attributes']
#
#         for i, entity1 in enumerate(entities):
#             for j, entity2 in enumerate(entities[i + 1:]):
#                 start1, end1, _ = entity1
#                 start2, end2, _ = entity2
#                 context = text[max(0, start1 - 50):min(len(text), end2 + 50)]
#                 relation_data.append({
#                     'text': context,
#                     'entity1': text[start1:end1],
#                     'entity2': text[start2:end2],
#                     'entity1_start': start1 - max(0, start1 - 50),
#                     'entity1_end': end1 - max(0, start1 - 50),
#                     'entity2_start': start2 - max(0, start1 - 50),
#                     'entity2_end': end2 - max(0, start1 - 50),
#                     'relation': 'unknown'
#                 })
#
#         for entity in entities:
#             for attribute in attributes:
#                 start_e, end_e, _ = entity
#                 start_a, end_a, _ = attribute
#                 context = text[max(0, start_e - 50):min(len(text), max(end_e, end_a) + 50)]
#                 relation_data.append({
#                     'text': context,
#                     'entity1': text[start_e:end_e],
#                     'entity2': text[start_a:end_a],
#                     'entity1_start': start_e - max(0, start_e - 50),
#                     'entity1_end': end_e - max(0, start_e - 50),
#                     'entity2_start': start_a - max(0, start_e - 50),
#                     'entity2_end': end_a - max(0, start_e - 50),
#                     'relation': 'has_attribute'
#                 })
#
#     return relation_data
#
#
# relation_data = prepare_data_for_relation_extraction(preprocessed_data)
#
#
# class RelationDataset(Dataset):
#     def __init__(self, data, tokenizer, max_length=512):
#         self.data = data
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         item = self.data[idx]
#         text = item['text']
#         entity1 = item['entity1']
#         entity2 = item['entity2']
#
#         encoded = self.tokenizer.encode_plus(
#             text,
#             add_special_tokens=True,
#             max_length=self.max_length,
#             return_token_type_ids=False,
#             padding='max_length',
#             truncation=True,
#             return_attention_mask=True,
#             return_tensors='pt'
#         )
#
#         entity1_start = encoded.char_to_token(item['entity1_start'])
#         entity1_end = encoded.char_to_token(item['entity1_end'] - 1)
#         entity2_start = encoded.char_to_token(item['entity2_start'])
#         entity2_end = encoded.char_to_token(item['entity2_end'] - 1)
#
#         return {
#             'input_ids': encoded['input_ids'].flatten(),
#             'attention_mask': encoded['attention_mask'].flatten(),
#             'entity1_pos': torch.tensor([entity1_start, entity1_end]),
#             'entity2_pos': torch.tensor([entity2_start, entity2_end]),
#             'label': torch.tensor(self.relation_to_id(item['relation']))
#         }
#
#     def relation_to_id(self, relation):
#         relation_map = {'unknown': 0, 'has_attribute': 1, 'one_to_many': 2, 'many_to_one': 3, 'many_to_many': 4}
#         return relation_map.get(relation, 0)
#
#
# class RelationExtractionModel(nn.Module):
#     def __init__(self, num_labels):
#         super(RelationExtractionModel, self).__init__()
#         self.bert = BertModel.from_pretrained("DeepPavlov/rubert-base-cased")
#         self.dropout = nn.Dropout(0.1)
#         self.classifier = nn.Linear(self.bert.config.hidden_size * 2, num_labels)
#
#     def forward(self, input_ids, attention_mask, entity1_pos, entity2_pos):
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         sequence_output = outputs[0]
#
#         entity1_output = torch.mean(sequence_output[torch.arange(sequence_output.size(0)).unsqueeze(1), entity1_pos],
#                                     dim=1)
#         entity2_output = torch.mean(sequence_output[torch.arange(sequence_output.size(0)).unsqueeze(1), entity2_pos],
#                                     dim=1)
#
#         concat_output = torch.cat((entity1_output, entity2_output), dim=1)
#         concat_output = self.dropout(concat_output)
#         logits = self.classifier(concat_output)
#         return logits
#
#
# def train_relation_extraction_model(model, train_dataloader, num_epochs, device):
#     optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
#     criterion = nn.CrossEntropyLoss()
#
#     model.to(device)
#     model.train()
#
#     for epoch in range(num_epochs):
#         total_loss = 0.0
#
#         for batch in train_dataloader:
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             entity1_pos = batch['entity1_pos'].to(device)
#             entity2_pos = batch['entity2_pos'].to(device)
#             labels = batch['label'].to(device)
#
#             optimizer.zero_grad()
#
#             logits = model(input_ids, attention_mask, entity1_pos, entity2_pos)
#             loss = criterion(logits, labels)
#
#             loss.backward()
#             optimizer.step()
#
#             total_loss += loss.item()
#
#         avg_loss = total_loss / len(train_dataloader)
#         print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
#
#
# tokenizer = BertTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
# relation_dataset = RelationDataset(relation_data, tokenizer)
# relation_dataloader = DataLoader(relation_dataset, batch_size=8, shuffle=True)
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# relation_model = RelationExtractionModel(num_labels=5)
# train_relation_extraction_model(relation_model, relation_dataloader, num_epochs=5, device=device)
#
#
# def analyze_text(entity_model, relation_model, text, tokenizer, device):
#     tokens = tokenizer.tokenize(text)
#     input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)]).to(device)
#     attention_mask = torch.tensor([[1] * len(tokens)]).to(device)
#
#     with torch.no_grad():
#         entity_logits = entity_model(input_ids, attention_mask)
#
#     entity_predictions = torch.argmax(entity_logits, dim=2).squeeze().tolist()
#
#     entities = []
#     attributes = []
#     current_entity = None
#     current_attribute = None
#
#     for i, (token, pred) in enumerate(zip(tokens, entity_predictions)):
#         if pred == 1:  # B-ENTITY
#             if current_entity:
#                 entities.append(current_entity)
#             current_entity = {'start': i, 'end': i + 1, 'text': token}
#         elif pred == 2:  # I-ENTITY
#             if current_entity:
#                 current_entity['end'] = i + 1
#                 current_entity['text'] += ' ' + token
#         elif pred == 3:  # B-ATTRIBUTE
#             if current_attribute:
#                 attributes.append(current_attribute)
#             current_attribute = {'start': i, 'end': i + 1, 'text': token, 'type': 'VARCHAR'}
#         elif pred == 4:  # I-ATTRIBUTE
#             if current_attribute:
#                 current_attribute['end'] = i + 1
#                 current_attribute['text'] += ' ' + token
#         elif pred in [5, 7, 9]:  # B-VARCHAR, B-INT, B-DATE
#             if current_attribute:
#                 current_attribute['type'] = ['VARCHAR', 'INT', 'DATE'][(pred - 5) // 2]
#
#     if current_entity:
#         entities.append(current_entity)
#     if current_attribute:
#         attributes.append(current_attribute)
#
#     relations = []
#
#     for i, entity1 in enumerate(entities):
#         for j, entity2 in enumerate(entities[i + 1:]):
#             context = text[max(0, entity1['start'] - 50):min(len(text), entity2['end'] + 50)]
#             encoded = tokenizer.encode_plus(
#                 context,
#                 add_special_tokens=True,
#                 max_length=512,
#                 return_token_type_ids=False,
#                 padding='max_length',
#                 truncation=True,
#                 return_attention_mask=True,
#                 return_tensors='pt'
#             )
#             entity1_start = tokenizer.encode(text[entity1['start']:entity1['end']], add_special_tokens=False)[0]
#             entity1_end = tokenizer.encode(text[entity1['start']:entity1['end']], add_special_tokens=False)[-1]
#             entity2_start = tokenizer.encode(text[entity2['start']:entity2['end']], add_special_tokens=False)[0]
#             entity2_end = tokenizer.encode(text[entity2['start']:entity2['end']], add_special_tokens=False)[-1]
#
#             with torch.no_grad():
#                 relation_logits = relation_model(
#                     encoded['input_ids'].to(device),
#                     encoded['attention_mask'].to(device),
#                     torch.tensor([[entity1_start, entity1_end]]).to(device),
#                     torch.tensor([[entity2_start, entity2_end]]).to(device)
#                 )
#
#             relation_pred = torch.argmax(relation_logits, dim=1).item()
#             if relation_pred != 0:
#                 relations.append({
#                     'entity1': entity1['text'],
#                     'entity2': entity2['text'],
#                     'type': ['unknown', 'has_attribute', 'one_to_many', 'many_to_one', 'many_to_many'][relation_pred]
#                 })
#
#     return entities, attributes, relations
#
#
# text = "Ваш текст для анализа"
# entities, attributes, relations = analyze_text(entity_model, relation_model, text, tokenizer, device)
#
#
# def process_analysis_results(entities, attributes, relations):
#     tables = {}
#     for entity in entities:
#         table_name = entity['text'].replace(' ', '_').lower()
#         tables[table_name] = {'attributes': []}
#
#     for attribute in attributes:
#         for relation in relations:
#             if relation['type'] == 'has_attribute' and relation['entity2'] == attribute['text']:
#                 table_name = relation['entity1'].replace(' ', '_').lower()
#                 if table_name in tables:
#                     tables[table_name]['attributes'].append({
#                         'name': attribute['text'].replace(' ', '_').lower(),
#                         'type': attribute['type']
#                     })
#
#     for relation in relations:
#         if relation['type'] in ['one_to_many', 'many_to_one', 'many_to_many']:
#             table1 = relation['entity1'].replace(' ', '_').lower()
#             table2 = relation['entity2'].replace(' ', '_').lower()
#             if table1 in tables and table2 in tables:
#                 if 'relations' not in tables[table1]:
#                     tables[table1]['relations'] = []
#                 tables[table1]['relations'].append({
#                     'table': table2,
#                     'type': relation['type']
#                 })
#
#     return tables
#
#
# def generate_sql_query(processed_results):
#     sql_query = ""
#
#     for table_name, table_info in processed_results.items():
#         sql_query += f"CREATE TABLE {table_name} (n"
#         sql_query += f"    id INT PRIMARY KEY AUTO_INCREMENT,n"
#         for attribute in table_info['attributes']:
#             sql_query += f"    {attribute['name']} {attribute['type']},n"
#         sql_query = sql_query.rstrip(',n') + "n);nn"
#
#     for table_name, table_info in processed_results.items():
#         if 'relations' in table_info:
#             for relation in table_info['relations']:
#                 if relation['type'] == 'many_to_one':
#                     sql_query += f"ALTER TABLE {table_name} ADD COLUMN {relation['table']}_id INT;n"
#                     sql_query += f"ALTER TABLE {table_name} ADD FOREIGN KEY ({relation['table']}_id) REFERENCES {relation['table']}(id);nn"
#                 elif relation['type'] == 'one_to_many':
#                     sql_query += f"ALTER TABLE {relation['table']} ADD COLUMN {table_name}_id INT;n"
#                     sql_query += f"ALTER TABLE {relation['table']} ADD FOREIGN KEY ({table_name}_id) REFERENCES {table_name}(id);nn"
#                 elif relation['type'] == 'many_to_many':
#                     junction_table = f"{table_name}_{relation['table']}"
#                     sql_query += f"CREATE TABLE {junction_table} (n"
#                     sql_query += f"    {table_name}_id INT,n"
#                     sql_query += f"    {relation['table']}_id INT,n"
#                     sql_query += f"    PRIMARY KEY ({table_name}_id, {relation['table']}_id),n"
#                     sql_query += f"    FOREIGN KEY ({table_name}_id) REFERENCES {table_name}(id),n"
#                     sql_query += f"    FOREIGN KEY ({relation['table']}_id) REFERENCES {relation['table']}(id)n"
#                     sql_query += ");nn"
#
#     return sql_query
#
#
# def text_to_sql(text, entity_model, relation_model, tokenizer, device):
#     entities, attributes, relations = analyze_text(entity_model, relation_model, text, tokenizer, device)
#
#     processed_results = process_analysis_results(entities, attributes, relations)
#
#     sql_query = generate_sql_query(processed_results)
#
#     return sql_query
#
#
# text = "Ваш текст для анализа"
# sql_query = text_to_sql(text, entity_model, relation_model, tokenizer, device)
# print(sql_query)
