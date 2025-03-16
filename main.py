import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import re
import torch

from without_dt.entity_model import load_model_and_tokenizer as load_entity_model_and_tokenizer
from without_dt.entity_model import predict as predict_entities
from without_dt.relation_model import load_model_and_tokenizer as load_relation_model_and_tokenizer
from without_dt.relation_model import predict as predict_relation
from without_dt.postprocessing import postprocess_entities


def split_into_sentences(text):
    sentences = []
    start = 0

    for match in re.finditer(r'[.!?]+', text):
        end = match.end()
        sentences.append((start, end))
        start = end

    if start < len(text):
        sentences.append((start, len(text)))
    return sentences


def get_sentence_index(span, sentence_spans):
    for i, (s_start, s_end) in enumerate(sentence_spans):
        if span['start'] >= s_start and span['end'] <= s_end:
            return i
    return -1


def are_in_same_sentence(entity_span, attribute_span, sentence_spans):
    ent_sent_id = get_sentence_index(entity_span, sentence_spans)
    attr_sent_id = get_sentence_index(attribute_span, sentence_spans)
    return ent_sent_id != -1 and attr_sent_id != -1 and ent_sent_id == attr_sent_id


def separate_entities_and_attributes(predicted_entities):
    entities = []
    attributes = []
    for ent in predicted_entities:
        label = ent['label']
        if label == 'ENTITY':
            entities.append({
                'text': ent['text'],
                'start': ent['start'],
                'end': ent['end']
            })
        else:
            attributes.append({
                'text': ent['text'],
                'start': ent['start'],
                'end': ent['end'],
                'type': label
            })
    return entities, attributes


def analyze_text(text, entity_model, entity_tokenizer, entity_label_map,
                 relation_model, relation_tokenizer, relation_label_map,
                 device):
    predicted_entities = predict_entities(
        text,
        entity_model,
        entity_tokenizer,
        entity_label_map,
        device
    )
    processed_entities = postprocess_entities(predicted_entities)

    entities, attributes = separate_entities_and_attributes(processed_entities)
    sentence_spans = split_into_sentences(text)

    relation_set = set()

    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            ent1 = entities[i]
            ent2 = entities[j]

            predicted_rel = predict_relation(
                relation_model,
                relation_tokenizer,
                text,
                (ent1['start'], ent1['end']),
                (ent2['start'], ent2['end']),
                device,
                relation_label_map
            )

            if predicted_rel != 'unknown':
                key = (ent1['text'].lower(), ent2['text'].lower(), predicted_rel)
                relation_set.add(key)

    for ent in entities:
        for attr in attributes:
            if are_in_same_sentence(ent, attr, sentence_spans):
                predicted_rel = predict_relation(
                    relation_model,
                    relation_tokenizer,
                    text,
                    (ent['start'], ent['end']),
                    (attr['start'], attr['end']),
                    device,
                    relation_label_map
                )
                if predicted_rel == 'has_attribute':
                    key = (ent['text'].lower(), attr['text'].lower(), 'has_attribute')
                    relation_set.add(key)

    relations = []
    for (e1, e2, rtype) in relation_set:
        relations.append({
            'entity1': e1,
            'entity2': e2,
            'type': rtype
        })

    return entities, attributes, relations


def process_analysis_results(entities, attributes, relations):
    def to_table_name(raw_text):
        return raw_text.replace(' ', '_').lower()

    tables = {}
    for e in entities:
        table_name = to_table_name(e['text'])
        if table_name not in tables:
            tables[table_name] = {'attributes': [], 'relations': []}

    for rel in relations:
        if rel['type'] == 'has_attribute':
            table_name = to_table_name(rel['entity1'])
            attr_name = to_table_name(rel['entity2'])
            attr_type = 'VARCHAR'
            for a in attributes:
                if a['text'].lower() == rel['entity2']:
                    attr_type = a['type']
                    break
            if table_name in tables:
                tables[table_name]['attributes'].append({
                    'name': attr_name,
                    'type': attr_type
                })

    for rel in relations:
        if rel['type'] in ['one_to_many', 'many_to_one', 'many_to_many']:
            table1 = to_table_name(rel['entity1'])
            table2 = to_table_name(rel['entity2'])
            if table1 in tables and table2 in tables:
                tables[table1]['relations'].append({
                    'table': table2,
                    'type': rel['type']
                })

    return tables


def generate_sql_query(processed_results):
    sql_query = ""

    for table_name, table_info in processed_results.items():
        sql_query += f"CREATE TABLE {table_name} (\n"
        sql_query += f"    id INT PRIMARY KEY AUTO_INCREMENT,\n"
        for attribute in table_info['attributes']:
            sql_type = attribute['type'].upper()
            if sql_type not in ["INT", "DATE", "DATETIME", "VARCHAR"]:
                sql_type = "VARCHAR(255)"
            elif sql_type == "VARCHAR":
                sql_type = "VARCHAR(255)"
            sql_query += f"    {attribute['name']} {sql_type},\n"
        sql_query = sql_query.rstrip(",\n") + "\n);\n\n"

    fk_set = set()

    for table_name, table_info in processed_results.items():
        for rel in table_info['relations']:
            relation_type = rel['type']
            other_table = rel['table']

            if table_name == other_table:
                continue

            if relation_type == 'many_to_one':
                fk_key = (table_name, other_table)
                if fk_key not in fk_set:
                    fk_set.add(fk_key)
                    sql_query += (
                        f"ALTER TABLE {table_name} ADD COLUMN {other_table}_id INT;\n"
                        f"ALTER TABLE {table_name} ADD FOREIGN KEY ({other_table}_id) REFERENCES {other_table}(id);\n\n"
                    )

            elif relation_type == 'one_to_many':
                fk_key = (other_table, table_name)
                if fk_key not in fk_set:
                    fk_set.add(fk_key)
                    sql_query += (
                        f"ALTER TABLE {other_table} ADD COLUMN {table_name}_id INT;\n"
                        f"ALTER TABLE {other_table} ADD FOREIGN KEY ({table_name}_id) REFERENCES {table_name}(id);\n\n"
                    )

            elif relation_type == 'many_to_many':
                junction_table = f"{table_name}_{other_table}"
                fk_key = ("m2m", table_name, other_table)
                if fk_key not in fk_set:
                    fk_set.add(fk_key)
                    sql_query += f"CREATE TABLE {junction_table} (\n"
                    sql_query += f"    {table_name}_id INT,\n"
                    sql_query += f"    {other_table}_id INT,\n"
                    sql_query += f"    PRIMARY KEY ({table_name}_id, {other_table}_id),\n"
                    sql_query += f"    FOREIGN KEY ({table_name}_id) REFERENCES {table_name}(id),\n"
                    sql_query += f"    FOREIGN KEY ({other_table}_id) REFERENCES {other_table}(id)\n"
                    sql_query += f");\n\n"

    return sql_query


def text_to_sql(text, entity_model, entity_tokenizer, entity_label_map,
                relation_model, relation_tokenizer, relation_label_map, device):

    entities, attributes, relations = analyze_text(
        text,
        entity_model,
        entity_tokenizer,
        entity_label_map,
        relation_model,
        relation_tokenizer,
        relation_label_map,
        device
    )

    processed_results = process_analysis_results(entities, attributes, relations)
    sql_query = generate_sql_query(processed_results)
    return sql_query


if __name__ == "__main__":
    entity_model_path = "D:/Magistracy/FQW/DDLRequestGenerator/saved_models/entity_model_without_dt"
    relation_model_path = "D:/Magistracy/FQW/DDLRequestGenerator/saved_models/relation_model_without_dt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaded_entity_model, loaded_entity_tokenizer, loaded_entity_label_map = load_entity_model_and_tokenizer(
        entity_model_path,
        num_labels=15
    )

    loaded_relation_model, loaded_relation_tokenizer, loaded_relation_label_map = load_relation_model_and_tokenizer(
        relation_model_path,
        num_labels=4
    )

    test_text = ("""
        Пациент заключает договор на лечение в заданном отделении больницы. У каждого пациента есть лечащий врач. 
        Пациент может оплачивать свой договор за счёт страховой компании. Обследование пациента проводит не только лечащий врач.
        О пациенте должна содержаться следующая информация: фамилия пациента, его категория, номер паспорта, номер страхового полиса, дата поступления, гражданство.
        О враче должна содержаться следующая информация: фамилия врача, категория, специальность, оклад, контактный телефон.
        Об обследовании должна содержаться следующая информация: название обследования, вид обследования, дата проведения обследования, стоимость обследования.
        О страховой компании должна содержаться следующая информация: название страховой компании, номер лицензии, фамилия руководителя, контактный телефон.""")

    sql_query = text_to_sql(
        test_text,
        loaded_entity_model,
        loaded_entity_tokenizer,
        loaded_entity_label_map,
        loaded_relation_model,
        loaded_relation_tokenizer,
        loaded_relation_label_map,
        device
    )

    print(sql_query)
