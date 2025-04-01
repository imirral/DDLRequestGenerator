import re


def simple_string_similarity(a: str, b: str) -> float:
    a, b = a.lower(), b.lower()
    if a == b:
        return 1.0

    set_a = set(re.split(r'[\W_]+', a))
    set_b = set(re.split(r'[\W_]+', b))

    common = len(set_a.intersection(set_b))
    avg_len = (len(set_a) + len(set_b)) / 2.0
    if avg_len == 0:
        return 0.0
    return common / avg_len


def parse_sql_solution(sql_text: str):
    tables_info = {}

    create_table_pattern = re.compile(
        r'CREATE\s+TABLE\s+([A-Za-яЁё_0-9]+)\s*\((.*?)\);',
        re.IGNORECASE | re.DOTALL
    )

    for match in create_table_pattern.finditer(sql_text):
        table_name = match.group(1)
        columns_block = match.group(2)
        table_name = table_name.strip().lower()

        columns = []
        columns_lines = [c.strip() for c in columns_block.split(',')]

        for col_line in columns_lines:
            col_parts = col_line.split()
            if len(col_parts) < 2:
                continue
            col_name = col_parts[0].lower()
            col_type = col_parts[1].upper()

            if "VARCHAR" in col_type:
                col_type = "VARCHAR"
            if "DATE" in col_type:
                col_type = "DATE"
            if "INT" in col_type:
                col_type = "INT"
            if "DECIMAL" in col_type:
                col_type = "DECIMAL"
            if "BOOLEAN" in col_type:
                col_type = "BOOLEAN"

            columns.append({
                'name': col_name,
                'type': col_type
            })

        tables_info[table_name] = {
            'columns': columns,
            'foreign_keys': []
        }

    fk_pattern = re.compile(
        r'ALTER\s+TABLE\s+([A-Za-яЁё_0-9]+)\s+ADD\s+FOREIGN\s+KEY\s*\((.*?)\)\s+REFERENCES\s+([A-Za-яЁё_0-9]+)\s*\((.*?)\);',
        re.IGNORECASE
    )

    for match in fk_pattern.finditer(sql_text):
        current_table = match.group(1).strip().lower()
        fk_column = match.group(2).strip().lower()
        ref_table = match.group(3).strip().lower()
        ref_column = match.group(4).strip().lower()

        if current_table in tables_info:
            tables_info[current_table]['foreign_keys'].append((fk_column, ref_table, ref_column))

    return {
        'tables': tables_info
    }


def compare_table_names(table1_name: str, table2_name: str) -> float:
    return simple_string_similarity(table1_name, table2_name)


def compare_column_sets(columns1: list, columns2: list) -> float:
    max_attributes = max(len(columns1), len(columns2))
    if max_attributes == 0:
        return 1.0

    matched_indices = set()
    total_score = 0.0

    for col1 in columns1:
        best_score_for_col1 = 0.0
        best_j = None

        for j, col2 in enumerate(columns2):
            if j in matched_indices:
                continue

            concept_similarity = simple_string_similarity(col1['name'], col2['name'])
            type_similarity = 1.0 if col1['type'] == col2['type'] else 0.0

            col_score = 0.5 * concept_similarity + 0.5 * type_similarity

            if col_score > best_score_for_col1:
                best_score_for_col1 = col_score
                best_j = j

        if best_j is not None:
            matched_indices.add(best_j)
            total_score += best_score_for_col1

    return total_score / max_attributes


def compare_foreign_keys(fk_list_1: list, fk_list_2: list) -> float:
    max_fks = max(len(fk_list_1), len(fk_list_2))
    if max_fks == 0:
        return 1.0

    matched_indices = set()
    total_score = 0.0

    for fk1 in fk_list_1:
        best_score = 0.0
        best_j = None
        for j, fk2 in enumerate(fk_list_2):
            if j in matched_indices:
                continue
            col_similarity = simple_string_similarity(fk1[0], fk2[0])
            ref_table_similarity = simple_string_similarity(fk1[1], fk2[1])
            ref_column_similarity = simple_string_similarity(fk1[2], fk2[2])

            fk_score = (col_similarity + ref_table_similarity + ref_column_similarity) / 3.0

            if fk_score > best_score:
                best_score = fk_score
                best_j = j

        if best_j is not None:
            matched_indices.add(best_j)
            total_score += best_score

    return total_score / max_fks


def compare_solutions(solution_sql_1: str, solution_sql_2: str) -> float:
    parsed1 = parse_sql_solution(solution_sql_1)
    parsed2 = parse_sql_solution(solution_sql_2)

    tables1 = parsed1['tables']
    tables2 = parsed2['tables']

    max_entities = max(len(tables1), len(tables2))
    if max_entities == 0:
        return 1.0

    matched_tables_2 = set()
    total_score = 0.0

    for table_name_1, table_info_1 in tables1.items():
        best_score_for_table = 0.0
        best_table_2 = None

        for table_name_2, table_info_2 in tables2.items():
            if table_name_2 in matched_tables_2:
                continue

            name_score = compare_table_names(table_name_1, table_name_2)

            columns_score = compare_column_sets(
                table_info_1['columns'],
                table_info_2['columns']
            )

            fk_score = compare_foreign_keys(
                table_info_1['foreign_keys'],
                table_info_2['foreign_keys']
            )

            entity_score = 0.1 * name_score + 0.9 * (0.8 * columns_score + 0.2 * fk_score)

            if entity_score > best_score_for_table:
                best_score_for_table = entity_score
                best_table_2 = table_name_2

            if best_table_2 is not None:
                matched_tables_2.add(best_table_2)
                total_score += best_score_for_table

        final_score = total_score / max_entities

        return final_score


if __name__ == "__main__":

    with open("standard.txt", "r", encoding="utf-8") as f:
        sql_solution_1 = f.read()

    with open("solution.txt", "r", encoding="utf-8") as f:
        sql_solution_2 = f.read()

    score = compare_solutions(sql_solution_1, sql_solution_2)
    print(f"Similarity score между решениями: {score:.4f}")
