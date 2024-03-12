class In_Out:

    def read_txt_file(self, file_name, encoding='utf-8'):
        file_path = 'data/' + file_name + '.txt'

        try:
            with open(file_path, 'r', encoding=encoding) as file:
                data = file.read()
            return data
        except FileNotFoundError:
            print(f"Файл {file_path} не найден.")
            return None
