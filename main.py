from classes.in_out import In_Out
from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    NamesExtractor,
    Doc
)

# Экземпляры классов

in_out = In_Out()

segmenter = Segmenter()
morph_vocab = MorphVocab()

emb = NewsEmbedding()

morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
ner_tagger = NewsNERTagger(emb)

names_extractor = NamesExtractor(morph_vocab)

# Считывание текста

file_name = '0'

text = in_out.read_txt_file(file_name)
doc = Doc(text)

# Сегментация

doc.segment(segmenter)
print(doc.tokens[:5])
print(doc.sents[:5])

print()

# Морфологический разбор

doc.tag_morph(morph_tagger)
print(doc.tokens[:5])
doc.sents[0].morph.print()

print()

# Лемматизация

for token in doc.tokens:
    token.lemmatize(morph_vocab)

print(doc.tokens[:5])

token_dict = {token.text: token.lemma for token in doc.tokens[:5]}

for key, value in token_dict.items():
    print(f"{key}: {value}")

print()

# Синтаксический разбор

doc.parse_syntax(syntax_parser)
print(doc.tokens[:5])
doc.sents[0].syntax.print()
