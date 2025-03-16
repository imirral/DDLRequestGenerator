from natasha import Segmenter, NewsEmbedding, NewsMorphTagger, MorphVocab,Doc

segmenter = Segmenter()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
morph_vocab = MorphVocab()


def to_nominative_plural(phrase):
    doc = Doc(phrase)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)

    words = []
    for token in doc.tokens:
        parse = morph_vocab.parse(token.text)
        if not parse:
            words.append(token.text)
            continue

        best = parse[0]
        lemma = best.normal

        grammemes_for_nominative_plural = {'Plur', 'Nom'}
        inflected = best.inflect(grammemes_for_nominative_plural)
        if inflected:
            transformed = inflected.word
        else:
            transformed = lemma

        words.append(transformed)
    return ' '.join(words)


def to_nominative_singular(phrase):
    doc = Doc(phrase)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)

    words = []
    for token in doc.tokens:
        parse = morph_vocab.parse(token.text)
        if not parse:
            words.append(token.text)
            continue

        best = parse[0]
        lemma = best.normal

        grammemes_for_nominative_singular = {'Sing', 'Nom'}
        inflected = best.inflect(grammemes_for_nominative_singular)
        if inflected:
            transformed = inflected.word
        else:
            transformed = lemma

        words.append(transformed)
    return ' '.join(words)


def merge_adjacent_entities(predicted_entities, max_gap=1):
    if not predicted_entities:
        return []

    predicted_entities = sorted(predicted_entities, key=lambda x: x['start'])
    merged = []
    current = predicted_entities[0]

    for i in range(1, len(predicted_entities)):
        nxt = predicted_entities[i]
        if (nxt['label'] == current['label']
                and nxt['start'] - current['end'] <= max_gap):
            if nxt['text'].startswith("##"):
                current['text'] += nxt['text'][2:]
            else:
                current['text'] += ' ' + nxt['text']
            current['end'] = nxt['end']
        else:
            merged.append(current)
            current = nxt

    merged.append(current)
    return merged


def postprocess_entities(predicted_entities):
    merged = merge_adjacent_entities(predicted_entities)

    for ent in merged:
        if ent['label'] == 'ENTITY':
            ent['text'] = to_nominative_plural(ent['text'])
        else:
            ent['text'] = to_nominative_singular(ent['text'])

    return merged
