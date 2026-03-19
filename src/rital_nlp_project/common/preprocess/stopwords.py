from __future__ import annotations

from functools import lru_cache

from unidecode import unidecode

_STATIC_STOPWORDS = {
    "english": [
        "l", "yeah", "oh", "a", "about", "above", "after", "again", "against", "ain", "all",
        "am", "an", "and", "any", "are", "aren", "aren't", "arent", "as", "at", "be",
        "because", "been", "before", "being", "below", "between", "both", "but", "by", "can",
        "couldn", "couldn't", "couldnt", "d", "did", "didn", "didn't", "didnt", "do", "does",
        "doesn", "doesn't", "doesnt", "doing", "don", "don't", "dont", "down", "during",
        "each", "few", "for", "from", "further", "had", "hadn", "hadn't", "hadnt", "has",
        "hasn", "hasn't", "hasnt", "have", "haven", "haven't", "havent", "having", "he",
        "he'd", "he'll", "he's", "hed", "hell", "her", "here", "hers", "herself", "hes",
        "him", "himself", "his", "how", "i", "i'd", "i'll", "i'm", "i've", "id", "if", "ill",
        "im", "in", "into", "is", "isn", "isn't", "isnt", "it", "it'd", "it'll", "it's",
        "itd", "itll", "its", "itself", "ive", "just", "ll", "m", "ma", "me", "mightn",
        "mightn't", "mightnt", "more", "most", "mustn", "mustn't", "mustnt", "my", "myself",
        "needn", "needn't", "neednt", "no", "nor", "not", "now", "o", "of", "off", "on",
        "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own",
        "re", "s", "same", "shan", "shan't", "shant", "she", "she'd", "she'll", "she's",
        "shed", "shell", "shes", "should", "should've", "shouldn", "shouldn't", "shouldnt",
        "shouldve", "so", "some", "such", "t", "than", "that", "that'll", "thatll", "the",
        "their", "theirs", "them", "themselves", "then", "there", "these", "they", "they'd",
        "they'll", "they're", "they've", "theyd", "theyll", "theyre", "theyve", "this",
        "those", "through", "to", "too", "under", "until", "up", "ve", "very", "was", "wasn",
        "wasn't", "wasnt", "we", "we'd", "we'll", "we're", "we've", "wed", "well", "were",
        "weren", "weren't", "werent", "weve", "what", "when", "where", "which", "while",
        "who", "whom", "why", "will", "with", "won", "won't", "wont", "wouldn", "wouldn't",
        "wouldnt", "y", "you", "you'd", "you'll", "you're", "you've", "youd", "youll",
        "your", "youre", "yours", "yourself", "yourselves", "youve",
    ],
    "french": [
        "au", "aux", "avec", "ce", "ces", "dans", "de", "des", "du", "elle", "en", "et",
        "eux", "il", "ils", "je", "la", "le", "les", "leur", "lui", "ma", "mais", "me",
        "même", "mes", "moi", "mon", "ne", "nos", "notre", "nous", "on", "ou", "par", "pas",
        "pour", "qu", "que", "qui", "sa", "se", "ses", "son", "sur", "ta", "te", "tes", "toi",
        "ton", "tu", "un", "une", "vos", "votre", "vous", "c", "d", "j", "l", "à", "m", "n",
        "s", "t", "y", "été", "étée", "étées", "étés", "étant", "étante", "étants", "étantes",
        "suis", "es", "est", "sommes", "êtes", "sont", "serai", "seras", "sera", "serons",
        "serez", "seront", "serais", "serait", "serions", "seriez", "seraient", "étais",
        "était", "étions", "étiez", "étaient", "fus", "fut", "fûmes", "fûtes", "furent",
        "sois", "soit", "soyons", "soyez", "soient", "fusse", "fusses", "fût", "fussions",
        "fussiez", "fussent", "ayant", "ayante", "ayantes", "ayants", "eu", "eue", "eues",
        "eus", "ai", "as", "avons", "avez", "ont", "aurai", "auras", "aura", "aurons",
        "aurez", "auront", "aurais", "aurait", "aurions", "auriez", "auraient", "avais",
        "avait", "avions", "aviez", "avaient", "eut", "eûmes", "eûtes", "eurent", "aie",
        "aies", "ait", "ayons", "ayez", "aient", "eusse", "eusses", "eût", "eussions",
        "eussiez", "eussent",
    ],
}


def _normalized_tokens(values: list[str] | set[str]) -> set[str]:
    normalized = set()
    for value in values:
        token = value.strip().lower()
        if not token:
            continue
        normalized.add(token)
        normalized.add(unidecode(token))
    return normalized


@lru_cache(maxsize=None)
def get_stopwords(language: str) -> frozenset[str]:
    if language not in _STATIC_STOPWORDS:
        raise KeyError(f"Unsupported stopword language: {language}")

    values = set(_STATIC_STOPWORDS[language])
    try:
        from nltk.corpus import stopwords as nltk_stopwords

        values.update(nltk_stopwords.words(language))
    except (LookupError, ModuleNotFoundError):
        pass

    return frozenset(_normalized_tokens(values))


STOPWORDS = {language: get_stopwords(language) for language in _STATIC_STOPWORDS}

