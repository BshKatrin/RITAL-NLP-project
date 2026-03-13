import re
from unidecode import unidecode


def remove_emails(text: str, pattern: str) -> str:
    return re.sub(pattern, ' ', text) if pattern else text


def remove_urls(text: str, pattern: str) -> str:
    return re.sub(pattern, ' ', text) if pattern else text


def remove_digits(text: str) -> str:
    return re.sub(r'[0-9]+', '', text)


def remove_punctuation(text: str, punctuation: str) -> str:
    return text.lower().translate(str.maketrans(punctuation, ' ' * len(punctuation)))


def remove_stopwords(text: str, stopwords, apply_unidecode: bool = False) -> str:
    def normalize(word):
        return unidecode(word) if apply_unidecode else word
    return " ".join([word for word in text.split() if normalize(word) not in stopwords])


def normalize(text: str) -> str:
    return unidecode(text)
