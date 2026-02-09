from .lexicon import URL_PATTERN, MAIL_PATTERN, STOPWORDS_NO_PUNC, PUNCTUATION
import re


def remove_emails(text: str) -> str:
    return re.sub(MAIL_PATTERN, ' ', text)


def remove_urls(text: str) -> str:
    return re.sub(URL_PATTERN, ' ', text)


def remove_digits(text: str) -> str:
    return re.sub(r'[0-9]+', '', text)


def remove_punctuation(text: str) -> str:
    return text.lower().translate(str.maketrans('', '', PUNCTUATION))


def remove_stopwords(text: str, stopwords) -> str:  # should be called after removing
    return " ".join([word for word in text.split() if word not in stopwords])


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = remove_emails(text)
    text = remove_urls(text)
    text = remove_digits(text)
    text = remove_punctuation(text)
    text = remove_stopwords(text, STOPWORDS_NO_PUNC)
    return text
