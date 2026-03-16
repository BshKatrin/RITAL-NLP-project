
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin

import nltk
from rital_nlp_project.common.preprocess import *

import re
from unidecode import unidecode


class TextPreprocessorBase(ABC, BaseEstimator, TransformerMixin):
    def __init__(self, *, stem=False, lemmatize=True, stopwords,
                 url_pattern=None, mail_pattern=None, punctuation=None):
        # In sklearn __init__ method should only assign parameters, can't have any logic
        self.stem = stem
        self.lemmatize = lemmatize
        self.stopwords = stopwords
        self.url_pattern = url_pattern
        self.mail_pattern = mail_pattern
        self.punctuation = punctuation

    @abstractmethod
    def fit(self, X: list[str], y=None):
        pass

    def transform(self, X: list[str]) -> list[str]:
        # Clean text
        texts = [self.preprocess_text(text) for text in X]

        # Tokenize
        texts = [self.tokenize(text) for text in texts]

        # Standardize
        # texts = [self._func(texts) for text in texts]
        texts = self._func(texts)

        return texts

    def tokenize(self, text: str) -> list[str]:
        return nltk.tokenize.word_tokenize(text)

    def identity(self, texts: list[str]) -> str:
        return [" ".join(text) for text in texts]

    @abstractmethod
    def stem_tokens(self, texts_tokens: list[str]) -> str:
        pass

    @abstractmethod
    def lemmatize_tokens(self, texts_tokens: list[str]) -> str:
        pass

    def preprocess_text(self, text: str) -> str:
        text = text.lower()
        text = self.remove_emails(text, self.mail_pattern)
        text = self.remove_urls(text, self.url_pattern)
        text = self.remove_digits(text)
        text = self.remove_punctuation(text, self.punctuation)
        text = self.normalize(text)
        text = self.remove_stopwords(text, self.stopwords)
        return text

    def remove_emails(self, text: str, pattern: str) -> str:
        return re.sub(pattern, ' ', text) if pattern else text

    def remove_urls(self, text: str, pattern: str) -> str:
        return re.sub(pattern, ' ', text) if pattern else text

    def remove_digits(self, text: str) -> str:
        return re.sub(r'[0-9]+', '', text)

    def remove_punctuation(self, text: str, punctuation: str) -> str:
        return text.lower().translate(str.maketrans(punctuation, ' ' * len(punctuation)))

    def remove_stopwords(self, text: str, stopwords, apply_unidecode: bool = False) -> str:
        def normalize(word):
            return unidecode(word) if apply_unidecode else word
        return " ".join([word for word in text.split() if normalize(word) not in stopwords])

    def normalize(self, text: str) -> str:
        return unidecode(text)
