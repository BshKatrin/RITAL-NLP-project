
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin

import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer

from src.common.preprocess import *


class TextPreprocessorBase(ABC, BaseEstimator, TransformerMixin):
    def __init__(self, *, stem=False, lemmatize=True, stopwords):
        # In sklearn __init__ method should only assign parameters, can't have any logic
        self.stem = stem
        self.lemmatize = lemmatize
        self.stopwords = stopwords

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
        text = remove_emails(text)
        text = remove_urls(text)
        text = remove_digits(text)
        text = remove_punctuation(text)
        text = remove_stopwords(text, self.stopwords)
        return text
