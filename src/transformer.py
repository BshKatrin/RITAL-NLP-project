
from sklearn.base import BaseEstimator, TransformerMixin
from string import punctuation

import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer

from .preprocess import *


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, *, stem=False, lemmatize=True, lang="english", stopwords):
        # In sklearn __init__ method should only assign parameters, can't have any logic
        self.stem = stem
        self.lemmatize = lemmatize
        self.lang = lang
        self.stopwords = stopwords

    def fit(self, X: list[str], y=None):
        if self.stem and self.lemmatize:
            raise ValueError(
                f"{self.__class__.__name__}: cannot have both `stem` and `lemmatize` set to True."
            )

        self.stemmer_ = None
        self.lemmatizer_ = None
        self._func = self.identity

        if self.stem:
            self.stemmer_ = SnowballStemmer(self.lang)
            self._func = self.stem_tokens

        if self.lemmatize:
            self.lemmatizer_ = WordNetLemmatizer()
            self._func = self.lemmatize_tokens

        return self

    def transform(self, X: list[str]) -> list[str]:
        # Clean text
        texts = [self.preprocess_text(text) for text in X]

        # Tokenize
        texts = [self.tokenize(text) for text in texts]

        # Standardize
        texts = [self._func(text) for text in texts]

        return texts

    def tokenize(self, text: str) -> list[str]:
        return nltk.tokenize.word_tokenize(text)

    def identity(self, text: str) -> str:
        return " ".join(text)

    def stem_tokens(self, text_tokens: list[str]) -> str:
        return " ".join([self.stemmer_.stem(token) for token in text_tokens])

    def lemmatize_tokens(self, text_tokens: list[str]) -> str:
        pos_tags = nltk.pos_tag(text_tokens)
        return " ".join([self.lemmatizer_.lemmatize(word, penn_to_wordnet(pos))
                         for (word, pos) in pos_tags])

    def preprocess_text(self, text: str) -> str:
        text = text.lower()
        text = remove_emails(text)
        text = remove_urls(text)
        text = remove_digits(text)
        text = remove_punctuation(text)
        text = remove_stopwords(text, self.stopwords)
        return text
