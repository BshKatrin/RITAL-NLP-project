from abc import ABC, abstractmethod
import re
from typing import Sequence

from sklearn.base import BaseEstimator, TransformerMixin
from unidecode import unidecode
import nltk
from nltk.stem.snowball import SnowballStemmer

from rital_nlp_project.common.preprocess import *


class TextPreprocessorBase(ABC, BaseEstimator, TransformerMixin):
    def __init__(
            self,
            *,
            stem=False,
            lemmatize=True,
            stopwords,
            url_pattern=None,
            mail_pattern=None,
            phone_pattern=None,
            punctuation=None,
            extra_sub=None,
            pipeline_mode="classic",  # classic or bert
            lang,  # english or french,
    ):
        # In sklearn __init__ method should only assign parameters, can't have any logic
        self.stem = stem
        self.lemmatize = lemmatize
        self.stopwords = stopwords
        self.url_pattern = url_pattern
        self.mail_pattern = mail_pattern
        self.phone_pattern = phone_pattern
        self.punctuation = punctuation
        self.extra_sub = extra_sub
        self.pipeline_mode = pipeline_mode
        self.lang = lang

    def fit(self, X: list[str], y=None):
        self.special_tokens_mode_ = "tokens" if self.pipeline_mode == "bert" else "space"
        self.lemmatize = False if self.pipeline_mode == "bert" else self.lemmatize
        self.stem = False if self.pipeline_mode == "bert" else self.stem

        if self.stem and self.lemmatize:
            raise ValueError(
                f"{self.__class__.__name__}: cannot have both `stem` and `lemmatize` set to True."
            )

        self.stemmer_ = None
        self.lemmatizer_ = None
        self._func = None

        if self.stem:
            self.stemmer_ = self._get_stemmer()
            self._func = self.stem_tokens

        if self.lemmatize:
            self.lemmatizer_ = self._get_lemmatizer()
            self._func = self.lemmatize_tokens
        return self

    def _get_lemmatizer(self):
        import spacy

        if self.lang == "french":
            return spacy.load("fr_core_news_md")

        # english
        return spacy.load("en_core_web_sm")

    def _get_stemmer(self):
        return SnowballStemmer(self.lang)

    def transform(self, X: list[str]) -> list[str]:
        # Clean text
        texts = [self.preprocess_text(text) for text in X]

        if self.pipeline_mode == "classic":
            # Tokenize
            texts = [self.tokenize(text) for text in texts]

            # Standardize
            texts = self._func(texts) if self._func else texts
        return texts

    def tokenize(self, text: str) -> list[str]:
        return nltk.tokenize.word_tokenize(text)

    # def identity(self, texts: list[str]) -> str:
    #     return [" ".join(text) for text in texts]

    def stem_tokens(self, texts_tokens: list[str]) -> str:
        return [" ".join([self.stemmer_.stem(token) for token in text_tokens])
                for text_tokens in texts_tokens]

    def lemmatize_tokens(self, texts_tokens: list[str]) -> str:
        return [" ".join([token.lemma_ for token in self.lemmatizer_(" ".join(tokens))])
                for tokens in texts_tokens]

    def preprocess_text(self, text: str) -> str:
        if self.pipeline_mode == "bert":
            text = self._preprocess_bert(text)

        if self.pipeline_mode == "classic":
            text = self._preprocess_classic(text)

        return text

    def _preprocess_classic(self, text: str) -> str:
        text = text.lower()
        text = self.apply_extra_sub(text)
        text = self.remove_emails(text, self.mail_pattern)
        text = self.remove_urls(text, self.url_pattern)
        text = self.remove_phones(text, self.phone_pattern)
        text = self.remove_digits(text)
        text = self.remove_punctuation(text, self.punctuation)
        text = self.normalize(text)
        text = self.remove_stopwords(text, self.stopwords)
        return text

    def _preprocess_bert(self, text: str) -> str:
        text = self.remove_emails(text, self.mail_pattern)
        text = self.remove_urls(text, self.url_pattern)
        text = self.remove_phones(text, self.phone_pattern)
        return text

    def apply_extra_sub(self, text: str) -> str:
        if not self.extra_sub:
            return text

        for pattern, replacement in self.extra_sub:
            text = re.sub(pattern, replacement, text)

        return text

    def remove_emails(self, text: str, pattern: str) -> str:
        if not pattern:
            return text

        if self.special_tokens_mode_ == "tokens":
            return re.sub(pattern, ' <EMAIL> ', text)
        return re.sub(pattern, ' ', text)

    def remove_urls(self, text: str, pattern: str) -> str:
        if not pattern:
            return text
        if self.special_tokens_mode_ == "tokens":
            return re.sub(pattern, ' <URL> ', text)
        return re.sub(pattern, ' ', text)

    def remove_phones(self, text: str, pattern: str) -> str:
        # print("remove_phones")
        if not pattern:
            return text
        if self.special_tokens_mode_ == "tokens":
            return re.sub(pattern, ' <PHONE> ', text)
        return re.sub(pattern, ' ', text)

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
