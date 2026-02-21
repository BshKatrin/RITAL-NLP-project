
from sklearn.base import BaseEstimator, TransformerMixin

import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer

from src.common.preprocess import *
from src.common.transformer_base import TextPreprocessorBase


class TextPreprocessor(TextPreprocessorBase):
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

    def stem_tokens(self, texts_tokens: list[str]) -> str:
        return [" ".join([self.stemmer_.stem(token) for token in text_tokens])
                for text_tokens in texts_tokens]

    def lemmatize_tokens(self, texts_tokens: list[str]) -> str:
        texts_pos_tags = nltk.pos_tag_sents(texts_tokens)
        return [" ".join([self.lemmatizer_.lemmatize(word, penn_to_wordnet(pos)) for (word, pos) in pos_tags])
                for pos_tags in texts_pos_tags]
