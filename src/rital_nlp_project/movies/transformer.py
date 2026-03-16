from nltk.stem.snowball import SnowballStemmer
import spacy

from rital_nlp_project.common.preprocess.transformer_base import TextPreprocessorBase
from rital_nlp_project.movies.lexicon import MAIL_PATTERN, PUNCTUATION, URL_PATTERN


class TextPreprocessor(TextPreprocessorBase):
    def __init__(self, *, stem=False, lemmatize=True, stopwords):
        super().__init__(stem=stem, lemmatize=lemmatize, stopwords=stopwords,
                         url_pattern=URL_PATTERN, mail_pattern=MAIL_PATTERN, punctuation=PUNCTUATION)
        self.lang = "english"

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
            self.lemmatizer_ = spacy.load("en_core_web_sm")
            self._func = self.lemmatize_tokens

        return self

    def stem_tokens(self, texts_tokens: list[str]) -> str:
        return [" ".join([self.stemmer_.stem(token) for token in text_tokens])
                for text_tokens in texts_tokens]

    def lemmatize_tokens(self, texts_tokens: list[str]) -> str:
        return [" ".join([token.lemma_ for token in self.lemmatizer_(" ".join(tokens))])
                for tokens in texts_tokens]
