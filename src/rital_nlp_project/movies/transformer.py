from nltk.stem.snowball import SnowballStemmer

from rital_nlp_project.common.preprocess.transformer_base import TextPreprocessorBase
from rital_nlp_project.movies.lexicon import MAIL_PATTERN, PUNCTUATION, URL_PATTERN, PHONE_PATTERN, EXTRA_SUB


class TextPreprocessor(TextPreprocessorBase):
    def __init__(self, *, stem=False, lemmatize=True, stopwords, pipeline_mode, extra_substitutions=None):
        super().__init__(stem=stem,
                         lemmatize=lemmatize,
                         stopwords=stopwords,
                         url_pattern=URL_PATTERN,
                         mail_pattern=MAIL_PATTERN,
                         phone_pattern=PHONE_PATTERN,
                         punctuation=PUNCTUATION,
                         extra_sub=EXTRA_SUB,
                         pipeline_mode=pipeline_mode,
                         lang="english")
