
from rital_nlp_project.common.preprocess import *
from rital_nlp_project.common.preprocess.transformer_base import TextPreprocessorBase
from rital_nlp_project.presidents.lexicon import PUNCTUATION


class TextPreprocessor(TextPreprocessorBase):
    def __init__(self, *, stem=False, lemmatize=True, stopwords, pipeline_mode):
        super().__init__(stem=stem,
                         lemmatize=lemmatize,
                         stopwords=stopwords,
                         punctuation=PUNCTUATION,
                         pipeline_mode=pipeline_mode,
                         lang="french"
                         )
