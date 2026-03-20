
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer

from rital_nlp_project.common.models.word_embeds import WordEmbeddingsPoolingTransformer


def get_word_embed_vectorizers(model):
    if model is None:
        return []

    return [
        ("w2v_mean", WordEmbeddingsPoolingTransformer(model, pooling="mean")),
        ("w2v_max", WordEmbeddingsPoolingTransformer(model, pooling="max")),
        ("w2v_mean_max", WordEmbeddingsPoolingTransformer(model, pooling="mean_max")),
        ("w2v_tfidf",
         WordEmbeddingsPoolingTransformer(
             model,
             pooling="tfidf",
             vectorizer=TfidfVectorizer()
         )),
        ("w2v_sif",
         WordEmbeddingsPoolingTransformer(
             model,
             pooling="sif",
             vectorizer=CountVectorizer()
         )),
    ]


def make_lsa_vectorizer(base_vectorizer, n_components=1000):
    # 1000 : to explain at least 80% of variance
    return Pipeline([
        ("vect", base_vectorizer),
        ("svd", TruncatedSVD(n_components=n_components)),
        ("norm", Normalizer(norm="l2", copy=False)),
    ])
