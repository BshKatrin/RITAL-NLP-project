from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from rital_nlp_project.common.models.word_embeds import WordEmbeddingsPoolingTransformer
from rital_nlp_project.presidents.models import SmoothedProbaClassifier

# TODO: define LSA for for tfidf, bow

max_df = 0.5
min_df = 5
cv_n_splits = 5
random_state = 42
test_size = 0.2
fasttext_model = "cc.fr.300.bin"

scoring = {
    'accuracy': 'accuracy',
    'f1': 'f1',
    'recall': 'recall',
    'precision': 'precision',
    'roc_auc': 'roc_auc'
}

COUNT_LIKE_VECTORIZERS = [
    ("char_1gram_count", TfidfVectorizer(use_idf=False, analyzer="char", ngram_range=(1, 1))),
    ("word_1gram_count", TfidfVectorizer(use_idf=False, analyzer="word", ngram_range=(1, 1))),
]

TFIDF_VECTORIZERS = [
    ("word_1gram_tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1, 1))),
    ("word_13gram_tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1, 3))),
]


def get_word_embed_vectorizers(model):
    return [
        ("w2v_mean", WordEmbeddingsPoolingTransformer(model, pooling="mean")),
        ("w2v_max", WordEmbeddingsPoolingTransformer(model, pooling="max")),
        ("w2v_mean_max", WordEmbeddingsPoolingTransformer(model, pooling="mean_max")),
        ("w2v_tfidf",
         WordEmbeddingsPoolingTransformer(
             model,
             pooling="tfidf",
             vectorizer=TfidfVectorizer(min_df=min_df, max_df=max_df)
         )),
        ("w2v_sif",
         WordEmbeddingsPoolingTransformer(
             model,
             pooling="sif",
             vectorizer=CountVectorizer(min_df=min_df, max_df=max_df)
         )),
    ]


COMPATIBILITY = {
    "count": ["nb", "logreg", "svm", "smooth_nb"],
    "tfidf": ["logreg", "smooth_logreg", "svm"],
    "fasttext": ["logreg", "smooth_logreg", "svm"],
    "cls": ["logreg", "smooth_logreg", "svm"]
}

CLASSIFIERS = {
    "nb": MultinomialNB(),
    "logreg": LogisticRegression(),
    "svm": LinearSVC(),
    "smooth_logreg": SmoothedProbaClassifier(LogisticRegression(), sigma=1),
    "smooth_nb": SmoothedProbaClassifier(MultinomialNB(), sigma=1),
}


def get_vectorizers_by_type():
    vectorizers_by_type = {
        "count": COUNT_LIKE_VECTORIZERS,
        "tfidf": TFIDF_VECTORIZERS,
        "fasttext": get_word_embed_vectorizers(fasttext_model)
    }
    return vectorizers_by_type
