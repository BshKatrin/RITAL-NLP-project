from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from rital_nlp_project.common.models.word_embeds import WordEmbeddingsPoolingTransformer

# TODO: define LSA for for tfidf, bow

max_df = 0.5
min_df = 5
cv_n_splits = 5
random_state = 42
test_size = 0.2
word2vec_model = "word2vec-google-news-300"
fasttext_model = "cc.en.300.bin"

scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
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
    "count": ["nb", "logreg", "svm"],
    "tfidf": ["logreg", "svm"],
    "word2vec": ["logreg", "svm"],
    "fasttext": ["logreg", "svm"],
    "cls": ["logreg", "svm"]
}

CLASSIFIERS = {
    "nb": MultinomialNB(),
    "logreg": LogisticRegression(),
    "svm": LinearSVC(),
}


def get_vectorizers_by_type():
    vectorizers_by_type = {
        "count": COUNT_LIKE_VECTORIZERS,
        "tfidf": TFIDF_VECTORIZERS,
        "word2vec": get_word_embed_vectorizers(word2vec_model),
        "fasttext": get_word_embed_vectorizers(fasttext_model)
    }
    return vectorizers_by_type


# def build_experiment_specs(
#     dataset_name,
#     *,
#     word2vec_model=None,
# ):
#     vectorizers_by_type = get_vectorizers_by_type(word2vec_model=word2vec_model)
#     active_types = enabled_vectorizer_types or list(vectorizers_by_type.keys())
#     active_classifiers = set(enabled_classifiers) if enabled_classifiers else None

#     experiments = []

#     for vect_type in active_types:
#         if vect_type not in vectorizers_by_type:
#             continue

#         allowed_clf_keys = COMPATIBILITY.get(vect_type, [])

#         for vect_name, vect in vectorizers_by_type[vect_type]:
#             for clf_key in allowed_clf_keys:
#                 if clf_key not in CLASSIFIERS:
#                     continue
#                 if active_classifiers is not None and clf_key not in active_classifiers:
#                     continue

#                 experiments.append({
#                     "dataset": dataset_name,
#                     "vect_type": vect_type,
#                     "vect_name": vect_name,
#                     "clf_key": clf_key,
#                     "name": f"{dataset_name}__{vect_name}__{clf_key}",
#                     "vectorizer": vect,
#                     "classifier": CLASSIFIERS[clf_key],
#                 })

#     return experiments
