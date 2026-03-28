from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from rital_nlp_project.common.models.utils import make_lsa_vectorizer, get_word_embed_vectorizers

max_df = 0.5
min_df = 5
max_features = 10000
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
    ("char_1gram_count", TfidfVectorizer(use_idf=False, analyzer="char", ngram_range=(1, 1), max_features=max_features)),
    ("char_4gram_count", TfidfVectorizer(use_idf=False, analyzer="char_wb", ngram_range=(4, 4), max_features=max_features)),
    ("word_1gram_count", TfidfVectorizer(use_idf=False, analyzer="word", ngram_range=(1, 1), max_features=max_features)),
    ("word_13gram_count", TfidfVectorizer(use_idf=False, analyzer="word", ngram_range=(1, 3), max_features=max_features)),
]

COUNT_LIKE_LSA_VECTORIZERS = [
    ("char_4gram_count_lsa", make_lsa_vectorizer(TfidfVectorizer(use_idf=False, analyzer="char_wb", ngram_range=(4, 4)))),
    ("word_1gram_count_lsa", make_lsa_vectorizer(TfidfVectorizer(use_idf=False, analyzer="word", ngram_range=(1, 1)))),
    ("word_13gram_count_lsa", make_lsa_vectorizer(TfidfVectorizer(use_idf=False, analyzer="word", ngram_range=(1, 3)))),
]

TFIDF_VECTORIZERS = [
    ("char_4gram_tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 4), max_features=max_features)),
    ("word_1gram_tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1, 1), max_features=max_features)),
    ("word_13gram_tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1, 3), max_features=max_features)),
]

TFIDF_LSA_VECTORIZERS = [
    ("char_4gram_tfidf_lsa", make_lsa_vectorizer(TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 4)))),
    ("word_1gram_tfidf_lsa", make_lsa_vectorizer(TfidfVectorizer(analyzer="word", ngram_range=(1, 1)))),
    ("word_13gram_tfidf_lsa", make_lsa_vectorizer(TfidfVectorizer(analyzer="word", ngram_range=(1, 3)))),
]


COMPATIBILITY = {
    "count": ["nb", "logreg", "svm"],
    "count_lsa": ["logreg", "svm"],
    "tfidf": ["logreg", "svm"],
    "tfidf_lsa": ["logreg", "svm"],
    "word2vec": ["logreg", "svm"],
    "fasttext": ["logreg", "svm"],
    "cls": ["logreg", "svm"]  # refers to CLS embedding
}

CLASSIFIERS = {
    "nb": MultinomialNB(),
    "logreg": LogisticRegression(),
    "svm": LinearSVC(),
}

def get_vectorizers_by_type(models):
    vectorizers_by_type = {
        "count": COUNT_LIKE_VECTORIZERS,
        "count_lsa": COUNT_LIKE_LSA_VECTORIZERS,
        "tfidf": TFIDF_VECTORIZERS,
        "tfidf_lsa": TFIDF_LSA_VECTORIZERS,
        "word2vec": get_word_embed_vectorizers(models.get("word2vec", None)),
        "fasttext": get_word_embed_vectorizers(models.get("fasttext", None))
    }
    return vectorizers_by_type
