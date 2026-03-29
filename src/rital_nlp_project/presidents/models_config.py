from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

from rital_nlp_project.common.models.utils import make_lsa_vectorizer, get_word_embed_vectorizers
# from rital_nlp_project.presidents.models import SmoothedProbaClassifier

max_df = 0.5
min_df = 5
max_features = 10000
cv_n_splits = 5
random_state = 42
test_size = 0.2
fasttext_model = "cc.fr.300.bin"

scoring = {
    'accuracy': 'accuracy',
    'f1': make_scorer(f1_score, zero_division=0),
    'recall': make_scorer(recall_score, zero_division=0),
    'precision': make_scorer(precision_score, zero_division=0),
    'roc_auc': 'roc_auc',
    'pr_auc': 'average_precision'
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
    "count": ["nb", "logreg_2_to_1", "logreg_balanced"],
    "count_lsa": ["logreg_2_to_1", "logreg_balanced"],
    "tfidf": ["logreg_2_to_1", "logreg_balanced"],
    "tfidf_lsa": ["logreg_2_to_1", "logreg_balanced"],
    "fasttext": ["logreg_2_to_1", "logreg_balanced"],
    "cls": ["logreg_2_to_1", "logreg_balanced"]  # refers to CLS embedding
}

CLASSIFIERS = {
    "nb": MultinomialNB(),
    "logreg_2_to_1": LogisticRegression(max_iter=5000, class_weight={1: 2, 0: 1}),
    "logreg_balanced": LogisticRegression(max_iter=5000, class_weight="balanced"),
}


def get_vectorizers_by_type(models):
    vectorizers_by_type = {
        "count": COUNT_LIKE_VECTORIZERS,
        "count_lsa": COUNT_LIKE_LSA_VECTORIZERS,
        "tfidf": TFIDF_VECTORIZERS,
        "tfidf_lsa": TFIDF_LSA_VECTORIZERS,
        "fasttext": get_word_embed_vectorizers(models.get("fasttext", None), advanced_pool=False)
    }
    return vectorizers_by_type
