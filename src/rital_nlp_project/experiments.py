from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from rital_nlp_project.common.models.word2vec import Word2VecPoolingTransformer
from rital_nlp_project.movies.models_config import random_state as movies_random_state
from rital_nlp_project.presidents.models import SmoothedProbaClassifier
from rital_nlp_project.presidents.models_config import random_state as presidents_random_state


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    task: str
    feature_kind: str
    build: Callable[[], object]
    description: str = ""


def _movies_text_experiments() -> list[ExperimentSpec]:
    max_features = 5000
    min_df = 5
    max_df = 0.5

    return [
        ExperimentSpec(
            name="movies_tfidf_word12_logreg",
            task="movies",
            feature_kind="text",
            description="TF-IDF word 1-2 grams with logistic regression",
            build=lambda: Pipeline(
                [
                    (
                        "vectorizer",
                        TfidfVectorizer(
                            analyzer="word",
                            ngram_range=(1, 2),
                            min_df=min_df,
                            max_df=max_df,
                            max_features=max_features,
                            sublinear_tf=True,
                        ),
                    ),
                    ("classifier", LogisticRegression(max_iter=2000, random_state=movies_random_state)),
                ]
            ),
        ),
        ExperimentSpec(
            name="movies_tfidf_word12_charwb35_logreg",
            task="movies",
            feature_kind="text",
            description="Combined TF-IDF word 1-2 grams and char wb 3-5 grams with logistic regression",
            build=lambda: Pipeline(
                [
                    (
                        "vectorizer",
                        FeatureUnion(
                            [
                                (
                                    "word",
                                    TfidfVectorizer(
                                        analyzer="word",
                                        ngram_range=(1, 2),
                                        min_df=min_df,
                                        max_df=max_df,
                                        max_features=max_features,
                                        sublinear_tf=True,
                                    ),
                                ),
                                (
                                    "char",
                                    TfidfVectorizer(
                                        analyzer="char_wb",
                                        ngram_range=(3, 5),
                                        min_df=min_df,
                                        max_df=max_df,
                                        max_features=max_features,
                                        sublinear_tf=True,
                                    ),
                                ),
                            ]
                        ),
                    ),
                    ("classifier", LogisticRegression(max_iter=2000, random_state=movies_random_state)),
                ]
            ),
        ),
        ExperimentSpec(
            name="movies_tfidf_charwb35_linearsvc",
            task="movies",
            feature_kind="text",
            description="TF-IDF character wb 3-5 grams with linear SVC",
            build=lambda: Pipeline(
                [
                    (
                        "vectorizer",
                        TfidfVectorizer(
                            analyzer="char_wb",
                            ngram_range=(3, 5),
                            min_df=min_df,
                            max_df=max_df,
                            max_features=max_features,
                            sublinear_tf=True,
                        ),
                    ),
                    ("classifier", LinearSVC(max_iter=5000)),
                ]
            ),
        ),
        ExperimentSpec(
            name="movies_count_word12_nb",
            task="movies",
            feature_kind="text",
            description="Count word 1-2 grams with MultinomialNB",
            build=lambda: Pipeline(
                [
                    (
                        "vectorizer",
                        CountVectorizer(
                            analyzer="word",
                            ngram_range=(1, 2),
                            min_df=min_df,
                            max_df=max_df,
                            max_features=max_features,
                        ),
                    ),
                    ("classifier", MultinomialNB()),
                ]
            ),
        ),
    ]


def _movies_embedding_experiments() -> list[ExperimentSpec]:
    return [
        ExperimentSpec(
            name="movies_embeddings_logreg",
            task="movies",
            feature_kind="embeddings",
            description="Stored transformer embeddings with logistic regression",
            build=lambda: LogisticRegression(max_iter=2000, random_state=movies_random_state),
        ),
        ExperimentSpec(
            name="movies_embeddings_linearsvc",
            task="movies",
            feature_kind="embeddings",
            description="Stored transformer embeddings with linear SVC",
            build=lambda: LinearSVC(max_iter=5000),
        ),
    ]


def _movies_word2vec_experiments(word2vec_model) -> list[ExperimentSpec]:
    return [
        ExperimentSpec(
            name="movies_word2vec_mean_logreg",
            task="movies",
            feature_kind="text",
            description="Word2Vec mean pooling with logistic regression",
            build=lambda: Pipeline(
                [
                    ("vectorizer", Word2VecPoolingTransformer(word2vec_model, pooling="mean")),
                    ("classifier", LogisticRegression(max_iter=2000, random_state=movies_random_state)),
                ]
            ),
        ),
        ExperimentSpec(
            name="movies_word2vec_tfidf_logreg",
            task="movies",
            feature_kind="text",
            description="Word2Vec TF-IDF pooling with logistic regression",
            build=lambda: Pipeline(
                [
                    (
                        "vectorizer",
                        Word2VecPoolingTransformer(
                            word2vec_model,
                            pooling="tfidf",
                            vectorizer=TfidfVectorizer(min_df=5, max_df=0.5),
                        ),
                    ),
                    ("classifier", LogisticRegression(max_iter=2000, random_state=movies_random_state)),
                ]
            ),
        ),
        ExperimentSpec(
            name="movies_word2vec_sif_linearsvc",
            task="movies",
            feature_kind="text",
            description="Word2Vec SIF pooling with linear SVC",
            build=lambda: Pipeline(
                [
                    (
                        "vectorizer",
                        Word2VecPoolingTransformer(
                            word2vec_model,
                            pooling="sif",
                            vectorizer=CountVectorizer(min_df=5, max_df=0.5),
                        ),
                    ),
                    ("classifier", LinearSVC(max_iter=5000)),
                ]
            ),
        ),
    ]


def _presidents_text_experiments() -> list[ExperimentSpec]:
    max_features = 5000
    min_df = 5
    max_df = 0.5

    return [
        ExperimentSpec(
            name="presidents_tfidf_word12_logreg",
            task="presidents",
            feature_kind="text",
            description="TF-IDF word 1-2 grams with logistic regression",
            build=lambda: Pipeline(
                [
                    (
                        "vectorizer",
                        TfidfVectorizer(
                            analyzer="word",
                            ngram_range=(1, 2),
                            min_df=min_df,
                            max_df=max_df,
                            max_features=max_features,
                            sublinear_tf=True,
                        ),
                    ),
                    ("classifier", LogisticRegression(max_iter=2000, random_state=presidents_random_state)),
                ]
            ),
        ),
        ExperimentSpec(
            name="presidents_tfidf_word12_balanced_logreg",
            task="presidents",
            feature_kind="text",
            description="TF-IDF word 1-2 grams with balanced logistic regression",
            build=lambda: Pipeline(
                [
                    (
                        "vectorizer",
                        TfidfVectorizer(
                            analyzer="word",
                            ngram_range=(1, 2),
                            min_df=min_df,
                            max_df=max_df,
                            max_features=max_features,
                            sublinear_tf=True,
                        ),
                    ),
                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=2000,
                            random_state=presidents_random_state,
                            class_weight="balanced",
                        ),
                    ),
                ]
            ),
        ),
        ExperimentSpec(
            name="presidents_tfidf_word12_balanced_linearsvc",
            task="presidents",
            feature_kind="text",
            description="TF-IDF word 1-2 grams with balanced linear SVC",
            build=lambda: Pipeline(
                [
                    (
                        "vectorizer",
                        TfidfVectorizer(
                            analyzer="word",
                            ngram_range=(1, 2),
                            min_df=min_df,
                            max_df=max_df,
                            max_features=max_features,
                            sublinear_tf=True,
                        ),
                    ),
                    ("classifier", LinearSVC(max_iter=5000, class_weight="balanced")),
                ]
            ),
        ),
        ExperimentSpec(
            name="presidents_tfidf_word12_charwb35_logreg",
            task="presidents",
            feature_kind="text",
            description="Combined TF-IDF word 1-2 grams and char wb 3-5 grams with logistic regression",
            build=lambda: Pipeline(
                [
                    (
                        "vectorizer",
                        FeatureUnion(
                            [
                                (
                                    "word",
                                    TfidfVectorizer(
                                        analyzer="word",
                                        ngram_range=(1, 2),
                                        min_df=min_df,
                                        max_df=max_df,
                                        max_features=max_features,
                                        sublinear_tf=True,
                                    ),
                                ),
                                (
                                    "char",
                                    TfidfVectorizer(
                                        analyzer="char_wb",
                                        ngram_range=(3, 5),
                                        min_df=min_df,
                                        max_df=max_df,
                                        max_features=max_features,
                                        sublinear_tf=True,
                                    ),
                                ),
                            ]
                        ),
                    ),
                    ("classifier", LogisticRegression(max_iter=2000, random_state=presidents_random_state)),
                ]
            ),
        ),
        ExperimentSpec(
            name="presidents_count_word13_nb",
            task="presidents",
            feature_kind="text",
            description="Count word 1-3 grams with MultinomialNB",
            build=lambda: Pipeline(
                [
                    (
                        "vectorizer",
                        CountVectorizer(
                            analyzer="word",
                            ngram_range=(1, 3),
                            min_df=min_df,
                            max_df=max_df,
                            max_features=max_features,
                        ),
                    ),
                    ("classifier", MultinomialNB()),
                ]
            ),
        ),
        ExperimentSpec(
            name="presidents_tfidf_word12_smoothed_logreg",
            task="presidents",
            feature_kind="text",
            description="TF-IDF word 1-2 grams with smoothed logistic regression",
            build=lambda: Pipeline(
                [
                    (
                        "vectorizer",
                        TfidfVectorizer(
                            analyzer="word",
                            ngram_range=(1, 2),
                            min_df=min_df,
                            max_df=max_df,
                            max_features=max_features,
                            sublinear_tf=True,
                        ),
                    ),
                    (
                        "classifier",
                        SmoothedProbaClassifier(
                            LogisticRegression(max_iter=2000, random_state=presidents_random_state),
                            sigma=0.7,
                        ),
                    ),
                ]
            ),
        ),
    ]


def _presidents_embedding_experiments() -> list[ExperimentSpec]:
    return [
        ExperimentSpec(
            name="presidents_embeddings_logreg",
            task="presidents",
            feature_kind="embeddings",
            description="Stored transformer embeddings with logistic regression",
            build=lambda: LogisticRegression(max_iter=2000, random_state=presidents_random_state),
        ),
        ExperimentSpec(
            name="presidents_embeddings_balanced_logreg",
            task="presidents",
            feature_kind="embeddings",
            description="Stored transformer embeddings with balanced logistic regression",
            build=lambda: LogisticRegression(
                max_iter=2000,
                random_state=presidents_random_state,
                class_weight="balanced",
            ),
        ),
        ExperimentSpec(
            name="presidents_embeddings_smoothed_logreg",
            task="presidents",
            feature_kind="embeddings",
            description="Stored transformer embeddings with smoothed logistic regression",
            build=lambda: SmoothedProbaClassifier(
                LogisticRegression(max_iter=2000, random_state=presidents_random_state),
                sigma=0.7,
            ),
        ),
        ExperimentSpec(
            name="presidents_embeddings_linearsvc",
            task="presidents",
            feature_kind="embeddings",
            description="Stored transformer embeddings with linear SVC",
            build=lambda: LinearSVC(max_iter=5000),
        ),
    ]


def get_experiments(task: str, *, word2vec_model=None) -> list[ExperimentSpec]:
    if task == "movies":
        experiments = _movies_text_experiments() + _movies_embedding_experiments()
        if word2vec_model is not None:
            experiments += _movies_word2vec_experiments(word2vec_model)
        return experiments
    if task == "presidents":
        return _presidents_text_experiments() + _presidents_embedding_experiments()
    raise ValueError(f"Unknown task: {task}")


def get_experiment(name: str, *, word2vec_model=None) -> ExperimentSpec:
    for task in ("movies", "presidents"):
        for experiment in get_experiments(task, word2vec_model=word2vec_model):
            if experiment.name == name:
                return experiment
    raise KeyError(f"Unknown experiment: {name}")


def load_word2vec_model(path: str):
    try:
        from gensim.models import KeyedVectors, Word2Vec
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            "gensim is not installed. Install it to run Word2Vec experiments."
        ) from exc

    model_path = Path(path)
    suffix = model_path.suffix.lower()

    if suffix == ".model":
        return Word2Vec.load(str(model_path)).wv
    if suffix in {".kv", ".kvx"}:
        return KeyedVectors.load(str(model_path))
    if suffix in {".bin", ".gz"}:
        return KeyedVectors.load_word2vec_format(str(model_path), binary=True)
    return KeyedVectors.load_word2vec_format(str(model_path), binary=False)


def train_word2vec_model(
    texts: list[str],
    *,
    output_path: str,
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 2,
    sg: int = 1,
    epochs: int = 10,
    workers: int = 1,
    seed: int = 42,
):
    try:
        from gensim.models import Word2Vec
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            "gensim is not installed. Install it to train Word2Vec models."
        ) from exc

    sentences = [text.split() for text in texts]
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        epochs=epochs,
        workers=workers,
        seed=seed,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output))
    return output
