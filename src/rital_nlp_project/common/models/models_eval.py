import time

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, get_scorer
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline


def eval_combination_matrix(X, y, vectorizer_list, classifier_list, scoring):
    results = []
    for vect in vectorizer_list:
        for clf in classifier_list:
            pipe = Pipeline([("vectorizer", vect), ("classifier", clf)])
            scores = cross_validate(pipe, X, y, scoring=scoring)
            # Print all available scores in the scoring dict
            # for score_name in scoring:
            #     print(
            #         f"Cross-validation {score_name} scores: {scores['test_' + score_name]}, mean: {np.mean(scores['test_' + score_name]):.4f}")
            results.append({
                "vectorizer": vect,
                "classifier": clf,
                **{score_name: np.mean(scores['test_' + score_name]) for score_name in scoring}
            })
    return pd.DataFrame(results)


def eval_pipeline(
    pipeline,  # pipeline or model
    X,
    y,
    *,
    split_fn=None,
    cv_fn=None,  # cv argument in cross validate
    scoring,
    clf_report: bool = False,
):
    print(pipeline)
    results = {}

    if cv_fn is not None:
        cv = cv_fn(X, y)
        scores = cross_validate(pipeline, X, y, scoring=scoring, cv=cv)
        for score_name in scoring:
            mean_value = float(np.mean(scores["test_" + score_name]))
            results["cv_" + score_name] = mean_value

    if split_fn is not None:
        split_out = split_fn(X, y)
        X_train, X_test, y_train, y_test = split_out

        # Training time
        t0 = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - t0
        results["train_train_time"] = train_time

        # Inference time
        t0 = time.time()
        predictions = pipeline.predict(X_test)
        infer_time = time.time() - t0

        results["test_infer_time"] = infer_time

        for score_name in scoring:
            scorer = get_scorer(scoring[score_name])
            score_value = scorer(pipeline, X_test, y_test)
            results[score_name] = score_value

        vectorizer = pipeline.named_steps["vect"]
        vocab_size = None
        if hasattr(vectorizer, "get_feature_names_out"):
            vocab_size = len(vectorizer.get_feature_names_out())
        results["train_vocab_size"] = vocab_size

        if clf_report:
            print(classification_report(y_test, predictions))

    return results


def eval_pipeline_movies(
    pipeline,
    X,
    y,
    *,
    cv: bool = False,
    split: bool = False,
    clf_report: bool = False,
):
    from rital_nlp_project.movies.models_config import scoring

    cv_fn = None
    split_fn = None

    if cv:
        from rital_nlp_project.movies.models_utils import cv_fn
    if split:
        from rital_nlp_project.movies.models_utils import split_fn

    return eval_pipeline(
        pipeline, X, y,
        split_fn=split_fn,
        cv_fn=cv_fn,
        scoring=scoring,
        clf_report=clf_report,
    )


def eval_pipeline_presidents(
    pipeline,
    X,
    y,
    *,
    cv: bool = False,
    split: bool = False,
    clf_report: bool = False,
):
    from rital_nlp_project.presidents.models_config import scoring

    cv_fn = None
    split_fn = None

    if cv:
        from rital_nlp_project.presidents.models_utils import cv_fn
    if split:
        from rital_nlp_project.presidents.models_utils import split_fn

    return eval_pipeline(
        pipeline, X, y,
        split_fn=split_fn,
        cv_fn=cv_fn,
        scoring=scoring,
        clf_report=clf_report,
    )


def add_experiments(vects, compatibility, classifiers):
    res = []
    for vect_type, vects in vects.items():
        if vects is None:
            continue

        for vect_name, vect in vects:
            for clf_key in compatibility[vect_type]:

                clf = classifiers[clf_key]
                name = f"{vect_name}__{clf_key}"

                pipeline = Pipeline([
                    ("vect", vect),
                    ("clf", clf)
                ])

                res.append({
                    "name": name,
                    "pipeline": pipeline
                })
    return res


def build_experiments(dataset, models=None):
    models = models or {}
    exps = {}

    if dataset == "movies":
        from rital_nlp_project.movies.models_config import (
            get_vectorizers_by_type,
            COMPATIBILITY,
            CLASSIFIERS,
        )
    elif dataset == "presidents":
        from rital_nlp_project.presidents.models_config import (
            get_vectorizers_by_type,
            COMPATIBILITY,
            CLASSIFIERS,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    vects = get_vectorizers_by_type(models)
    return add_experiments(vects, COMPATIBILITY, CLASSIFIERS)


def eval_experiments(experiments,
                     X,
                     y,
                     *,
                     eval_pipeline_fn,
                     cv: bool = False,
                     split: bool = False,
                     clf_report: bool = False
                     ):

    eval_report = dict()
    for exp in experiments:
        pipe_name = exp["name"]
        pipe = exp["pipeline"]
        eval_report[pipe_name] = eval_pipeline_fn(pipe, X, y, cv=cv, split=split, clf_report=clf_report)
    return pd.DataFrame(eval_report).T.rename_axis("pipeline").reset_index()
