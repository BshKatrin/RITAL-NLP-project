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

    results = {}

    if hasattr(pipeline, "named_steps"):
        for name, step in pipeline.named_steps.items():
            print(f"  {name}: {step}")
            results[name] = step
    else:
        print(f"  model: {pipeline}")
        results["model"] = pipeline

    if cv_fn is not None:
        cv = cv_fn(X, y)
        scores = cross_validate(pipeline, X, y, scoring=scoring, cv=cv)
        for score_name in scoring:
            mean_value = float(np.mean(scores["test_" + score_name]))
            results["cv_" + score_name] = mean_value
            print(f"Cross-validation {score_name} mean: {mean_value:.4f}")

    if split_fn is not None:
        split_out = split_fn(X, y)
        X_train, X_test, y_train, y_test = split_out

        # Training time
        t0 = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - t0
        print(f"Training time: {train_time:.3f} seconds")

        # Inference time
        t0 = time.time()
        predictions = pipeline.predict(X_test)
        infer_time = time.time() - t0
        print(f"Inference time: {infer_time:.3f} seconds")

        for score_name in scoring:
            scorer = get_scorer(scoring[score_name])
            score_value = scorer(pipeline, X_test, y_test)
            results[score_name] = score_value
            print(f"Test {score_name}: {score_value:.4f}")

        if clf_report:
            print(classification_report(y_test, predictions))
        print()

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
        from rital_nlp_project.movies.models import cv_fn
    if split:
        from rital_nlp_project.movies.models import split_fn

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
        from rital_nlp_project.presidents.models import cv_fn
    if split:
        from rital_nlp_project.presidents.models import split_fn

    return eval_pipeline(
        pipeline, X, y,
        split_fn=split_fn,
        cv_fn=cv_fn,
        scoring=scoring,
        clf_report=clf_report,
    )
