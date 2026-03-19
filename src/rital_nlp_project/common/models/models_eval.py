from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

from rital_nlp_project.common.metrics import (
    DEFAULT_SCORING_NAMES,
    BinaryTaskSpec,
    classification_report_for_task,
    compute_binary_metrics,
    get_positive_class_scores,
    infer_binary_task_spec,
    make_binary_cv_scorers,
)


def eval_combination_matrix(
    X,
    y,
    vectorizer_list,
    classifier_list,
    scoring=None,
    *,
    cv=None,
    task: BinaryTaskSpec | None = None,
):
    task = task or infer_binary_task_spec("dataset", y)
    scorers = scoring or make_binary_cv_scorers(task)
    if cv is None:
        from sklearn.model_selection import StratifiedKFold

        cv = StratifiedKFold(
            n_splits=task.cv_n_splits,
            shuffle=not task.sequential,
            random_state=None if task.sequential else task.random_state,
        )

    results = []
    for vect in vectorizer_list:
        for clf in classifier_list:
            pipe = Pipeline([("vectorizer", vect), ("classifier", clf)])
            scores = cross_validate(pipe, X, y, scoring=scorers, cv=cv)
            result = {
                "vectorizer": vect,
                "classifier": clf,
                "train_time": float(np.mean(scores["fit_time"])),
                "score_time": float(np.mean(scores["score_time"])),
            }
            for score_name in scorers:
                result[score_name] = float(np.mean(scores["test_" + score_name]))
            results.append(result)
    return pd.DataFrame(results)


def _evaluate_split(
    pipeline,
    X_train,
    X_test,
    y_train,
    y_test,
    *,
    task: BinaryTaskSpec,
    clf_report: bool = False,
):
    results = {}

    t0 = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"Training time: {train_time:.3f} seconds")
    results["train_time"] = train_time

    t0 = time.time()
    predictions = pipeline.predict(X_test)
    infer_time = time.time() - t0
    print(f"Inference time: {infer_time:.3f} seconds")
    results["infer_time"] = infer_time

    y_score = get_positive_class_scores(pipeline, X_test, task.positive_label)
    metrics = compute_binary_metrics(y_test, predictions, task=task, y_score=y_score)
    for score_name, score_value in metrics.items():
        results[score_name] = score_value
        print(f"Test {score_name}: {score_value:.4f}")

    if clf_report:
        print(classification_report_for_task(y_test, predictions, task))
    print()
    return results


def _eval_pipeline_modern(
    pipeline,
    X,
    y,
    *,
    split_fn=None,
    cv_fn=None,
    scoring=None,
    clf_report: bool = False,
    task: BinaryTaskSpec | None = None,
):
    task = task or infer_binary_task_spec("dataset", y)
    scorers = scoring or make_binary_cv_scorers(task)
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
        scores = cross_validate(pipeline, X, y, scoring=scorers, cv=cv)
        results["cv_train_time"] = float(np.mean(scores["fit_time"]))
        results["cv_score_time"] = float(np.mean(scores["score_time"]))
        for score_name in scorers:
            mean_value = float(np.mean(scores["test_" + score_name]))
            results["cv_" + score_name] = mean_value
            print(f"Cross-validation {score_name} mean: {mean_value:.4f}")

    if split_fn is not None:
        X_train, X_test, y_train, y_test = split_fn(X, y)
        results.update(
            _evaluate_split(
                pipeline,
                X_train,
                X_test,
                y_train,
                y_test,
                task=task,
                clf_report=clf_report,
            )
        )

    return results


def _default_scoring_for_legacy(y) -> dict[str, Any]:
    task = infer_binary_task_spec("legacy", y)
    return make_binary_cv_scorers(task)


def _eval_pipeline_legacy(*args, **kwargs):
    if len(args) < 7:
        raise TypeError("legacy eval_pipeline expects at least 7 positional arguments")

    X, y, X_train, y_train, X_test, y_test, pipeline = args[:7]
    remaining = list(args[7:])
    scoring = None
    task = kwargs.pop("task", None) or infer_binary_task_spec("legacy", y)
    clf_report = kwargs.pop("clf_report", False)

    if remaining and isinstance(remaining[0], dict):
        scoring = remaining.pop(0)
    if remaining and isinstance(remaining[0], bool):
        remaining.pop(0)  # old cv flag, ignored because train/test are explicit
    if remaining and isinstance(remaining[0], bool):
        clf_report = remaining.pop(0)

    scoring = scoring or _default_scoring_for_legacy(y)

    results = {}
    print("Pipeline steps:" if hasattr(pipeline, "named_steps") else "Model:")
    if hasattr(pipeline, "named_steps"):
        for name, step in pipeline.named_steps.items():
            print(f"  {name}: {step}")
            results[name] = step
    else:
        print(f"  model: {pipeline}")
        results["model"] = pipeline

    split_results = _evaluate_split(
        pipeline,
        X_train,
        X_test,
        y_train,
        y_test,
        task=task,
        clf_report=clf_report,
    )
    for key in DEFAULT_SCORING_NAMES:
        if key in split_results:
            results[key] = split_results[key]
    results.update(split_results)
    return results


def eval_pipeline(*args, **kwargs):
    if args and len(args) >= 7 and not hasattr(args[0], "fit"):
        return _eval_pipeline_legacy(*args, **kwargs)
    return _eval_pipeline_modern(*args, **kwargs)


def eval_pipeline_movies(
    pipeline,
    X,
    y,
    *,
    cv: bool = False,
    split: bool = False,
    clf_report: bool = False,
):
    from rital_nlp_project.movies.models import cv_fn, split_fn
    from rital_nlp_project.movies.models_config import task

    return _eval_pipeline_modern(
        pipeline,
        X,
        y,
        split_fn=split_fn if split else None,
        cv_fn=cv_fn if cv else None,
        scoring=make_binary_cv_scorers(task),
        clf_report=clf_report,
        task=task,
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
    from rital_nlp_project.presidents.models import cv_fn, split_fn
    from rital_nlp_project.presidents.models_config import task

    return _eval_pipeline_modern(
        pipeline,
        X,
        y,
        split_fn=split_fn if split else None,
        cv_fn=cv_fn if cv else None,
        scoring=make_binary_cv_scorers(task),
        clf_report=clf_report,
        task=task,
    )
