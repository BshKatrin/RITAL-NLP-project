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


def eval_pipeline(X, y, X_train, y_train, X_test, y_test, pipeline, scoring, cross_val: bool = False, clf_report: bool = False):
    print("Pipeline steps:")
    results = dict()

    for name, step in pipeline.named_steps.items():
        print(f"  {name}: {step}")
        results[name] = step

    if cross_val:
        scores = cross_validate(pipeline, X, y, scoring=scoring)
        for score_name in scoring:
            print(
                f"Cross-validation {score_name} scores: {scores['test_' + score_name]}, mean: {np.mean(scores['test_' + score_name]):.4f}")

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
        score_value = scorer._score_func(y_test, predictions)
        results[score_name] = score_value
        print(f"Test {score_name}: {score_value:.4f}")

    if clf_report:
        print(classification_report(y_test, predictions))
    print()

    return results
