import time

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

import numpy as np
import pandas as pd


def eval_combination_matrix(X_train, y_train, X_test, y_test, vectorizer_list, classifier_list):
    results = []
    for vect in vectorizer_list:
        for clf in classifier_list:
            pipe = Pipeline([("vectorizer", vect), ("classifier", clf)])
            print(pipe)
            # Train-test eval
            t0 = time.time()
            pipe.fit(X_train, y_train)
            train_time = time.time() - t0

            t0 = time.time()
            preds = pipe.predict(X_test)
            inference_time = time.time() - t0

            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average="weighted")

            # Cross-val eval
            mean_acc = np.mean(cross_val_score(pipe, X_train, y_train, cv=5, scoring="f1"))

            results.append({
                "vectorizer": vect,
                "classifier": clf,
                "accuracy": acc,
                "f1": f1,
                "train_time": train_time,
                "inference_time": inference_time,
                "cross_val_accuracy": mean_acc,
            })
    return pd.DataFrame(results)


def eval_pipeline(X_train, y_train, X_test, y_test, pipeline, clf_report: bool = False):
    # Pipeline desc
    print("Pipeline steps:")
    for name, step in pipeline.named_steps.items():
        print(f"  {name}: {step}")

    # Cross validation
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="f1")
    print(f"Scores : {scores}, mean score : {np.mean(scores)}")

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

    print("Vocabulary size:", len(pipeline.named_steps["vectorizer"].vocabulary_))
    print("Train/test accuracy score:", f1_score(y_test, predictions))
    if clf_report:
        print(classification_report(y_test, predictions))
    print()
