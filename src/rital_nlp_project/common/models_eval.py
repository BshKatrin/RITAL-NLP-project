import time

from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight


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


def eval_pipeline(X, y, X_train, y_train, X_test, y_test, pipeline, clf_report: bool = False):
    # Pipeline desc
    print("Pipeline steps:")
    for name, step in pipeline.named_steps.items():
        print(f"  {name}: {step}")

    scoring = {
        'accuracy': 'accuracy',                  # built-in string
        'f1': "f1",
        'roc_auc': 'roc_auc'                     # built-in string
    }

    # Cross validation
    scores = cross_validate(pipeline, X, y, scoring=scoring)

    print(f"Cross-validation F1 scores: {scores["test_f1"]}, mean: {np.mean(scores["test_f1"]):.4f}")
    print(f"Cross-validation Accuracy scores: {scores["test_accuracy"]}, mean: {np.mean(scores["test_accuracy"]): .4f}")
    print(f"Cross-validation AUC scores: {scores["test_roc_auc"]}, mean: {np.mean(scores["test_roc_auc"]):.4f}")

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
    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    auc = roc_auc_score(y_test, predictions)

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test F1-score: {f1:.4f}")
    print(f"Test ROC-AUC: {auc:.4f}")

    if clf_report:
        print(classification_report(y_test, predictions))
    print()
