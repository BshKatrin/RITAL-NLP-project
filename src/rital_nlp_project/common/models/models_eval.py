import time

from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

import numpy as np
import pandas as pd

scoring = {
    'accuracy': 'accuracy',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}


def eval_combination_matrix(X, y, vectorizer_list, classifier_list):
    global scoring

    results = []
    for vect in vectorizer_list:
        for clf in classifier_list:
            pipe = Pipeline([("vectorizer", vect), ("classifier", clf)])

            # Cross-val eval
            scores = cross_validate(pipe, X, y, scoring=scoring)

            print(f"Cross-validation F1 scores: {scores["test_f1"]}, mean: {np.mean(scores["test_f1"]):.4f}")
            print(
                f"Cross-validation Accuracy scores: {scores["test_accuracy"]}, mean: {np.mean(scores["test_accuracy"]): .4f}")
            print(f"Cross-validation AUC scores: {scores["test_roc_auc"]}, mean: {np.mean(scores["test_roc_auc"]):.4f}")

            results.append({
                "vectorizer": vect,
                "classifier": clf,
                "accuracy": np.mean(scores["test_accuracy"]),
                "f1": np.mean(scores["test_f1"]),
                "roc_auc": np.mean(scores["test_roc_auc"]),
            })
    return pd.DataFrame(results)


def eval_pipeline(X, y, X_train, y_train, X_test, y_test, pipeline, cross_val: bool = False, clf_report: bool = False):
    # Pipeline desc
    global scoring

    print("Pipeline steps:")
    for name, step in pipeline.named_steps.items():
        print(f"  {name}: {step}")

    if cross_val:
        scores = cross_validate(pipeline, X, y, scoring=scoring)

        print(f"Cross-validation F1 scores: {scores["test_f1"]}, mean: {np.mean(scores["test_f1"]):.4f}")
        print(
            f"Cross-validation Accuracy scores: {scores["test_accuracy"]}, mean: {np.mean(scores["test_accuracy"]): .4f}")
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

    # print("Vocabulary size:", len(pipeline.named_steps["vectorizer"].vocabulary_))
    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    auc = roc_auc_score(y_test, predictions)

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test F1-score: {f1:.4f}")
    print(f"Test ROC-AUC: {auc:.4f}")

    if clf_report:
        print(classification_report(y_test, predictions))
    print()
