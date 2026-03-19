from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from rital_nlp_project.cli import run_benchmark, run_predict, run_prepare, run_train_word2vec
from rital_nlp_project.common.models.models_eval import eval_pipeline
from rital_nlp_project.common.preprocess.stopwords import STOPWORDS
from rital_nlp_project.common.utils import load_movies
from rital_nlp_project.presidents.utils import SmoothedProbaClassifier, split_for_cv


def _toy_movie_dataset(n_per_class: int = 10) -> pd.DataFrame:
    negatives = [f"bad boring dull movie {i}" for i in range(n_per_class)]
    positives = [f"great fun enjoyable movie {i}" for i in range(n_per_class)]
    texts = negatives + positives
    labels = [0] * n_per_class + [1] * n_per_class
    return pd.DataFrame({"text": texts, "label": labels})


class CorePipelineTests(unittest.TestCase):
    def test_load_movies_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "neg").mkdir()
            (root / "pos").mkdir()
            (root / "neg" / "b.txt").write_text("bad second")
            (root / "neg" / "a.txt").write_text("bad first")
            (root / "pos" / "c.txt").write_text("good first")
            texts, labels = load_movies(str(root))
            self.assertEqual(texts, ["bad first", "bad second", "good first"])
            self.assertEqual(labels, [0, 0, 1])

    def test_french_stopwords_include_unaccented_forms(self):
        self.assertIn("ete", STOPWORDS["french"])
        self.assertIn("meme", STOPWORDS["french"])
        self.assertIn("a", STOPWORDS["french"])

    def test_blocked_cv_is_contiguous(self):
        y = np.array([-1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1])
        cv = split_for_cv(y, 4)
        for _, test_idx in cv.split():
            if len(test_idx) > 1:
                self.assertTrue(np.all(np.diff(test_idx) == 1))

    def test_smoothed_classifier_uses_smoothed_probabilities_for_scores(self):
        X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
        y = np.array([0, 0, 0, 1, 1, 1])
        clf = SmoothedProbaClassifier(LogisticRegression(max_iter=1000), sigma=1.0)
        clf.fit(X, y)
        scores = clf.decision_function(X)
        probas = clf.predict_proba(X)[:, 1]
        np.testing.assert_allclose(scores, probas)

    def test_legacy_eval_pipeline_signature_still_works(self):
        texts = np.array(
            [
                "bad film",
                "bad and dull",
                "awful movie",
                "terrible film",
                "great film",
                "great and fun",
                "excellent movie",
                "wonderful film",
            ]
        )
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        pipeline = Pipeline(
            [
                ("vectorizer", CountVectorizer()),
                ("classifier", LogisticRegression(max_iter=1000)),
            ]
        )
        results = eval_pipeline(
            texts,
            y,
            texts[[0, 1, 3, 4, 5, 7]],
            y[[0, 1, 3, 4, 5, 7]],
            texts[[2, 6]],
            y[[2, 6]],
            pipeline,
            False,
            True,
        )
        self.assertIn("accuracy", results)
        self.assertIn("f1_macro", results)

    def test_cli_benchmark_and_predict_text_experiment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_path = root / "movies.csv"
            output_csv = root / "benchmark.csv"
            predict_input = root / "predict.csv"
            predict_output = root / "predictions.csv"

            train_df = _toy_movie_dataset(10)
            train_df.to_csv(train_path, index=False)
            pd.DataFrame({"text": ["great fun sequel", "bad dull sequel"]}).to_csv(
                predict_input,
                index=False,
            )

            benchmark_args = Namespace(
                task="movies",
                data_path=str(train_path),
                embeddings_path=None,
                word2vec_path=None,
                experiment="movies_count_word12_nb",
                feature_kind=None,
                output_csv=str(output_csv),
                output_json=None,
                clf_report=False,
            )
            self.assertEqual(run_benchmark(benchmark_args), 0)
            self.assertTrue(output_csv.exists())

            predict_args = Namespace(
                task="movies",
                experiment="movies_count_word12_nb",
                train_data_path=str(train_path),
                predict_data_path=str(predict_input),
                train_embeddings_path=None,
                predict_embeddings_path=None,
                word2vec_path=None,
                output_path=str(predict_output),
                output_kind="labels",
                smooth_group_col=None,
                smooth_sigma=0.0,
            )
            self.assertEqual(run_predict(predict_args), 0)
            preds = pd.read_csv(predict_output)
            self.assertEqual(len(preds), 2)
            self.assertIn("prediction", preds.columns)

    def test_cli_benchmark_and_predict_embedding_experiment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_path = root / "movies.csv"
            embeddings_path = root / "train.npy"
            predict_embeddings_path = root / "predict.npy"
            output_csv = root / "benchmark_embeddings.csv"
            predict_output = root / "predictions_embeddings.csv"

            train_df = _toy_movie_dataset(10)
            train_df.to_csv(train_path, index=False)
            train_embeddings = np.vstack(
                [
                    np.tile(np.array([-1.0, 0.0, 0.0]), (10, 1)),
                    np.tile(np.array([1.0, 0.0, 0.0]), (10, 1)),
                ]
            )
            predict_embeddings = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
            np.save(embeddings_path, train_embeddings)
            np.save(predict_embeddings_path, predict_embeddings)

            benchmark_args = Namespace(
                task="movies",
                data_path=str(train_path),
                embeddings_path=str(embeddings_path),
                word2vec_path=None,
                experiment="movies_embeddings_logreg",
                feature_kind=None,
                output_csv=str(output_csv),
                output_json=None,
                clf_report=False,
            )
            self.assertEqual(run_benchmark(benchmark_args), 0)
            self.assertTrue(output_csv.exists())

            predict_args = Namespace(
                task="movies",
                experiment="movies_embeddings_logreg",
                train_data_path=str(train_path),
                predict_data_path=None,
                train_embeddings_path=str(embeddings_path),
                predict_embeddings_path=str(predict_embeddings_path),
                word2vec_path=None,
                output_path=str(predict_output),
                output_kind="labels",
                smooth_group_col=None,
                smooth_sigma=0.0,
            )
            self.assertEqual(run_predict(predict_args), 0)
            preds = pd.read_csv(predict_output)
            self.assertEqual(len(preds), 2)
            self.assertIn("prediction", preds.columns)

    def test_prepare_test_split_preserves_hidden_test_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_path = root / "corpus.tache1.test.utf8"
            output_path = root / "presidents_test_clean"
            raw_path.write_text("<10:1> Bonjour la nation.\n<10:2> Vive la France.\n", encoding="utf-8")

            args = Namespace(
                task="presidents",
                split="test",
                input_path=str(raw_path),
                output_path=str(output_path),
                pipeline_mode="classic",
                stem=False,
                no_lemmatize=True,
            )
            self.assertEqual(run_prepare(args), 0)

            prepared = pd.read_parquet(output_path.with_suffix(".parquet"))
            self.assertEqual(list(prepared.columns), ["speech_id", "sentence_id", "text"])
            self.assertEqual(prepared["speech_id"].tolist(), [10, 10])
            self.assertEqual(prepared["sentence_id"].tolist(), [1, 2])

    def test_predict_presidents_supports_score_and_full_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_path = root / "presidents.csv"
            predict_path = root / "presidents_test.csv"
            score_output = root / "presidents_scores.txt"
            full_output = root / "presidents_predictions.csv"

            train_df = pd.DataFrame(
                {
                    "text": (
                        [f"nation solidarite emploi reforme {i}" for i in range(10)]
                        + [f"monsieur president france culture {i}" for i in range(10)]
                    ),
                    "label": ([-1] * 10) + ([1] * 10),
                }
            )
            predict_df = pd.DataFrame(
                {
                    "speech_id": [101, 101],
                    "sentence_id": [1, 2],
                    "text": ["nation travail", "monsieur president"],
                }
            )
            train_df.to_csv(train_path, index=False)
            predict_df.to_csv(predict_path, index=False)

            score_args = Namespace(
                task="presidents",
                experiment="presidents_count_word13_nb",
                train_data_path=str(train_path),
                predict_data_path=str(predict_path),
                train_embeddings_path=None,
                predict_embeddings_path=None,
                word2vec_path=None,
                output_path=str(score_output),
                output_kind="scores",
                smooth_group_col=None,
                smooth_sigma=0.0,
            )
            self.assertEqual(run_predict(score_args), 0)
            scores = score_output.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(scores), 2)

            full_args = Namespace(
                task="presidents",
                experiment="presidents_count_word13_nb",
                train_data_path=str(train_path),
                predict_data_path=str(predict_path),
                train_embeddings_path=None,
                predict_embeddings_path=None,
                word2vec_path=None,
                output_path=str(full_output),
                output_kind="full",
                smooth_group_col=None,
                smooth_sigma=0.0,
            )
            self.assertEqual(run_predict(full_args), 0)
            preds = pd.read_csv(full_output)
            self.assertEqual(
                list(preds.columns),
                ["speech_id", "sentence_id", "prediction_int", "prediction_label", "score"],
            )
            self.assertEqual(len(preds), 2)

    def test_cli_train_word2vec_command(self):
        try:
            import gensim  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("gensim not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_path = root / "movies.csv"
            model_path = root / "movies_w2v.model"
            _toy_movie_dataset(10).to_csv(train_path, index=False)

            args = Namespace(
                data_path=str(train_path),
                output_path=str(model_path),
                vector_size=20,
                window=2,
                min_count=1,
                sg=1,
                epochs=5,
                workers=1,
            )
            self.assertEqual(run_train_word2vec(args), 0)
            self.assertTrue(model_path.exists())


if __name__ == "__main__":
    unittest.main()
