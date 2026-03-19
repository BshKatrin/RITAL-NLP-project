from __future__ import annotations

import unittest

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from rital_nlp_project.common.models.word2vec import Word2VecPoolingTransformer


class Word2VecPoolingTransformerTests(unittest.TestCase):
    def setUp(self):
        self.model = {
            "good": np.array([1.0, 0.0, 0.0]),
            "movie": np.array([0.0, 1.0, 0.0]),
            "bad": np.array([-1.0, 0.0, 0.0]),
        }
        self.texts = ["good movie", "", "unknown token", "bad movie"]

    def test_mean_pooling_handles_empty_documents(self):
        transformer = Word2VecPoolingTransformer(self.model, pooling="mean")
        transformer.fit(self.texts)
        vectors = transformer.transform(self.texts)
        self.assertEqual(vectors.shape, (4, 3))
        np.testing.assert_array_equal(vectors[1], np.zeros(3))
        np.testing.assert_array_equal(vectors[2], np.zeros(3))

    def test_tfidf_pooling_runs_end_to_end(self):
        transformer = Word2VecPoolingTransformer(
            self.model,
            pooling="tfidf",
            vectorizer=TfidfVectorizer(),
        )
        transformer.fit(self.texts)
        vectors = transformer.transform(self.texts)
        self.assertEqual(vectors.shape, (4, 3))
        self.assertFalse(np.isnan(vectors).any())

    def test_sif_pooling_runs_end_to_end(self):
        transformer = Word2VecPoolingTransformer(
            self.model,
            pooling="sif",
            vectorizer=CountVectorizer(),
        )
        transformer.fit(self.texts)
        vectors = transformer.transform(self.texts)
        self.assertEqual(vectors.shape, (4, 3))
        self.assertFalse(np.isnan(vectors).any())


if __name__ == "__main__":
    unittest.main()
