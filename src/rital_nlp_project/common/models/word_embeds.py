import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD


class WordEmbeddingsPoolingTransformer(BaseEstimator, TransformerMixin):
    # Static word embeddings (w2v, fasttext)
    def __init__(self, model, pooling='mean', vectorizer=None):
        self.model = model
        self.pooling = pooling
        self.vectorizer = vectorizer

    def fit(self, X, y=None):
        if self.vectorizer:
            self.vectorizer.fit(X)
        return self

    def transform(self, X):
        if self.pooling == 'mean':
            return self.mean_pooling(self.model, X)
        elif self.pooling == 'max':
            return self.max_pooling(self.model, X)
        elif self.pooling == 'mean_max':
            return self.mean_max_pooling(self.model, X)
        elif self.pooling == 'tfidf':
            if self.vectorizer is None:
                raise ValueError('vectorizer and matrix must be set for tfidf pooling')
            matrix = self.vectorizer.transform(X)
            return self.tfidf_pooling(self.model, X, self.vectorizer, matrix)
        elif self.pooling == 'sif':
            if self.vectorizer is None:
                raise ValueError('vectorizer and matrix must be set for sif pooling')
            matrix = self.vectorizer.transform(X)
            return self.sif_pooling(self.model, X, self.vectorizer, matrix)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

    def __repr__(self):
        model_name = type(self.model).__name__.lower()

        if 'fasttext' in model_name:
            model_kind = 'fasttext'
        elif 'word2vec' in model_name or 'keyedvectors' in model_name:
            model_kind = 'word2vec'
        else:
            model_kind = 'unknown'

        return f"WordEmbeddingsPoolingTransformer(pooling={self.pooling}, model={model_kind})"

    def _get_keyed_vectors(self, model):
        return model.wv if hasattr(model, 'wv') else model

    def _get_vector_size(self, model):
        keyed_vectors = self._get_keyed_vectors(model)
        return keyed_vectors.vector_size

    def mean_pooling(self, model, texts):
        keyed_vectors = self._get_keyed_vectors(model)
        vector_size = self._get_vector_size(model)

        return np.array([
            np.mean([keyed_vectors[word] for word in text.split() if word in keyed_vectors], axis=0)
            if any(word in keyed_vectors for word in text.split())
            else np.zeros(vector_size)
            for text in texts
        ])

    def max_pooling(self, model, texts):
        keyed_vectors = self._get_keyed_vectors(model)
        vector_size = self._get_vector_size(model)

        return np.array([
            np.max([keyed_vectors[word] for word in text.split() if word in keyed_vectors], axis=0)
            if any(word in keyed_vectors for word in text.split())
            else np.zeros(vector_size)
            for text in texts
        ])

    def mean_max_pooling(self, model, texts):
        # Mean, max concatenation
        mean_pool = self.mean_pooling(model, texts)
        max_pool = self.max_pooling(model, texts)
        return np.concatenate([mean_pool, max_pool], axis=1)

    def tfidf_pooling(self, model, texts, vectorizer, M):
        keyed_vectors = self._get_keyed_vectors(model)
        vector_size = self._get_vector_size(model)
        embeddings = []

        for doc_id, text in enumerate(texts):
            tfidf_vec = M[doc_id]
            valid_words = [
                (word, tfidf_vec[0, vectorizer.vocabulary_[word]])
                for word in text.split()
                if word in keyed_vectors and word in vectorizer.vocabulary_
            ]
            if not valid_words:
                embeddings.append(np.zeros(vector_size))
                continue

            words, weights = zip(*valid_words)
            words_vecs = np.array([keyed_vectors[w] for w in words])
            weights = np.array(weights).reshape(-1, 1)
            weighted_avg = np.sum(words_vecs * weights, axis=0) / np.sum(weights)
            embeddings.append(weighted_avg)

        return np.array(embeddings)

    def sif_pooling(self, model, texts, vectorizer, M, a=1e-3):
        keyed_vectors = self._get_keyed_vectors(model)
        vector_size = self._get_vector_size(model)
        probas = np.array(M.sum(axis=0), dtype=float).flatten()
        probas /= np.sum(probas)
        embeddings = []

        for text in texts:
            valid_words = [
                (word, a / (a + probas[vectorizer.vocabulary_[word]]))
                for word in text.split()
                if word in keyed_vectors and word in vectorizer.vocabulary_
            ]

            if not valid_words:
                embeddings.append(np.zeros(vector_size))
                continue

            words, weights = zip(*valid_words)
            words_vecs = np.array([keyed_vectors[w] for w in words])
            weights = np.array(weights).reshape(-1, 1)
            weighted_avg = np.sum(words_vecs * weights, axis=0) / np.sum(weights)
            embeddings.append(weighted_avg)

        embeddings = np.array(embeddings)

        # recommended
        svd = TruncatedSVD(n_components=1, n_iter=7, random_state=42)
        u = svd.fit(embeddings).components_[0]
        return embeddings - embeddings.dot(u.reshape(-1, 1)) * u
