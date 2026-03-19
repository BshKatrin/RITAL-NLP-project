import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD


class Word2VecPoolingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model, pooling='mean', vectorizer=None):
        self.model = model
        self.pooling = pooling
        self.vectorizer = vectorizer

    def fit(self, X, y=None):
        self.vector_size_ = self._get_vector_size()
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
        return f"Word2VecPoolingTransformer(pooling={self.pooling})"

    def _get_vector_size(self):
        if hasattr(self.model, "vector_size"):
            return int(self.model.vector_size)
        try:
            first_key = next(iter(self.model))
        except StopIteration as exc:  # pragma: no cover - defensive
            raise ValueError("word2vec model is empty") from exc
        return int(np.asarray(self.model[first_key]).shape[0])

    def _tokens(self, text):
        if isinstance(text, str):
            return text.split()
        return list(text)

    def _contains(self, word):
        try:
            return word in self.model
        except TypeError:  # pragma: no cover - defensive
            return False

    def _vector(self, word):
        return np.asarray(self.model[word], dtype=float)

    def _zero(self):
        return np.zeros(self.vector_size_, dtype=float)

    def _valid_vectors(self, text):
        tokens = self._tokens(text)
        return [self._vector(word) for word in tokens if self._contains(word)]

    def mean_pooling(self, model, texts):
        embeddings = []
        for text in texts:
            vectors = self._valid_vectors(text)
            if not vectors:
                embeddings.append(self._zero())
                continue
            embeddings.append(np.mean(vectors, axis=0))
        return np.vstack(embeddings)

    def max_pooling(self, model, texts):
        embeddings = []
        for text in texts:
            vectors = self._valid_vectors(text)
            if not vectors:
                embeddings.append(self._zero())
                continue
            embeddings.append(np.max(vectors, axis=0))
        return np.vstack(embeddings)

    def mean_max_pooling(self, model, texts):
        # Mean, max concatenation
        mean_pool = self.mean_pooling(model, texts)
        max_pool = self.max_pooling(model, texts)
        return np.concatenate([mean_pool, max_pool], axis=1)

    def tfidf_pooling(self, model, texts, vectorizer, M):
        embeddings = []

        for doc_id, text in enumerate(texts):
            tfidf_vec = M[doc_id]
            valid_words = [
                (word, tfidf_vec[0, vectorizer.vocabulary_[word]])
                for word in self._tokens(text)
                if self._contains(word) and word in vectorizer.vocabulary_
            ]
            if not valid_words:
                embeddings.append(self._zero())
                continue

            words, weights = zip(*valid_words)
            words_vecs = np.array([self._vector(w) for w in words])
            weights = np.array(weights).reshape(-1, 1)
            weighted_avg = np.sum(words_vecs * weights, axis=0) / np.sum(weights)
            embeddings.append(weighted_avg)

        return np.array(embeddings)

    def sif_pooling(self, model, texts, vectorizer, M, a=1e-3):
        probas = np.array(M.sum(axis=0), dtype=float).flatten()
        probas /= np.sum(probas)
        embeddings = []

        for text in texts:
            valid_words = [
                (word, a / (a + probas[vectorizer.vocabulary_[word]]))
                for word in self._tokens(text)
                if self._contains(word) and word in vectorizer.vocabulary_
            ]

            if not valid_words:
                embeddings.append(self._zero())
                continue

            words, weights = zip(*valid_words)
            words_vecs = np.array([self._vector(w) for w in words])
            weights = np.array(weights).reshape(-1, 1)
            weighted_avg = np.sum(words_vecs * weights, axis=0) / np.sum(weights)
            embeddings.append(weighted_avg)

        embeddings = np.array(embeddings)

        if embeddings.shape[0] < 2 or not np.any(embeddings):
            return embeddings

        # recommended
        svd = TruncatedSVD(n_components=1, n_iter=7, random_state=42)
        u = svd.fit(embeddings).components_[0]
        return embeddings - embeddings.dot(u.reshape(-1, 1)) * u
