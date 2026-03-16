from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD

import numpy as np


class Word2VecPoolingTransformer(BaseEstimator, TransformerMixin):
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
        return f"Word2VecPoolingTransformer(pooling={self.pooling})"

    def mean_pooling(self, model, texts):
        return np.array([np.mean([model[word] for word in text.split() if word in model], axis=0)
                         for text in texts])

    def max_pooling(self, model, texts):
        return np.array([np.max([model[word] for word in text.split() if word in model], axis=0)
                        for text in texts])

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
                for word in text.split()
                if word in model and word in vectorizer.vocabulary_
            ]
            if not valid_words:
                embeddings.append(np.zeros(model.vector_size))
                continue

            words, weights = zip(*valid_words)
            words_vecs = np.array([model[w] for w in words])
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
                for word in text.split()
                if word in model and word in vectorizer.vocabulary_
            ]

            if not valid_words:
                embeddings.append(np.zeros(model.vector_size))
                continue

            words, weights = zip(*valid_words)
            words_vecs = np.array([model[w] for w in words])
            weights = np.array(weights).reshape(-1, 1)
            weighted_avg = np.sum(words_vecs * weights, axis=0) / np.sum(weights)
            embeddings.append(weighted_avg)

        embeddings = np.array(embeddings)

        # recommended
        svd = TruncatedSVD(n_components=1, n_iter=7, random_state=42)
        u = svd.fit(embeddings).components_[0]
        return embeddings - embeddings.dot(u.reshape(-1, 1)) * u
