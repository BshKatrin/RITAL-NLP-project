import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np


def plot_class_wordclouds(texts_cleaned, classes, vectorizer):
    fig, axes = plt.subplots(1, 2, figsize=(8, 5))
    axes = axes.flatten()

    cloud = WordCloud(background_color='white', stopwords=[], max_words=100)
    for i, target in enumerate(np.unique(classes)):
        # Extract data related to target
        idx = np.where(classes == target)[0]
        data_target = [texts_cleaned[i] for i in idx]

        # Extract frequencies
        vectors = vectorizer.fit_transform(data_target)
        freqs = {word: int(freq) for word, freq in zip(vectorizer.get_feature_names_out(), vectors.sum(axis=0).A1)}

        # Generate wordcloud
        cloud = cloud.generate_from_frequencies(freqs)

        axes[i].imshow(cloud)
        axes[i].axis("off")
        axes[i].set_title(target)

    plt.tight_layout()
