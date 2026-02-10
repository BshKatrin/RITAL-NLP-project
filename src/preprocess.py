from .lexicon import URL_PATTERN, MAIL_PATTERN, PUNCTUATION
import re
from unidecode import unidecode
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import nltk


def remove_emails(text: str) -> str:
    return re.sub(MAIL_PATTERN, ' ', text)


def remove_urls(text: str) -> str:
    return re.sub(URL_PATTERN, ' ', text)


def remove_digits(text: str) -> str:
    return re.sub(r'[0-9]+', '', text)


def remove_punctuation(text: str) -> str:
    return text.lower().translate(str.maketrans('', '', PUNCTUATION))


def remove_stopwords(text: str, stopwords) -> str:  # should be called after removing
    return " ".join([word for word in text.split() if word not in stopwords])


def normalize_text(text: str) -> str:
    return unidecode(text)


def penn_to_wordnet(tag: str) -> str:
    """
    Map Penn Treebank POS tags to WordNet part of speech tags.
    Source : https://medium.com/techmind-chronicles/nlp-series-part-3-lemmatization-with-nltk-smarter-text-normalization-with-pos-tags-3f2d9ea212ea
    """
    if tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ  # 'a'
    if tag.startswith('V'):
        return nltk.corpus.wordnet.VERB  # 'v'
    if tag.startswith('N'):
        return nltk.corpus.wordnet.NOUN  # 'n'
    if tag.startswith('R'):
        return nltk.corpus.wordnet.ADV  # 'r'
    return nltk.corpus.wordnet.NOUN     # sensible fallback
