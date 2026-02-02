import nltk

REQUIRED = [
    "punkt",
    "stopwords",
]

# Other packages that might be interesting
# "vader_lexicon" for sentiment analysis
# "averaged_perceptron_tagger" for POS tagging
# "wordnet"
# "omw-1.4"

if __name__ == "__main__":
    for pkg in REQUIRED:
        nltk.download(pkg)