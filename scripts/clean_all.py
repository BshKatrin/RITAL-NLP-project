
from rital_nlp_project.common.utils import load_pres, load_movies, clean_dataset
from rital_nlp_project.common.stopwords import STOPWORDS

from rital_nlp_project.movies.transformer import TextPreprocessor as TextPreprocessorMovies
from rital_nlp_project.presidents.transformer import TextPreprocessor as TextPreprocessorPresidents


if __name__ == "__main__":
    print("Started cleaning movies")
    preprocessor = TextPreprocessorMovies(stem=False, lemmatize=True, stopwords=STOPWORDS["english"])
    clean_dataset(load_movies, "Dataset/movies1000/", "Dataset/movies_clean", preprocessor)

    print("Started cleaning presidents")
    preprocessor = TextPreprocessorPresidents(stem=False, lemmatize=True, stopwords=STOPWORDS["french"])
    clean_dataset(load_pres, "Dataset/presidents.utf8", "Dataset/presidents_clean", preprocessor)
