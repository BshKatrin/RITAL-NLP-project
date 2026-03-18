
import argparse
import time

from rital_nlp_project.common.utils import load_pres, load_movies, clean_dataset
from rital_nlp_project.common.preprocess.stopwords import STOPWORDS

from rital_nlp_project.movies.transformer import TextPreprocessor as TextPreprocessorMovies
from rital_nlp_project.presidents.transformer import TextPreprocessor as TextPreprocessorPresidents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean movies and presidents datasets.")
    parser.add_argument(
        "--dataset",
        choices=["movies", "presidents"],
        default=None,
        help="Dataset to clean."
    )

    parser.add_argument(
        "--input-path",
        default=None,
        help="Path to the original dataset.",
    )

    parser.add_argument(
        "--pipeline-mode",
        choices=["classic", "bert"],
        default="classic",
        help="Preprocessing mode to use: 'classic' (BoW/TF-IDF) or 'bert'. Default: classic.",
    )

    # parser.add_argument(
    #     "--special-tokens-mode",
    #     choices=["space", "tokens"],
    #     default=None,  # if None, derive it from pipeline_mode
    #     help="How to handle special tokens (emails/URLs/phones): "
    #          "'space' to remove them, 'tokens' to replace with <EMAIL>/<URL>/<PHONE>. "
    #          "Default: derived from --pipeline-mode (bert -> tokens, classic -> space).",
    # )

    parser.add_argument(
        "--output-path",
        default=None,
        help="Output path for cleaned dataset."
    )

    return parser.parse_args()


def clean_movies(args):
    print(f"Started cleaning movies with pipeline_mode={args.pipeline_mode}")

    preprocessor = TextPreprocessorMovies(
        stem=False,
        lemmatize=True,
        stopwords=STOPWORDS["english"],
        pipeline_mode=args.pipeline_mode,
    )

    t0 = time.time()
    clean_dataset(load_movies, args.input_path, args.output_path, preprocessor)
    t0 = time.time() - t0
    print(f"Finished cleaning movies. Took {t0} seconds.")


def clean_presidents(args):
    print(f"Started cleaning presidents with pipeline_mode={args.pipeline_mode}")

    preprocessor = TextPreprocessorPresidents(
        stem=False,
        lemmatize=True,
        stopwords=STOPWORDS["french"],
        pipeline_mode=args.pipeline_mode,
    )

    t0 = time.time()
    clean_dataset(load_pres, args.input_path, args.output_path, preprocessor)
    t0 = time.time() - t0
    print(f"Finished cleaning presidents. Took {t0} seconds.")


if __name__ == "__main__":
    # Parse args
    args = parse_args()

    dataset = args.dataset
    print(dataset)
    if dataset == "movies":
        clean_movies(args)

    if dataset == "presidents":
        clean_presidents(args)
