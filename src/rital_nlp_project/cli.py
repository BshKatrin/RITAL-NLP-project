from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rital_nlp_project.common.metrics import get_positive_class_scores
from rital_nlp_project.common.models.embeddings import (
    TransformerEmbeddingConfig,
    TransformerEmbeddingExtractor,
)
from rital_nlp_project.common.models.models_eval import (
    eval_pipeline_movies,
    eval_pipeline_presidents,
)
from rital_nlp_project.common.utils import (
    clean_dataframe,
    load_clean_data,
    load_dataframe,
    load_movies_dataframe,
    load_pres_dataframe,
    save_dataframe,
    save_prediction_lines,
    save_predictions,
)
from rital_nlp_project.experiments import (
    get_experiment,
    get_experiments,
    load_word2vec_model,
    train_word2vec_model,
)
from rital_nlp_project.movies.models_config import task as movies_task
from rital_nlp_project.presidents.models_config import task as presidents_task


TASK_SPECS = {
    "movies": movies_task,
    "presidents": presidents_task,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RITAL NLP project CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Clean a raw dataset into parquet")
    prepare.add_argument("--task", choices=["movies", "presidents"], required=True)
    prepare.add_argument("--split", choices=["train", "test"], default="train")
    prepare.add_argument("--input-path", required=True)
    prepare.add_argument("--output-path", required=True)
    prepare.add_argument("--pipeline-mode", choices=["classic", "bert"], default="classic")
    prepare.add_argument("--stem", action="store_true")
    prepare.add_argument("--no-lemmatize", action="store_true")

    benchmark = subparsers.add_parser("benchmark", help="Evaluate one or more experiments")
    benchmark.add_argument("--task", choices=["movies", "presidents"], required=True)
    benchmark.add_argument("--data-path", required=True)
    benchmark.add_argument("--embeddings-path")
    benchmark.add_argument("--word2vec-path")
    benchmark.add_argument("--experiment")
    benchmark.add_argument("--feature-kind", choices=["text", "embeddings"])
    benchmark.add_argument("--output-csv")
    benchmark.add_argument("--output-json")
    benchmark.add_argument("--clf-report", action="store_true")

    predict = subparsers.add_parser("predict", help="Train on full data and predict labels")
    predict.add_argument("--task", choices=["movies", "presidents"], required=True)
    predict.add_argument("--experiment", required=True)
    predict.add_argument("--train-data-path", required=True)
    predict.add_argument("--predict-data-path")
    predict.add_argument("--train-embeddings-path")
    predict.add_argument("--predict-embeddings-path")
    predict.add_argument("--word2vec-path")
    predict.add_argument("--output-path", required=True)
    predict.add_argument("--output-kind", choices=["auto", "labels", "scores", "full"], default="auto")
    predict.add_argument("--smooth-group-col")
    predict.add_argument("--smooth-sigma", type=float, default=0.0)

    embed = subparsers.add_parser("embed", help="Extract transformer embeddings from a cleaned dataset")
    embed.add_argument("--data-path", required=True)
    embed.add_argument("--output-path", required=True)
    embed.add_argument("--model-name", required=True)
    embed.add_argument("--pooling", choices=["cls", "mean"], default="mean")
    embed.add_argument("--batch-size", type=int, default=16)
    embed.add_argument("--max-length", type=int, default=256)
    embed.add_argument("--device", default="auto")
    embed.add_argument("--normalize", action="store_true")
    embed.add_argument("--text-col", default="text")

    list_cmd = subparsers.add_parser("list-experiments", help="List available experiments")
    list_cmd.add_argument("--task", choices=["movies", "presidents"])
    list_cmd.add_argument("--word2vec-path")

    train_w2v = subparsers.add_parser("train-word2vec", help="Train a Word2Vec model from cleaned text")
    train_w2v.add_argument("--data-path", required=True)
    train_w2v.add_argument("--output-path", required=True)
    train_w2v.add_argument("--vector-size", type=int, default=100)
    train_w2v.add_argument("--window", type=int, default=5)
    train_w2v.add_argument("--min-count", type=int, default=2)
    train_w2v.add_argument("--sg", type=int, default=1)
    train_w2v.add_argument("--epochs", type=int, default=10)
    train_w2v.add_argument("--workers", type=int, default=1)

    return parser.parse_args()


def _get_preprocessor(args: argparse.Namespace):
    from rital_nlp_project.common.preprocess.stopwords import STOPWORDS
    from rital_nlp_project.movies.transformer import TextPreprocessor as MoviesPreprocessor
    from rital_nlp_project.presidents.transformer import TextPreprocessor as PresidentsPreprocessor

    lemmatize = not args.no_lemmatize
    if args.task == "movies":
        return MoviesPreprocessor(
            stem=args.stem,
            lemmatize=lemmatize,
            stopwords=STOPWORDS["english"],
            pipeline_mode=args.pipeline_mode,
        )
    return PresidentsPreprocessor(
        stem=args.stem,
        lemmatize=lemmatize,
        stopwords=STOPWORDS["french"],
        pipeline_mode=args.pipeline_mode,
    )


def _load_raw_dataframe(task: str, input_path: str, split: str) -> pd.DataFrame:
    labeled = split == "train"
    if task == "movies":
        return load_movies_dataframe(input_path, labeled=labeled)
    return load_pres_dataframe(input_path, labeled=labeled)


def run_prepare(args: argparse.Namespace) -> int:
    preprocessor = _get_preprocessor(args)
    df = _load_raw_dataframe(args.task, args.input_path, args.split)
    clean_dataframe(df, args.output_path, preprocessor)
    return 0


def _select_experiments(args: argparse.Namespace):
    word2vec_model = load_word2vec_model(args.word2vec_path) if getattr(args, "word2vec_path", None) else None
    if getattr(args, "experiment", None):
        return [get_experiment(args.experiment, word2vec_model=word2vec_model)], word2vec_model

    experiments = get_experiments(args.task, word2vec_model=word2vec_model)
    filtered = []
    for experiment in experiments:
        if getattr(args, "feature_kind", None) and experiment.feature_kind != args.feature_kind:
            continue
        if experiment.feature_kind == "embeddings" and not getattr(args, "embeddings_path", None):
            continue
        if "word2vec" in experiment.name and word2vec_model is None:
            continue
        filtered.append(experiment)
    return filtered, word2vec_model


def _load_features(feature_kind: str, *, data_path: str, embeddings_path: str | None):
    texts, labels = load_clean_data(data_path)
    if feature_kind == "text":
        return texts, labels
    if not embeddings_path:
        raise ValueError("This experiment requires --embeddings-path")
    return np.load(embeddings_path), labels


def _evaluate_experiment(task: str, experiment, X, y, *, clf_report: bool):
    model = experiment.build()
    if task == "movies":
        return eval_pipeline_movies(model, X, y, cv=True, split=True, clf_report=clf_report)
    return eval_pipeline_presidents(model, X, y, cv=True, split=True, clf_report=clf_report)


def _serialize_value(value):
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def run_benchmark(args: argparse.Namespace) -> int:
    experiments, _ = _select_experiments(args)
    if not experiments:
        raise ValueError("No experiments matched the provided inputs.")

    rows = []
    for experiment in experiments:
        print(f"=== {experiment.name} ===")
        X, y = _load_features(
            experiment.feature_kind,
            data_path=args.data_path,
            embeddings_path=args.embeddings_path,
        )
        results = _evaluate_experiment(args.task, experiment, X, y, clf_report=args.clf_report)
        row = {
            "experiment": experiment.name,
            "task": experiment.task,
            "feature_kind": experiment.feature_kind,
            "description": experiment.description,
        }
        row.update({key: _serialize_value(value) for key, value in results.items()})
        rows.append(row)

    df = pd.DataFrame(rows)
    if args.output_csv:
        save_dataframe(args.output_csv, df)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(df.to_dict(orient="records"), indent=2))

    print(df.to_string(index=False))
    return 0


def _load_prediction_dataframe(path: str | None) -> pd.DataFrame | None:
    if not path:
        return None
    input_path = Path(path)
    if input_path.suffix == ".txt":
        return pd.DataFrame({"text": input_path.read_text(encoding="utf-8").splitlines()})
    return load_dataframe(input_path)


def _load_prediction_features(experiment, args: argparse.Namespace):
    if experiment.feature_kind == "text":
        if not args.predict_data_path:
            raise ValueError("Text experiments require --predict-data-path")
        predict_df = _load_prediction_dataframe(args.predict_data_path)
        return predict_df["text"].tolist(), predict_df

    if not args.predict_embeddings_path:
        raise ValueError("Embedding experiments require --predict-embeddings-path")
    predict_df = _load_prediction_dataframe(args.predict_data_path)
    return np.load(args.predict_embeddings_path), predict_df


def _load_training_features(experiment, args: argparse.Namespace):
    X_text, y = load_clean_data(args.train_data_path)
    if experiment.feature_kind == "text":
        return X_text, y
    if not args.train_embeddings_path:
        raise ValueError("Embedding experiments require --train-embeddings-path")
    return np.load(args.train_embeddings_path), y


def _resolve_output_kind(task: str, output_path: str, requested: str) -> str:
    if requested != "auto":
        return requested

    suffix = Path(output_path).suffix.lower()
    if suffix == ".txt":
        return "scores" if task == "presidents" else "labels"
    return "full"


def _prediction_ints(task: str, predictions) -> np.ndarray:
    predictions = np.asarray(predictions)
    if task == "presidents":
        return (predictions == presidents_task.positive_label).astype(int)
    return predictions.astype(int)


def _prediction_labels(task: str, predictions) -> list[str]:
    if task == "presidents":
        mapping = {
            presidents_task.negative_label: "M",
            presidents_task.positive_label: "C",
        }
        return [mapping[int(prediction)] for prediction in predictions]
    return [str(int(prediction)) for prediction in predictions]


def _build_prediction_frame(task: str, predict_df: pd.DataFrame | None, predictions, scores) -> pd.DataFrame:
    prediction_int = _prediction_ints(task, predictions)
    if task == "movies":
        if predict_df is not None and "review_id" in predict_df.columns:
            review_ids = predict_df["review_id"].to_numpy()
        else:
            review_ids = np.arange(len(prediction_int))

        frame = pd.DataFrame(
            {
                "review_id": review_ids,
                "prediction": prediction_int,
            }
        )
        if scores is not None:
            frame["score"] = scores
        return frame

    frame = pd.DataFrame()
    if predict_df is not None and "speech_id" in predict_df.columns:
        frame["speech_id"] = predict_df["speech_id"].to_numpy()
    if predict_df is not None and "sentence_id" in predict_df.columns:
        frame["sentence_id"] = predict_df["sentence_id"].to_numpy()
    frame["prediction_int"] = prediction_int
    frame["prediction_label"] = _prediction_labels(task, predictions)
    if scores is not None:
        frame["score"] = scores
    return frame


def _save_prediction_output(
    *,
    task: str,
    output_path: str,
    output_kind: str,
    predict_df: pd.DataFrame | None,
    predictions,
    scores,
):
    if output_kind == "scores":
        if scores is None:
            raise ValueError("This experiment does not expose continuous scores.")
        save_prediction_lines(output_path, scores, formatter=lambda value: f"{float(value):.8f}")
        return

    if output_kind == "labels":
        prediction_int = _prediction_ints(task, predictions)
        suffix = Path(output_path).suffix.lower()
        if suffix == ".txt":
            save_prediction_lines(output_path, prediction_int, formatter=lambda value: str(int(value)))
            return

        if task == "movies" and predict_df is not None and "review_id" in predict_df.columns:
            save_predictions(
                output_path,
                prediction_int,
                ids=predict_df["review_id"].to_numpy(),
                label_col="prediction",
            )
            return

        save_predictions(output_path, prediction_int, label_col="prediction")
        return

    if output_kind == "full":
        frame = _build_prediction_frame(task, predict_df, predictions, scores)
        save_dataframe(output_path, frame)
        return

    raise ValueError(f"Unsupported output kind: {output_kind}")


def run_predict(args: argparse.Namespace) -> int:
    word2vec_model = load_word2vec_model(args.word2vec_path) if args.word2vec_path else None
    experiment = get_experiment(args.experiment, word2vec_model=word2vec_model)
    if experiment.task != args.task:
        raise ValueError(f"{experiment.name} belongs to task {experiment.task}, not {args.task}")

    X_train, y_train = _load_training_features(experiment, args)
    X_pred, predict_df = _load_prediction_features(experiment, args)

    model = experiment.build()
    model.fit(X_train, y_train)
    task_spec = TASK_SPECS[args.task]
    scores = get_positive_class_scores(model, X_pred, task_spec.positive_label)
    predictions = model.predict(X_pred)

    if args.smooth_sigma > 0:
        if scores is None:
            raise ValueError("Score smoothing requires an estimator with continuous scores.")
        if predict_df is None or not args.smooth_group_col or args.smooth_group_col not in predict_df.columns:
            raise ValueError("Score smoothing requires --smooth-group-col and matching metadata.")
        from rital_nlp_project.presidents.utils import smooth_positive_scores

        scores = smooth_positive_scores(
            scores,
            sigma=args.smooth_sigma,
            groups=predict_df[args.smooth_group_col].to_numpy(),
        )
        predictions = np.where(scores >= 0.5, task_spec.positive_label, task_spec.negative_label)

    _save_prediction_output(
        task=args.task,
        output_path=args.output_path,
        output_kind=_resolve_output_kind(args.task, args.output_path, args.output_kind),
        predict_df=predict_df,
        predictions=predictions,
        scores=scores,
    )
    print(f"Wrote {len(predictions)} predictions to {args.output_path}")
    return 0


def run_embed(args: argparse.Namespace) -> int:
    input_path = Path(args.data_path)
    if input_path.suffix == ".txt":
        texts = input_path.read_text(encoding="utf-8").splitlines()
    else:
        texts = load_dataframe(input_path)[args.text_col].tolist()

    config = TransformerEmbeddingConfig(
        model_name=args.model_name,
        pooling=args.pooling,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
        normalize=args.normalize,
    )
    extractor = TransformerEmbeddingExtractor(config)
    embeddings = extractor.encode(texts)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)
    print(f"Wrote embeddings with shape {embeddings.shape} to {output_path}")
    return 0


def run_list_experiments(args: argparse.Namespace) -> int:
    word2vec_model = load_word2vec_model(args.word2vec_path) if args.word2vec_path else None
    tasks = [args.task] if args.task else ["movies", "presidents"]
    for task in tasks:
        print(f"[{task}]")
        for experiment in get_experiments(task, word2vec_model=word2vec_model):
            print(f"- {experiment.name}: {experiment.description} ({experiment.feature_kind})")
    return 0


def run_train_word2vec(args: argparse.Namespace) -> int:
    texts, _ = load_clean_data(args.data_path)
    output = train_word2vec_model(
        texts,
        output_path=args.output_path,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        sg=args.sg,
        epochs=args.epochs,
        workers=args.workers,
    )
    print(f"Wrote Word2Vec model to {output}")
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "prepare":
        return run_prepare(args)
    if args.command == "benchmark":
        return run_benchmark(args)
    if args.command == "predict":
        return run_predict(args)
    if args.command == "embed":
        return run_embed(args)
    if args.command == "list-experiments":
        return run_list_experiments(args)
    if args.command == "train-word2vec":
        return run_train_word2vec(args)
    raise ValueError(f"Unknown command: {args.command}")
