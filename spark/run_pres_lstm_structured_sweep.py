from __future__ import annotations

import argparse
import itertools
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from rital_nlp_project.presidents.eval_utils import compute_boundary_diagnostics


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class StructuredConfig:
    name: str
    hidden_dim: int
    projection_dim: int
    dropout: float
    min_negative_span: int
    learning_rate: float
    epochs: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the explainable structured presidents BiLSTM sweep: same BiLSTM core "
            "as the simple lane, but decoded with a single contiguous minority span."
        )
    )
    parser.add_argument(
        "--embeddings-path",
        default="Dataset/embeddings/presidents_camembert_base_mean.npy",
    )
    parser.add_argument(
        "--metadata-path",
        default="Dataset/clean/presidents_clean_bert.parquet",
    )
    parser.add_argument(
        "--test-embeddings-path",
        default="Dataset/embeddings/presidents_test_camembert_base_mean.npy",
    )
    parser.add_argument(
        "--test-metadata-path",
        default="Dataset/clean/presidents_test_clean_bert.parquet",
    )
    parser.add_argument(
        "--proxy-labels-path",
        default=None,
    )
    parser.add_argument(
        "--output-root",
        default="Dataset/out",
    )
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--inner-validation-size", type=float, default=0.15)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--min-negative-span", type=int, default=3)
    parser.add_argument("--threshold-start", type=float, default=0.05)
    parser.add_argument("--threshold-end", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--hidden-dims",
        nargs="+",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--projection-dims",
        nargs="+",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--dropouts",
        nargs="+",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--min-negative-spans",
        nargs="+",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--learning-rates",
        nargs="+",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--epochs-grid",
        nargs="+",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--ranking-mode",
        choices=("clean", "proxy", "hybrid"),
        default="hybrid",
    )
    parser.add_argument(
        "--per-run-prediction",
        action="store_true",
        help="Predict and proxy-score every config instead of only the shortlisted ones.",
    )
    parser.add_argument(
        "--skip-prediction",
        action="store_true",
    )
    parser.add_argument(
        "--top-k-no-span-bias",
        type=int,
        default=4,
        help="How many clean top runs receive the late no-span-bias sweep.",
    )
    parser.add_argument("--no-span-bias-start", type=float, default=0.0)
    parser.add_argument("--no-span-bias-end", type=float, default=1.5)
    parser.add_argument("--no-span-bias-step", type=float, default=0.1)
    parser.add_argument(
        "--skip-no-span-bias-sweep",
        action="store_true",
    )
    return parser.parse_args()


def ensure_input_exists(path_str: str | None) -> None:
    if path_str is None:
        return
    path = Path(path_str)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Required input file does not exist: {path}")


def float_tag(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def run_base_dir(output_root: Path, config: StructuredConfig) -> Path:
    return output_root / f"presidents_lstm_structured_{config.name}"


def calibration_output_dir(output_root: Path, config: StructuredConfig) -> Path:
    return run_base_dir(output_root, config) / "cv_calibration"


def full_train_output_dir(output_root: Path, config: StructuredConfig) -> Path:
    return run_base_dir(output_root, config) / "full_train"


def submission_output_dir(output_root: Path, config_name: str, *, suffix: str | None = None) -> Path:
    base = output_root / f"presidents_lstm_structured_submission_{config_name}"
    return base if suffix is None else base / suffix


def run_command(command: list[str], *, log_path: Path) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now().astimezone().isoformat()
    start = time.perf_counter()

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("$ " + shlex.join(command) + "\n\n")
        subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

    return {
        "command": command,
        "shell_command": shlex.join(command),
        "log_path": str(log_path.resolve()),
        "started_at": started_at,
        "ended_at": datetime.now().astimezone().isoformat(),
        "seconds": time.perf_counter() - start,
    }


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    return json.loads(path.read_text())


def resolve_grid_values(
    override: Iterable[Any] | None,
    fallback: Iterable[Any],
) -> list[Any]:
    values = list(fallback if override is None else override)
    return list(dict.fromkeys(values))


def build_configs(args: argparse.Namespace) -> list[StructuredConfig]:
    hidden_dims = resolve_grid_values(args.hidden_dims, [args.hidden_dim])
    projection_dims = resolve_grid_values(args.projection_dims, [args.projection_dim])
    dropouts = resolve_grid_values(args.dropouts, [args.dropout])
    min_negative_spans = resolve_grid_values(
        args.min_negative_spans,
        [args.min_negative_span],
    )
    learning_rates = resolve_grid_values(args.learning_rates, [args.learning_rate])
    epochs_grid = resolve_grid_values(args.epochs_grid, [args.epochs])

    configs = []
    for hidden_dim, projection_dim, dropout, min_negative_span, learning_rate, epochs in itertools.product(
        hidden_dims,
        projection_dims,
        dropouts,
        min_negative_spans,
        learning_rates,
        epochs_grid,
    ):
        name = (
            f"hd{hidden_dim}_pd{projection_dim}_do{float_tag(dropout)}_"
            f"mns{min_negative_span}_lr{float_tag(learning_rate)}_ep{epochs}"
        )
        configs.append(
            StructuredConfig(
                name=name,
                hidden_dim=int(hidden_dim),
                projection_dim=int(projection_dim),
                dropout=float(dropout),
                min_negative_span=int(min_negative_span),
                learning_rate=float(learning_rate),
                epochs=int(epochs),
            )
        )

    if args.max_runs is not None and args.max_runs < len(configs):
        rng = np.random.default_rng(args.seed)
        selected = np.sort(
            rng.choice(len(configs), size=args.max_runs, replace=False)
        )
        configs = [configs[int(idx)] for idx in selected]
    return configs


def median_fold_epoch(metrics: dict[str, Any]) -> int:
    epochs = [int(fold["best_epoch"]) for fold in metrics["folds"]]
    if not epochs:
        raise ValueError("No fold epochs found in calibration metrics")
    return int(np.rint(np.median(np.asarray(epochs, dtype=np.float64))))


def build_summary_row(
    *,
    metrics: dict[str, Any],
    calibration_dir: Path,
    final_epoch: int,
) -> dict[str, Any]:
    calibration = metrics["recommended_calibration"]
    selected_score_name = str(metrics["selected_score_name"])
    oof_path = Path(metrics["outputs"]["oof_predictions"])
    oof_frame = pd.read_csv(oof_path)
    boundary_summary, per_speech = compute_boundary_diagnostics(
        oof_frame,
        threshold=float(calibration["threshold"]),
        score_column=selected_score_name,
    )
    per_speech_path = calibration_dir / "oof_speech_metrics_at_best_threshold.csv"
    per_speech.to_csv(per_speech_path, index=False)

    config = metrics["config"]
    return {
        "run_name": calibration_dir.parent.name,
        "run_dir": str(calibration_dir.parent.resolve()),
        "calibration_dir": str(calibration_dir.resolve()),
        "hidden_dim": int(config["hidden_dim"]),
        "projection_dim": int(config["projection_dim"]),
        "dropout": float(config["dropout"]),
        "epochs": int(config["epochs"]),
        "learning_rate": float(config["learning_rate"]),
        "min_negative_span": int(config["min_negative_span"]),
        "selected_score_name": selected_score_name,
        "oof_f1_at_tuned_threshold": float(calibration["f1_macro"]),
        "oof_accuracy_at_tuned_threshold": float(calibration["accuracy"]),
        "oof_balanced_accuracy_at_tuned_threshold": float(
            calibration["balanced_accuracy"]
        ),
        "tuned_threshold": float(calibration["threshold"]),
        "delta": float(calibration["equivalent_score_delta"]),
        "oof_mitterrand_count": int(calibration["mitterrand_count"]),
        "final_train_epochs": int(final_epoch),
        "start_lag_mean": boundary_summary["start_lag_mean"],
        "start_lag_median": boundary_summary["start_lag_median"],
        "end_lag_mean": boundary_summary["end_lag_mean"],
        "end_lag_median": boundary_summary["end_lag_median"],
        "multi_block_speeches": int(
            boundary_summary["speeches_with_multiple_predicted_mitterrand_blocks"]
        ),
        "speeches_with_any_predicted_block": int(
            boundary_summary["speeches_with_predicted_mitterrand_block"]
        ),
        "oof_speech_metrics_path": str(per_speech_path.resolve()),
    }


def choose_best_run(summary_frame: pd.DataFrame, *, ranking_mode: str) -> pd.Series:
    has_proxy = "proxy_f1_macro" in summary_frame.columns and summary_frame[
        "proxy_f1_macro"
    ].notna().any()

    if ranking_mode == "clean" or not has_proxy:
        ordered = summary_frame.sort_values(
            [
                "oof_f1_at_tuned_threshold",
                "end_lag_mean",
                "multi_block_speeches",
            ],
            ascending=[False, True, True],
            na_position="last",
        ).reset_index(drop=True)
        return ordered.iloc[0]

    if ranking_mode == "proxy":
        ordered = summary_frame.sort_values(
            [
                "proxy_f1_macro",
                "proxy_balanced_accuracy",
                "oof_f1_at_tuned_threshold",
                "end_lag_mean",
                "multi_block_speeches",
            ],
            ascending=[False, False, False, True, True],
            na_position="last",
        ).reset_index(drop=True)
        return ordered.iloc[0]

    ordered = summary_frame.sort_values(
        [
            "proxy_f1_macro",
            "oof_f1_at_tuned_threshold",
            "proxy_balanced_accuracy",
            "end_lag_mean",
            "multi_block_speeches",
        ],
        ascending=[False, False, False, True, True],
        na_position="last",
    ).reset_index(drop=True)
    return ordered.iloc[0]


def dataframe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def series_record(series: pd.Series) -> dict[str, Any]:
    return dataframe_records(series.to_frame().T)[0]


def predict_structured(
    *,
    args: argparse.Namespace,
    config: StructuredConfig,
    output_root: Path,
    no_span_bias: float = 0.0,
    suffix: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
    predict_dir = submission_output_dir(output_root, config.name, suffix=suffix)
    predict_command = [
        args.python_executable,
        "spark/predict_pres_lstm.py",
        "--checkpoint-path",
        str(full_train_output_dir(output_root, config) / "presidents_lstm.pt"),
        "--calibration-path",
        str(calibration_output_dir(output_root, config) / "calibration.json"),
        "--test-embeddings-path",
        args.test_embeddings_path,
        "--test-metadata-path",
        args.test_metadata_path,
        "--prior-embeddings-path",
        args.embeddings_path,
        "--prior-metadata-path",
        args.metadata_path,
        "--output-dir",
        str(predict_dir),
        "--device",
        args.device,
        "--no-span-bias",
        str(no_span_bias),
    ]
    prediction_record = run_command(predict_command, log_path=predict_dir / "run.log")
    prediction_metrics = load_json(predict_dir / "metrics.json")

    proxy_metrics = None
    if args.proxy_labels_path is not None:
        proxy_dir = predict_dir / "proxy_eval"
        calibrated_outputs = prediction_metrics.get("calibrated_outputs") or {}
        label_path = calibrated_outputs.get("submission_label_calibrated")
        score_path = calibrated_outputs.get("submission_prob_calibrated")
        if label_path is None or score_path is None:
            raise ValueError(
                "Predict metrics did not expose calibrated outputs for proxy scoring"
            )
        proxy_command = [
            args.python_executable,
            "scripts/eval_presidents_proxy.py",
            "--prediction-path",
            str(label_path),
            "--score-path",
            str(score_path),
            "--proxy-labels-path",
            args.proxy_labels_path,
            "--output-dir",
            str(proxy_dir),
        ]
        run_command(proxy_command, log_path=proxy_dir / "run.log")
        proxy_metrics = load_json(proxy_dir / "metrics.json")

    return prediction_record, proxy_metrics, prediction_metrics


def attach_proxy_metrics(row: dict[str, Any], *, proxy_metrics: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(row)
    enriched["proxy_f1_macro"] = float(proxy_metrics["f1_macro"])
    enriched["proxy_accuracy"] = float(proxy_metrics["accuracy"])
    enriched["proxy_balanced_accuracy"] = float(proxy_metrics["balanced_accuracy"])
    enriched["proxy_agreement"] = float(proxy_metrics["agreement"])
    enriched["proxy_disagreement_count"] = int(proxy_metrics["disagreement_count"])
    enriched["proxy_accepted_rows"] = int(proxy_metrics["accepted_rows"])
    enriched["proxy_unresolved_rows"] = int(proxy_metrics["unresolved_rows"])
    return enriched


def build_bias_grid(start: float, end: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("no-span-bias-step must be strictly positive")
    if end < start:
        raise ValueError("no-span-bias-end must be >= no-span-bias-start")
    num_steps = int(np.floor((end - start) / step))
    values = [round(start + step * idx, 10) for idx in range(num_steps + 1)]
    if values[-1] < end:
        values.append(round(end, 10))
    return values


def main() -> None:
    args = parse_args()
    ensure_input_exists(args.embeddings_path)
    ensure_input_exists(args.metadata_path)
    if args.proxy_labels_path is not None:
        ensure_input_exists(args.proxy_labels_path)

    should_predict_per_run = bool(args.per_run_prediction or args.proxy_labels_path is not None)
    if should_predict_per_run and args.skip_prediction:
        raise ValueError("Cannot combine --skip-prediction with per-run prediction")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    sweep_output_dir = output_root / "presidents_lstm_structured_sweep"
    sweep_output_dir.mkdir(parents=True, exist_ok=True)

    configs = build_configs(args)
    if not configs:
        raise ValueError("No structured sweep configurations were generated")

    command_records: list[dict[str, Any]] = []
    summary_rows = []

    for config in configs:
        calib_dir = calibration_output_dir(output_root, config)
        calibrate_command = [
            args.python_executable,
            "spark/calibrate_pres_lstm_cv.py",
            "--embeddings-path",
            args.embeddings_path,
            "--metadata-path",
            args.metadata_path,
            "--output-dir",
            str(calib_dir),
            "--n-splits",
            str(args.n_splits),
            "--inner-validation-size",
            str(args.inner_validation_size),
            "--hidden-dim",
            str(config.hidden_dim),
            "--projection-dim",
            str(config.projection_dim),
            "--num-layers",
            str(args.num_layers),
            "--dropout",
            str(config.dropout),
            "--batch-size",
            str(args.batch_size),
            "--epochs",
            str(config.epochs),
            "--learning-rate",
            str(config.learning_rate),
            "--weight-decay",
            str(args.weight_decay),
            "--gradient-clip",
            str(args.gradient_clip),
            "--min-negative-span",
            str(config.min_negative_span),
            "--position-features",
            "none",
            "--position-bins",
            "0",
            "--position-prior-weight",
            "0.0",
            "--selection-decoder",
            "single_negative_span",
            "--threshold-start",
            str(args.threshold_start),
            "--threshold-end",
            str(args.threshold_end),
            "--threshold-step",
            str(args.threshold_step),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
        ]
        command_records.append(run_command(calibrate_command, log_path=calib_dir / "run.log"))
        calibration_metrics = load_json(calib_dir / "metrics.json")
        final_epoch = median_fold_epoch(calibration_metrics)

        full_dir = full_train_output_dir(output_root, config)
        full_train_command = [
            args.python_executable,
            "spark/train_pres_lstm.py",
            "--embeddings-path",
            args.embeddings_path,
            "--metadata-path",
            args.metadata_path,
            "--output-dir",
            str(full_dir),
            "--hidden-dim",
            str(config.hidden_dim),
            "--projection-dim",
            str(config.projection_dim),
            "--num-layers",
            str(args.num_layers),
            "--dropout",
            str(config.dropout),
            "--batch-size",
            str(args.batch_size),
            "--epochs",
            str(final_epoch),
            "--learning-rate",
            str(config.learning_rate),
            "--weight-decay",
            str(args.weight_decay),
            "--gradient-clip",
            str(args.gradient_clip),
            "--min-negative-span",
            str(config.min_negative_span),
            "--position-features",
            "none",
            "--position-bins",
            "0",
            "--position-prior-weight",
            "0.0",
            "--selection-decoder",
            "single_negative_span",
            "--seed",
            str(args.seed + 999 + final_epoch),
            "--device",
            args.device,
            "--final-train-full-data",
        ]
        command_records.append(run_command(full_train_command, log_path=full_dir / "run.log"))

        row = build_summary_row(
            metrics=calibration_metrics,
            calibration_dir=calib_dir,
            final_epoch=final_epoch,
        )

        if should_predict_per_run:
            ensure_input_exists(args.test_embeddings_path)
            ensure_input_exists(args.test_metadata_path)
            prediction_record, proxy_metrics, _ = predict_structured(
                args=args,
                config=config,
                output_root=output_root,
                no_span_bias=0.0,
            )
            command_records.append(prediction_record)
            if proxy_metrics is not None:
                row = attach_proxy_metrics(row, proxy_metrics=proxy_metrics)

        summary_rows.append(row)

    summary_frame = pd.DataFrame(summary_rows)
    best_clean_run = choose_best_run(summary_frame, ranking_mode="clean")
    best_ranked_run = choose_best_run(summary_frame, ranking_mode=args.ranking_mode)

    if not should_predict_per_run and not args.skip_prediction:
        ensure_input_exists(args.test_embeddings_path)
        ensure_input_exists(args.test_metadata_path)
        best_name = str(best_ranked_run["run_name"]).replace("presidents_lstm_structured_", "")
        best_config = next(config for config in configs if config.name == best_name)
        prediction_record, proxy_metrics, _ = predict_structured(
            args=args,
            config=best_config,
            output_root=output_root,
            no_span_bias=0.0,
        )
        command_records.append(prediction_record)
        if proxy_metrics is not None:
            summary_frame.loc[
                summary_frame["run_name"] == best_ranked_run["run_name"],
                [
                    "proxy_f1_macro",
                    "proxy_accuracy",
                    "proxy_balanced_accuracy",
                    "proxy_agreement",
                    "proxy_disagreement_count",
                    "proxy_accepted_rows",
                    "proxy_unresolved_rows",
                ],
            ] = [
                float(proxy_metrics["f1_macro"]),
                float(proxy_metrics["accuracy"]),
                float(proxy_metrics["balanced_accuracy"]),
                float(proxy_metrics["agreement"]),
                int(proxy_metrics["disagreement_count"]),
                int(proxy_metrics["accepted_rows"]),
                int(proxy_metrics["unresolved_rows"]),
            ]
            best_ranked_run = choose_best_run(summary_frame, ranking_mode=args.ranking_mode)

    bias_summary_records: list[dict[str, Any]] = []
    if (
        not args.skip_no_span_bias_sweep
        and args.proxy_labels_path is not None
        and args.top_k_no_span_bias > 0
        and not summary_frame.empty
    ):
        clean_order = summary_frame.sort_values(
            [
                "oof_f1_at_tuned_threshold",
                "end_lag_mean",
                "multi_block_speeches",
            ],
            ascending=[False, True, True],
            na_position="last",
        ).reset_index(drop=True)
        top_names = clean_order.head(args.top_k_no_span_bias)["run_name"].tolist()
        bias_values = build_bias_grid(
            args.no_span_bias_start,
            args.no_span_bias_end,
            args.no_span_bias_step,
        )
        for run_name in top_names:
            config_name = str(run_name).replace("presidents_lstm_structured_", "")
            config = next(item for item in configs if item.name == config_name)
            per_run_records = []
            for bias in bias_values:
                suffix = f"bias_{float_tag(bias)}"
                prediction_record, proxy_metrics, prediction_metrics = predict_structured(
                    args=args,
                    config=config,
                    output_root=output_root,
                    no_span_bias=float(bias),
                    suffix=suffix,
                )
                command_records.append(prediction_record)
                if proxy_metrics is None:
                    continue
                per_run_records.append(
                    {
                        "run_name": run_name,
                        "config_name": config_name,
                        "no_span_bias": float(bias),
                        "prediction_dir": str(
                            submission_output_dir(output_root, config.name, suffix=suffix).resolve()
                        ),
                        "proxy_f1_macro": float(proxy_metrics["f1_macro"]),
                        "proxy_accuracy": float(proxy_metrics["accuracy"]),
                        "proxy_balanced_accuracy": float(proxy_metrics["balanced_accuracy"]),
                        "proxy_disagreement_count": int(
                            proxy_metrics["disagreement_count"]
                        ),
                        "label_counts": prediction_metrics.get("label_counts"),
                    }
                )
            if per_run_records:
                per_run_frame = pd.DataFrame(per_run_records).sort_values(
                    [
                        "proxy_f1_macro",
                        "proxy_balanced_accuracy",
                        "no_span_bias",
                    ],
                    ascending=[False, False, True],
                ).reset_index(drop=True)
                best_record = per_run_frame.iloc[0].to_dict()
                bias_summary_records.append(
                    {
                        "run_name": run_name,
                        "best_variant": best_record,
                        "variants": dataframe_records(per_run_frame),
                    }
                )
                per_run_frame.to_csv(
                    sweep_output_dir / f"{config_name}_no_span_bias_sweep.csv",
                    index=False,
                )

    comparison_csv_path = sweep_output_dir / "comparison_summary.csv"
    comparison_json_path = sweep_output_dir / "comparison_summary.json"
    commands_path = sweep_output_dir / "commands_used.json"
    bias_summary_path = sweep_output_dir / "no_span_bias_shortlist_summary.json"

    summary_frame.to_csv(comparison_csv_path, index=False)
    comparison_payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "ranking_mode": args.ranking_mode,
        "best_ranked_run": series_record(best_ranked_run),
        "best_clean_run": series_record(best_clean_run),
        "runs": dataframe_records(summary_frame),
    }
    comparison_json_path.write_text(json.dumps(comparison_payload, indent=2))
    commands_path.write_text(json.dumps(command_records, indent=2))
    bias_summary_path.write_text(json.dumps(bias_summary_records, indent=2))

    payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "comparison_csv": str(comparison_csv_path.resolve()),
        "comparison_json": str(comparison_json_path.resolve()),
        "commands_used": str(commands_path.resolve()),
        "ranking_mode": args.ranking_mode,
        "best_ranked_run": series_record(best_ranked_run),
        "best_clean_run": series_record(best_clean_run),
        "no_span_bias_shortlist_summary": str(bias_summary_path.resolve()),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
