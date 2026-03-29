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


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SweepConfig:
    name: str
    hidden_dim: int
    projection_dim: int
    dropout: float
    minority_weight_scale: float
    epochs: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a configurable simple presidents BiLSTM sweep, aggregate clean OOF "
            "metrics, and optionally score every submission against the accepted "
            "archive proxy labels."
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
        help="Optional accepted archive-label CSV used for offline proxy scoring.",
    )
    parser.add_argument(
        "--output-root",
        default="Dataset/out",
        help="Base directory used for all per-run outputs and summaries.",
    )
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python interpreter used to launch the training and scoring scripts.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument(
        "--selection-threshold-mode",
        choices=("fixed", "sweep"),
        default="sweep",
    )
    parser.add_argument("--selection-threshold-start", type=float, default=0.80)
    parser.add_argument("--selection-threshold-end", type=float, default=0.95)
    parser.add_argument("--selection-threshold-step", type=float, default=0.005)
    parser.add_argument("--threshold-start", type=float, default=0.05)
    parser.add_argument("--threshold-end", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--extra-raw-threshold-offsets",
        nargs="*",
        type=float,
        default=[0.0, 0.01, 0.02],
        help="Raw-threshold offsets used for extra label-only submission variants.",
    )
    parser.add_argument(
        "--hidden-dims",
        nargs="+",
        type=int,
        default=None,
        help="Optional hidden-dim sweep values. Defaults to --hidden-dim.",
    )
    parser.add_argument(
        "--projection-dims",
        nargs="+",
        type=int,
        default=None,
        help="Optional projection-dim sweep values. Defaults to --projection-dim.",
    )
    parser.add_argument(
        "--dropouts",
        nargs="+",
        type=float,
        default=None,
        help="Dropout values for the sweep. Defaults to [0.2].",
    )
    parser.add_argument(
        "--minority-weight-scales",
        nargs="+",
        type=float,
        default=None,
        help="Minority-class loss scales. Defaults to [1.0, 0.9, 0.75, 0.6].",
    )
    parser.add_argument(
        "--epochs-grid",
        nargs="+",
        type=int,
        default=None,
        help="Epoch counts to sweep. Defaults to [--epochs].",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional cap; sampled deterministically from the full grid using --seed.",
    )
    parser.add_argument(
        "--ranking-mode",
        choices=("clean", "proxy", "hybrid"),
        default="hybrid",
        help="How the best run is selected once all metrics are available.",
    )
    parser.add_argument(
        "--per-run-prediction",
        action="store_true",
        help="Run test-set prediction for every config instead of only the best clean run.",
    )
    parser.add_argument(
        "--skip-prediction",
        action="store_true",
        help="Skip the final test-set prediction step entirely.",
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
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def training_output_dir(output_root: Path, config: SweepConfig) -> Path:
    return output_root / f"presidents_lstm_simple_tuned_{config.name}"


def submission_output_dir(output_root: Path, config_name: str) -> Path:
    return output_root / f"presidents_lstm_simple_submission_tuned_{config_name}"


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


def build_sweep_configs(args: argparse.Namespace) -> list[SweepConfig]:
    hidden_dims = resolve_grid_values(args.hidden_dims, [args.hidden_dim])
    projection_dims = resolve_grid_values(args.projection_dims, [args.projection_dim])
    dropouts = resolve_grid_values(args.dropouts, [0.2])
    minority_weight_scales = resolve_grid_values(
        args.minority_weight_scales,
        [1.0, 0.9, 0.75, 0.6],
    )
    epochs_grid = resolve_grid_values(args.epochs_grid, [args.epochs])

    configs = []
    for hidden_dim, projection_dim, dropout, minority_weight_scale, epochs in itertools.product(
        hidden_dims,
        projection_dims,
        dropouts,
        minority_weight_scales,
        epochs_grid,
    ):
        name = (
            f"hd{hidden_dim}_pd{projection_dim}_do{float_tag(dropout)}_"
            f"mw{float_tag(minority_weight_scale)}_ep{epochs}"
        )
        configs.append(
            SweepConfig(
                name=name,
                hidden_dim=int(hidden_dim),
                projection_dim=int(projection_dim),
                dropout=float(dropout),
                minority_weight_scale=float(minority_weight_scale),
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


def summary_row_from_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    config = metrics["config"]
    calibration = metrics["recommended_calibration"]
    tuned_metrics = calibration["oof_metrics_at_best_threshold"]
    tuned_boundary = calibration["oof_boundary_diagnostics_at_best_threshold"]
    default_metrics = calibration["oof_metrics_at_0p5"]

    return {
        "run_name": Path(config["output_dir"]).name,
        "output_dir": str(Path(config["output_dir"]).resolve()),
        "hidden_dim": int(config["hidden_dim"]),
        "projection_dim": int(config["projection_dim"]),
        "dropout": float(config["dropout"]),
        "epochs": int(config["epochs"]),
        "minority_weight_scale": float(config["minority_weight_scale"]),
        "best_epoch": int(metrics["best_epoch"]),
        "validation_best_f1": float(metrics["best_validation_f1_macro"]),
        "validation_best_threshold": float(
            metrics["best_validation_selection"]["threshold"]
        ),
        "validation_mitterrand_count": int(
            metrics["best_validation_selection"]["mitterrand_count"]
        ),
        "oof_f1_at_0p5": float(default_metrics["f1_macro"]),
        "oof_f1_at_tuned_threshold": float(tuned_metrics["f1_macro"]),
        "tuned_threshold": float(calibration["best_threshold"]),
        "delta": float(calibration["equivalent_score_delta"]),
        "oof_mitterrand_count": int(tuned_metrics["mitterrand_count"]),
        "start_lag_mean": tuned_boundary["start_lag_mean"],
        "start_lag_median": tuned_boundary["start_lag_median"],
        "end_lag_mean": tuned_boundary["end_lag_mean"],
        "end_lag_median": tuned_boundary["end_lag_median"],
        "speeches_with_any_predicted_block": int(
            tuned_boundary["speeches_with_predicted_mitterrand_block"]
        ),
        "multi_block_speeches": int(
            tuned_boundary["speeches_with_multiple_predicted_mitterrand_blocks"]
        ),
    }


def attach_proxy_metrics(
    row: dict[str, Any],
    *,
    proxy_metrics: dict[str, Any],
    proxy_eval_dir: Path,
) -> dict[str, Any]:
    enriched = dict(row)
    enriched["proxy_eval_dir"] = str(proxy_eval_dir.resolve())
    enriched["proxy_f1_macro"] = float(proxy_metrics["f1_macro"])
    enriched["proxy_accuracy"] = float(proxy_metrics["accuracy"])
    enriched["proxy_balanced_accuracy"] = float(proxy_metrics["balanced_accuracy"])
    enriched["proxy_agreement"] = float(proxy_metrics["agreement"])
    enriched["proxy_disagreement_count"] = int(proxy_metrics["disagreement_count"])
    enriched["proxy_accepted_rows"] = int(proxy_metrics["accepted_rows"])
    enriched["proxy_unresolved_rows"] = int(proxy_metrics["unresolved_rows"])
    return enriched


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


def predict_and_optionally_score(
    *,
    args: argparse.Namespace,
    config_name: str,
    checkpoint_path: Path,
    calibration_path: Path,
    output_root: Path,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    predict_dir = submission_output_dir(output_root, config_name)
    predict_log_path = predict_dir / "run.log"
    predict_command = [
        args.python_executable,
        "spark/predict_pres_lstm_simple.py",
        "--checkpoint-path",
        str(checkpoint_path),
        "--calibration-path",
        str(calibration_path),
        "--test-embeddings-path",
        args.test_embeddings_path,
        "--test-metadata-path",
        args.test_metadata_path,
        "--output-dir",
        str(predict_dir),
        "--device",
        args.device,
        "--decision-threshold",
        "0.5",
    ]
    if args.extra_raw_threshold_offsets:
        predict_command.append("--extra-raw-threshold-offsets")
        predict_command.extend(str(offset) for offset in args.extra_raw_threshold_offsets)

    prediction_record = run_command(predict_command, log_path=predict_log_path)

    proxy_metrics = None
    if args.proxy_labels_path is not None:
        proxy_eval_dir = predict_dir / "proxy_eval"
        proxy_command = [
            args.python_executable,
            "scripts/eval_presidents_proxy.py",
            "--prediction-path",
            str(predict_dir / "submission_label_calibrated.csv"),
            "--score-path",
            str(predict_dir / "submission_prob_mitterrand_calibrated.csv"),
            "--proxy-labels-path",
            args.proxy_labels_path,
            "--output-dir",
            str(proxy_eval_dir),
        ]
        run_command(proxy_command, log_path=proxy_eval_dir / "run.log")
        proxy_metrics = load_json(proxy_eval_dir / "metrics.json")
    return prediction_record, proxy_metrics


def main() -> None:
    args = parse_args()
    ensure_input_exists(args.embeddings_path)
    ensure_input_exists(args.metadata_path)
    if args.proxy_labels_path is not None:
        ensure_input_exists(args.proxy_labels_path)

    should_predict_per_run = bool(args.per_run_prediction or args.proxy_labels_path is not None)
    if should_predict_per_run and args.skip_prediction:
        raise ValueError("Cannot use --skip-prediction together with per-run prediction")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    sweep_output_dir = output_root / "presidents_lstm_simple_tuned_sweep"
    sweep_output_dir.mkdir(parents=True, exist_ok=True)

    configs = build_sweep_configs(args)
    if not configs:
        raise ValueError("No sweep configurations were generated")

    command_records: list[dict[str, Any]] = []
    summary_rows = []

    for config in configs:
        run_dir = training_output_dir(output_root, config)
        train_command = [
            args.python_executable,
            "spark/train_pres_lstm_simple.py",
            "--embeddings-path",
            args.embeddings_path,
            "--metadata-path",
            args.metadata_path,
            "--output-dir",
            str(run_dir),
            "--validation-size",
            str(args.validation_size),
            "--n-folds",
            str(args.n_folds),
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
            str(args.learning_rate),
            "--weight-decay",
            str(args.weight_decay),
            "--gradient-clip",
            str(args.gradient_clip),
            "--selection-threshold-mode",
            args.selection_threshold_mode,
            "--selection-threshold-start",
            str(args.selection_threshold_start),
            "--selection-threshold-end",
            str(args.selection_threshold_end),
            "--selection-threshold-step",
            str(args.selection_threshold_step),
            "--threshold-start",
            str(args.threshold_start),
            "--threshold-end",
            str(args.threshold_end),
            "--threshold-step",
            str(args.threshold_step),
            "--minority-weight-scale",
            str(config.minority_weight_scale),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
        ]
        command_records.append(run_command(train_command, log_path=run_dir / "run.log"))

        metrics = load_json(run_dir / "metrics.json")
        row = summary_row_from_metrics(metrics)

        if should_predict_per_run:
            ensure_input_exists(args.test_embeddings_path)
            ensure_input_exists(args.test_metadata_path)
            prediction_record, proxy_metrics = predict_and_optionally_score(
                args=args,
                config_name=config.name,
                checkpoint_path=run_dir / "checkpoint.pt",
                calibration_path=run_dir / "calibration.json",
                output_root=output_root,
            )
            command_records.append(prediction_record)
            if proxy_metrics is not None:
                row = attach_proxy_metrics(
                    row,
                    proxy_metrics=proxy_metrics,
                    proxy_eval_dir=submission_output_dir(output_root, config.name) / "proxy_eval",
                )

        summary_rows.append(row)

    summary_frame = pd.DataFrame(summary_rows)
    best_clean_run = choose_best_run(summary_frame, ranking_mode="clean")
    best_ranked_run = choose_best_run(summary_frame, ranking_mode=args.ranking_mode)

    prediction_record = None
    if not should_predict_per_run and not args.skip_prediction:
        ensure_input_exists(args.test_embeddings_path)
        ensure_input_exists(args.test_metadata_path)

        best_run_dir = Path(best_ranked_run["output_dir"])
        best_config_name = str(best_ranked_run["run_name"]).replace(
            "presidents_lstm_simple_tuned_",
            "",
        )
        prediction_record, proxy_metrics = predict_and_optionally_score(
            args=args,
            config_name=best_config_name,
            checkpoint_path=best_run_dir / "checkpoint.pt",
            calibration_path=best_run_dir / "calibration.json",
            output_root=output_root,
        )
        command_records.append(prediction_record)
        if proxy_metrics is not None:
            summary_frame.loc[
                summary_frame["run_name"] == best_ranked_run["run_name"],
                [
                    "proxy_eval_dir",
                    "proxy_f1_macro",
                    "proxy_accuracy",
                    "proxy_balanced_accuracy",
                    "proxy_agreement",
                    "proxy_disagreement_count",
                    "proxy_accepted_rows",
                    "proxy_unresolved_rows",
                ],
            ] = [
                str((submission_output_dir(output_root, best_config_name) / "proxy_eval").resolve()),
                float(proxy_metrics["f1_macro"]),
                float(proxy_metrics["accuracy"]),
                float(proxy_metrics["balanced_accuracy"]),
                float(proxy_metrics["agreement"]),
                int(proxy_metrics["disagreement_count"]),
                int(proxy_metrics["accepted_rows"]),
                int(proxy_metrics["unresolved_rows"]),
            ]
            best_ranked_run = choose_best_run(summary_frame, ranking_mode=args.ranking_mode)

    comparison_csv_path = sweep_output_dir / "comparison_summary.csv"
    comparison_json_path = sweep_output_dir / "comparison_summary.json"
    commands_path = sweep_output_dir / "commands_used.json"

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

    payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "comparison_csv": str(comparison_csv_path.resolve()),
        "comparison_json": str(comparison_json_path.resolve()),
        "commands_used": str(commands_path.resolve()),
        "ranking_mode": args.ranking_mode,
        "best_ranked_run": series_record(best_ranked_run),
        "best_clean_run": series_record(best_clean_run),
        "prediction": prediction_record,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
