from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SweepConfig:
    name: str
    minority_weight_scale: float
    dropout: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the requested simple presidents BiLSTM sweep, aggregate OOF "
            "metrics, and optionally generate best-run submission files."
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
        "--output-root",
        default="Dataset/out",
        help="Base directory used for all per-run outputs and summaries.",
    )
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python interpreter used to launch the train/predict scripts.",
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
        "--include-optional-dropout-03",
        action="store_true",
        help="Also run the optional dropout=0.3 configs described in the request.",
    )
    parser.add_argument(
        "--skip-prediction",
        action="store_true",
        help="Skip the final test-set prediction step even if test files exist.",
    )
    parser.add_argument(
        "--extra-raw-threshold-offsets",
        nargs="*",
        type=float,
        default=[0.0, 0.01, 0.02],
        help="Raw-threshold offsets used for extra label-only submission variants.",
    )
    return parser.parse_args()


def build_sweep_configs(include_optional_dropout_03: bool) -> list[SweepConfig]:
    configs = [
        SweepConfig("mw100", 1.00, 0.2),
        SweepConfig("mw090", 0.90, 0.2),
        SweepConfig("mw075", 0.75, 0.2),
        SweepConfig("mw060", 0.60, 0.2),
    ]
    if include_optional_dropout_03:
        configs.extend(
            [
                SweepConfig("mw090_do030", 0.90, 0.3),
                SweepConfig("mw075_do030", 0.75, 0.3),
            ]
        )
    return configs


def training_output_dir(output_root: Path, config: SweepConfig) -> Path:
    return output_root / f"presidents_lstm_simple_tuned_{config.name}"


def submission_output_dir(output_root: Path, config_name: str) -> Path:
    return output_root / f"presidents_lstm_simple_submission_tuned_{config_name}"


def ensure_input_exists(path_str: str) -> None:
    path = Path(path_str)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Required input file does not exist: {path}")


def run_command(
    command: list[str],
    *,
    log_path: Path,
) -> dict[str, Any]:
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


def load_run_metrics(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
    return json.loads(metrics_path.read_text())


def summary_row_from_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    config = metrics["config"]
    calibration = metrics["recommended_calibration"]
    tuned_metrics = calibration["oof_metrics_at_best_threshold"]
    tuned_boundary = calibration["oof_boundary_diagnostics_at_best_threshold"]
    default_metrics = calibration["oof_metrics_at_0p5"]

    return {
        "run_name": Path(config["output_dir"]).name,
        "output_dir": str(Path(config["output_dir"]).resolve()),
        "minority_weight_scale": float(config["minority_weight_scale"]),
        "dropout": float(config["dropout"]),
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


def choose_best_run(summary_frame: pd.DataFrame) -> pd.Series:
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


def dataframe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def series_record(series: pd.Series) -> dict[str, Any]:
    return dataframe_records(series.to_frame().T)[0]


def main() -> None:
    args = parse_args()
    ensure_input_exists(args.embeddings_path)
    ensure_input_exists(args.metadata_path)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    sweep_output_dir = output_root / "presidents_lstm_simple_tuned_sweep"
    sweep_output_dir.mkdir(parents=True, exist_ok=True)

    configs = build_sweep_configs(args.include_optional_dropout_03)
    command_records: list[dict[str, Any]] = []
    summary_rows = []

    for config in configs:
        run_dir = training_output_dir(output_root, config)
        log_path = run_dir / "run.log"
        command = [
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
            str(args.hidden_dim),
            "--projection-dim",
            str(args.projection_dim),
            "--num-layers",
            str(args.num_layers),
            "--dropout",
            str(config.dropout),
            "--batch-size",
            str(args.batch_size),
            "--epochs",
            str(args.epochs),
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
        command_records.append(run_command(command, log_path=log_path))
        metrics = load_run_metrics(run_dir)
        summary_rows.append(summary_row_from_metrics(metrics))

    summary_frame = pd.DataFrame(summary_rows)
    best_run = choose_best_run(summary_frame)

    comparison_csv_path = sweep_output_dir / "comparison_summary.csv"
    comparison_json_path = sweep_output_dir / "comparison_summary.json"
    commands_path = sweep_output_dir / "commands_used.json"

    summary_frame.to_csv(comparison_csv_path, index=False)
    best_run_payload = series_record(best_run)
    comparison_payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "best_run": best_run_payload,
        "runs": dataframe_records(summary_frame),
    }
    comparison_json_path.write_text(json.dumps(comparison_payload, indent=2))
    commands_path.write_text(json.dumps(command_records, indent=2))

    prediction_record = None
    if not args.skip_prediction:
        ensure_input_exists(args.test_embeddings_path)
        ensure_input_exists(args.test_metadata_path)

        best_run_dir = Path(best_run["output_dir"])
        best_config_name = str(best_run["run_name"]).replace(
            "presidents_lstm_simple_tuned_",
            "",
        )
        predict_dir = submission_output_dir(output_root, best_config_name)
        predict_log_path = predict_dir / "run.log"

        predict_command = [
            args.python_executable,
            "spark/predict_pres_lstm_simple.py",
            "--checkpoint-path",
            str(best_run_dir / "checkpoint.pt"),
            "--calibration-path",
            str(best_run_dir / "calibration.json"),
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
        command_records.append(prediction_record)
        commands_path.write_text(json.dumps(command_records, indent=2))

    payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "comparison_csv": str(comparison_csv_path.resolve()),
        "comparison_json": str(comparison_json_path.resolve()),
        "commands_used": str(commands_path.resolve()),
        "best_run": best_run_payload,
        "prediction": prediction_record,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
