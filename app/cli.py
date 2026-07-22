from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .config import settings

APP_MODULE = "app.api.main:app"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CLI for shampoo quality forecasting system"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("inspect-db", help="Inspect production SQLite database")
    subparsers.add_parser("build-dataset", help="Build training dataset from production DB")
    subparsers.add_parser("validate-data", help="Run data validation checks")
    train_parser = subparsers.add_parser("train", help="Train registered models")
    train_parser.add_argument(
        "--model-types",
        nargs="+",
        default=["baseline", "ridge", "pls", "bayesian_ridge", "catboost"],
        choices=["baseline", "ridge", "pls", "bayesian_ridge", "catboost"],
    )
    train_parser.add_argument(
        "--protocol-policy",
        choices=["latest", "first", "all"],
        default=settings.target_protocol_policy,
    )
    train_parser.add_argument(
        "--batch-mode",
        action="store_true",
        help="Use one row per batch instead of snapshot training",
    )

    evaluate_parser = subparsers.add_parser("evaluate", help="Run evaluation and produce reports")
    evaluate_parser.add_argument(
        "--model-types",
        nargs="+",
        default=["baseline", "ridge", "pls", "bayesian_ridge", "catboost"],
        choices=["baseline", "ridge", "pls", "bayesian_ridge", "catboost"],
    )
    evaluate_parser.add_argument(
        "--protocol-policy",
        choices=["latest", "first", "all"],
        default=settings.target_protocol_policy,
    )
    evaluate_parser.add_argument(
        "--batch-mode",
        action="store_true",
        help="Use one row per batch instead of snapshot evaluation",
    )
    predict_parser = subparsers.add_parser("predict", help="Predict for one batch")
    predict_parser.add_argument("--batch-id", type=int, required=True)
    predict_parser.add_argument("--up-to-step", type=int)
    promote_parser = subparsers.add_parser("promote-model", help="Promote model to champion status")
    promote_parser.add_argument("model-id", type=str)
    subparsers.add_parser("list-models", help="List available trained models")
    subparsers.add_parser("update-actuals", help="Update actual values for predictions")
    subparsers.add_parser("run-api", help="Run FastAPI server")
    subparsers.add_parser("run-dashboard", help="Run Streamlit dashboard")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    match args.command:
        case "inspect-db":
            from .database.source import inspect_database

            inspect_database(settings.db_path)
            return 0
        case "build-dataset":
            from .data.dataset_builder import DatasetBuilder
            builder = DatasetBuilder(settings.db_path)
            summary = builder.get_data_summary()
            print(f"Dataset Summary:")
            print(f"  Total Measurements: {summary['total_measurements']}")
            print(f"  Total Batches: {summary['total_batches']}")
            print(f"  Batches with Targets: {summary['batches_with_targets']}")
            print(f"  Measurements with Issues: {summary['measurements_with_issues']}")
            print(f"  Missing Values: {summary['missing_count']}")
            print(f"  Invalid Values: {summary['invalid_count']}")
            print(f"  Out of Range Values: {summary['out_of_range_count']}")
            print(f"  Issues by Field: {summary['issues_by_field']}")
            return 0
        case "validate-data":
            from .data.dataset_builder import DatasetBuilder
            builder = DatasetBuilder(settings.db_path)
            measurements = builder.load_measurements()
            report = builder.quality_inspector.inspect_measurements(measurements)
            print(f"Data Validation Report:")
            print(f"  Total Measurements: {report.total_measurements}")
            print(f"  Measurements with Issues: {report.measurements_with_issues}")
            print(f"  Issues: {len(report.sample_issues)} samples shown")
            for issue in report.sample_issues[:5]:
                print(f"    - Batch {issue.batch_id}, Measurement {issue.measurement_id}: {issue.description}")
            return 0
        case "train":
            from .ml.service import TrainingPipeline
            pipeline = TrainingPipeline(settings.db_path, settings.ml_storage_path)
            results = pipeline.train_all(
                snapshot_mode=not args.batch_mode,
                protocol_policy=args.protocol_policy,
                model_types=args.model_types,
            )
            pipeline.save_cv_report(results)
            print("\nModels trained successfully")
            return 0
        case "evaluate":
            from .ml.service import TrainingPipeline

            pipeline = TrainingPipeline(settings.db_path, settings.ml_storage_path)
            results = pipeline.train_all(
                snapshot_mode=not args.batch_mode,
                protocol_policy=args.protocol_policy,
                model_types=args.model_types,
            )
            pipeline.save_cv_report(results)
            print("\nEvaluation complete")
            return 0
        case "predict":
            from .ml.prediction_service import PredictionService

            service = PredictionService(settings.db_path, settings.ml_storage_path)
            try:
                service.train_if_no_model()
                prediction = service.predict_batch(args.batch_id, up_to_step_order=args.up_to_step)
                print(json.dumps(prediction, indent=2, ensure_ascii=False))
                return 0
            except ValueError as error:
                print(f"Error: {error}")
                return 1
        case "list-models":
            from .database.ml_storage import ModelRegistry

            registry = ModelRegistry(settings.ml_storage_path)
            models = registry.list_models()
            print(json.dumps(models, indent=2, ensure_ascii=False))
            return 0
        case "promote-model":
            from .database.ml_storage import ModelRegistry

            registry = ModelRegistry(settings.ml_storage_path)
            registry.promote_model(args.model_id)
            print(f"Promoted model {args.model_id}")
            return 0
        case "update-actuals":
            from .ml.prediction_service import PredictionService

            service = PredictionService(settings.db_path, settings.ml_storage_path)
            updated_count = service.update_prediction_actuals()
            print(f"Updated actual values for {updated_count} prediction records.")
            return 0
        case "run-api":
            return _run_api()
        case "run-dashboard":
            return _run_dashboard()
        case _:
            parser.print_help()
            return 1


def _run_api() -> int:
    print("Starting FastAPI server on http://127.0.0.1:8000")
    subprocess.run(
        [sys.executable, "-m", "uvicorn", APP_MODULE, "--reload"], check=False
    )
    return 0


def _run_dashboard() -> int:
    print("Starting Streamlit dashboard on http://127.0.0.1:8501")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app/dashboard/app.py"], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
