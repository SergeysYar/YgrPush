"""Export cross-validation results to files."""

import json
from pathlib import Path
from typing import Any
import pandas as pd
import numpy as np


class CVResultsExporter:
    """Export CV results to various formats."""

    @staticmethod
    def export_cv_report_json(results: dict[str, Any], output_path: Path | str) -> None:
        """Export CV metrics to JSON report.
        
        Args:
            results: Training results with cv_metrics from all models
            output_path: Where to save the JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report = {}
        for model_type, model_results in results.items():
            if model_results is None:
                continue

            report[model_type] = {}
            for target, target_result in model_results.items():
                if target_result is None:
                    report[model_type][target] = None
                    continue

                if "cv_result" in target_result:
                    cv_data = target_result["cv_result"]
                    report[model_type][target] = {
                        "cv_metrics": cv_data["cv_metrics"],
                        "fold_results": cv_data["fold_results"],
                        "n_samples": target_result["n_samples"],
                        "n_folds": target_result["n_folds"],
                    }
                elif "status" in target_result:
                    # For PLS multivariate
                    report[model_type][target] = {
                        "status": target_result["status"],
                        "n_samples": target_result.get("n_samples", 0),
                    }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"✓ CV report exported to {output_path}")

    @staticmethod
    def export_predictions_csv(results: dict[str, Any], output_path: Path | str) -> None:
        """Export cross-validation predictions to CSV.
        
        Args:
            results: Training results with predictions
            output_path: Where to save the CSV file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = []
        for model_type, model_results in results.items():
            if model_results is None:
                continue

            for target, target_result in model_results.items():
                if target_result is None:
                    continue

                if "cv_result" in target_result:
                    cv_data = target_result["cv_result"]
                    predictions = cv_data["predictions"]

                    for idx, pred in enumerate(predictions):
                        data.append({
                            "model": model_type,
                            "target": target,
                            "row_idx": idx,
                            "prediction": float(pred),
                        })

        if data:
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)
            print(f"✓ Predictions exported to {output_path}")
        else:
            print(f"⚠ No predictions to export")

    @staticmethod
    def export_summary_metrics(results: dict[str, Any], output_path: Path | str) -> None:
        """Export summary metrics comparison table.
        
        Args:
            results: Training results
            output_path: Where to save the CSV file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for model_type, model_results in results.items():
            if model_results is None:
                continue

            for target, target_result in model_results.items():
                if target_result is None:
                    continue

                if "cv_result" in target_result:
                    metrics = target_result["cv_result"]["cv_metrics"]
                    rows.append({
                        "Model": model_type,
                        "Target": target,
                        "MAE": f"{metrics['mae']:.4f}",
                        "RMSE": f"{metrics['rmse']:.4f}",
                        "Median AE": f"{metrics['median_ae']:.4f}",
                        "R²": f"{metrics['r2']:.4f}",
                        "N Samples": metrics["n_samples"],
                        "N Folds": metrics["n_folds"],
                    })

        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)
            print(f"✓ Summary metrics exported to {output_path}")
        else:
            print(f"⚠ No metrics to export")

    @staticmethod
    def print_summary_table(results: dict[str, Any]) -> None:
        """Print summary table to console."""
        print("\n" + "=" * 90)
        print("CROSS-VALIDATION RESULTS SUMMARY")
        print("=" * 90)

        rows = []
        for model_type, model_results in results.items():
            if model_results is None:
                continue

            for target, target_result in model_results.items():
                if target_result is None:
                    continue

                if "cv_result" in target_result:
                    metrics = target_result["cv_result"]["cv_metrics"]
                    rows.append({
                        "Model": f"{model_type:<15}",
                        "Target": f"{target:<12}",
                        "MAE": f"{metrics['mae']:>8.4f}",
                        "RMSE": f"{metrics['rmse']:>8.4f}",
                        "Median AE": f"{metrics['median_ae']:>8.4f}",
                        "R²": f"{metrics['r2']:>8.4f}",
                        "Samples": f"{metrics['n_samples']:>6}",
                    })

        if rows:
            # Print header
            header = "Model          | Target       | MAE      | RMSE     | Median AE | R²       | Samples"
            print(header)
            print("-" * 90)

            # Print rows
            for row in rows:
                print(f"{row['Model']} | {row['Target']} | {row['MAE']} | {row['RMSE']} | {row['Median AE']} | {row['R²']} | {row['Samples']}")

        print("=" * 90)
