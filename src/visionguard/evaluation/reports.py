import json
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd

class BenchmarkReporter:
    """Saves and formats model comparison reports and metrics to JSON and CSV."""

    def __init__(self, reports_dir: str | Path = "./reports"):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def save_experiment_report(
        self,
        experiment_id: str,
        metrics: Dict[str, Any],
        model_info: Dict[str, Any],
        dataset_info: Dict[str, Any]
    ) -> Path:
        """Saves a structured experiment report to JSON."""
        report = {
            "experiment_id": experiment_id,
            "model_info": model_info,
            "dataset_info": dataset_info,
            "metrics": metrics
        }
        json_path = self.reports_dir / f"{experiment_id}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        return json_path

    def save_comparison_table(
        self,
        records: List[Dict[str, Any]],
        filename: str = "model_comparison.csv"
    ) -> Path:
        """Saves tabular comparison of multiple models or categories."""
        df = pd.DataFrame(records)
        csv_path = self.reports_dir / filename
        df.to_csv(csv_path, index=False)
        return csv_path
