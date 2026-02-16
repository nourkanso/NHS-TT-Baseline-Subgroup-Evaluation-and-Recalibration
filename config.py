from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class Paper3Config:
    seed: int = 42
    test_size: float = 0.30

    # Quantile bins for ECE and reliability diagrams
    ece_bins: int = 10

    # Min subgroup size to report metrics (avoid noisy tiny groups)
    min_group_n_report: int = 100

    # Subgroup recalibration safeguards (train-only)
    min_subgroup_train_n: int = 50
    min_subgroup_train_pos: int = 5
    min_subgroup_train_neg: int = 5

    # Column names
    outcome_col: str = ""
    base_pred_col: str = ""  # will be filled per run, e.g. "pred_base"

    # Group columns (single-attribute)
    group_cols: List[str] = None  # e.g. ["gender", "sexual_orientation", "ethnicity", "employment"]

    # For intersectional (gender x other)
    gender_col: str = "gender"
    intersection_with: List[str] = None  # e.g. ["sexual_orientation", "ethnicity", "employment"]

    # Paths
    data_path: str = "data/analysis_table_baseline.parquet"
    model_path: str = ""  # per outcome
    output_dir: str = "outputs"

    # Predictors expected by the Paper 1 model pipeline
    predictor_columns: List[str] = None

    # Optional: outcome-specific analytic sample filter
    filter_query: Optional[str] = None


def default_groups() -> List[str]:
    return ["gender", "sexual_orientation", "ethnicity", "employment"]
