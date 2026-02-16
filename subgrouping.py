from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def make_gender_intersections(
    df: pd.DataFrame,
    gender_col: str,
    other_cols: List[str],
) -> pd.DataFrame:
    """
    Creates columns like:
      gender_X_ethnicity, gender_X_sexual_orientation, gender_X_employment
    """
    df = df.copy()
    for c in other_cols:
        out = f"{gender_col}_X_{c}"
        df[out] = df[gender_col].astype(str) + "_" + df[c].astype(str)
        df.loc[df[gender_col].isna() | df[c].isna(), out] = np.nan
    return df


def list_group_values(df: pd.DataFrame, group_col: str) -> List[str]:
    vals = df[group_col].dropna().astype(str).unique().tolist()
    vals = [v for v in vals if v.lower() not in ("unknown", "nan")]
    return sorted(vals)


def subgroup_masks(df: pd.DataFrame, group_col: str) -> Dict[str, np.ndarray]:
    masks = {}
    for v in list_group_values(df, group_col):
        masks[v] = (df[group_col].astype(str) == v).to_numpy()
    return masks
