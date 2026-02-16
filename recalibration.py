from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

from .metrics import clip_prob


@dataclass(frozen=True)
class RecalibrationParams:
    intercept: float
    slope: float


def fit_logistic_recalibration(
    y_train: np.ndarray,
    p_train: np.ndarray,
    eps: float = 1e-6,
) -> RecalibrationParams:
    """
    Fit logistic recalibration:
      logit(y) ~ a + b * logit(p)
    Using unpenalised logistic regression on one feature (logit(p)).
    """
    y_train = np.asarray(y_train, dtype=int)
    p_train = clip_prob(np.asarray(p_train, dtype=float), eps=eps)
    x = logit(p_train).reshape(-1, 1)

    clf = LogisticRegression(penalty=None, solver="lbfgs")
    clf.fit(x, y_train)

    return RecalibrationParams(
        intercept=float(clf.intercept_[0]),
        slope=float(clf.coef_[0][0]),
    )


def apply_logistic_recalibration(
    p: np.ndarray,
    params: RecalibrationParams,
    eps: float = 1e-6,
) -> np.ndarray:
    p = clip_prob(np.asarray(p, dtype=float), eps=eps)
    lp = logit(p)
    lp_new = params.intercept + params.slope * lp
    return expit(lp_new)


def global_recalibration(
    df: pd.DataFrame,
    outcome_col: str,
    base_pred_col: str,
    train_mask: np.ndarray,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, RecalibrationParams]:
    y_train = df.loc[train_mask, outcome_col].to_numpy()
    p_train = df.loc[train_mask, base_pred_col].to_numpy()

    # Need both classes in train
    if len(np.unique(y_train)) < 2:
        raise ValueError("Global recalibration: train set has single-class outcome.")

    params = fit_logistic_recalibration(y_train, p_train, eps=eps)
    p_all = df[base_pred_col].to_numpy()
    p_recal = apply_logistic_recalibration(p_all, params, eps=eps)
    return p_recal, params


def subgroup_recalibration(
    df: pd.DataFrame,
    outcome_col: str,
    base_pred_col: str,
    train_mask: np.ndarray,
    group_col: str,
    min_train_n: int,
    min_train_pos: int,
    min_train_neg: int,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, Dict[str, RecalibrationParams]]:
    """
    Fit separate recalibration params per subgroup using subgroup members in TRAIN only,
    then apply to subgroup members across the full dataset.

    Returns:
      p_subrecal_all: np.ndarray of recalibrated probabilities (NaN where subgroup not fitted)
      params_by_group_value: dict mapping subgroup label -> params
    """
    p_out = np.full(shape=(len(df),), fill_value=np.nan, dtype=float)
    params: Dict[str, RecalibrationParams] = {}

    # Work with string labels, ignore Unknown/nan
    values = df[group_col].dropna().astype(str).unique().tolist()
    values = [v for v in values if v.lower() not in ("unknown", "nan")]

    for v in values:
        mask_all = (df[group_col].astype(str) == v).to_numpy()
        mask_train = mask_all & train_mask

        n_train = int(np.sum(mask_train))
        if n_train < min_train_n:
            continue

        y_tr = df.loc[mask_train, outcome_col].to_numpy()
        p_tr = df.loc[mask_train, base_pred_col].to_numpy()

        n_pos = int(np.sum(y_tr == 1))
        n_neg = int(np.sum(y_tr == 0))
        if n_pos < min_train_pos or n_neg < min_train_neg:
            continue

        prm = fit_logistic_recalibration(y_tr, p_tr, eps=eps)
        params[v] = prm

        p_all = df.loc[mask_all, base_pred_col].to_numpy()
        p_out[mask_all] = apply_logistic_recalibration(p_all, prm, eps=eps)

    return p_out, params
