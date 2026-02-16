from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.special import logit
import statsmodels.api as sm
from sklearn.metrics import brier_score_loss, roc_auc_score


def clip_prob(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return np.clip(p, eps, 1 - eps)


def calibration_intercept_slope_logistic(
    y: np.ndarray,
    p: np.ndarray,
    eps: float = 1e-6
) -> Tuple[float, float]:
    """
    logit(P(Y=1)) = a + b * logit(p)
    """
    y = np.asarray(y, dtype=float)
    p = clip_prob(p, eps=eps)
    eta = logit(p)
    X = sm.add_constant(eta)
    fit = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    return float(fit.params[0]), float(fit.params[1])


def ece_quantile_bins(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """
    Quantile-binned Expected Calibration Error.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)

    # Quantile bins on predictions
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(p, qs)
    edges[0] = -np.inf
    edges[-1] = np.inf

    ece = 0.0
    n = len(p)
    for i in range(n_bins):
        m = (p > edges[i]) & (p <= edges[i + 1])
        if not np.any(m):
            continue
        p_mean = float(np.mean(p[m]))
        y_mean = float(np.mean(y[m]))
        w = float(np.sum(m)) / n
        ece += w * abs(p_mean - y_mean)
    return float(ece)


def reliability_table_quantile(
    y: np.ndarray,
    p: np.ndarray,
    n_bins: int = 10
) -> pd.DataFrame:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)

    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(p, qs)
    edges[0] = -np.inf
    edges[-1] = np.inf

    rows = []
    for i in range(n_bins):
        m = (p > edges[i]) & (p <= edges[i + 1])
        if not np.any(m):
            continue
        rows.append(
            {
                "bin": i + 1,
                "n": int(np.sum(m)),
                "p_mean": float(np.mean(p[m])),
                "y_mean": float(np.mean(y[m])),
                "p_min": float(np.min(p[m])),
                "p_max": float(np.max(p[m])),
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class Perf:
    n: int
    prevalence: float
    auc: float
    brier: float
    ece: float
    cal_intercept: float
    cal_slope: float


def performance_summary(y: np.ndarray, p: np.ndarray, ece_bins: int = 10) -> Perf:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)

    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
    brier = float(brier_score_loss(y, p))
    ece = float(ece_quantile_bins(y, p, n_bins=ece_bins))
    ci, cs = calibration_intercept_slope_logistic(y, p)
    return Perf(
        n=int(len(y)),
        prevalence=float(np.mean(y)),
        auc=auc,
        brier=brier,
        ece=ece,
        cal_intercept=ci,
        cal_slope=cs,
    )
