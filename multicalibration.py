from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from pmc import MultiCalibrator, Auditor

from .metrics import ece_quantile_bins


@dataclass(frozen=True)
class PMCHyperGrid:
    alpha: Tuple[float, ...] = (0.0001, 0.005, 0.01)
    gamma: Tuple[float, ...] = (0.0001, 0.005, 0.01)
    eta: Tuple[float, ...] = (0.001, 0.005, 0.01)
    max_iters: Tuple[int, ...] = (5_000, 10_000, 25_000, 50_000, 100_000)


@dataclass(frozen=True)
class PMCSelected:
    alpha: float
    gamma: float
    eta: float
    max_iters: int
    val_ece: float


def tune_and_fit_pmc(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    p_base_train: np.ndarray,
    p_base_test: np.ndarray,
    group_cols: List[str],
    grid: PMCHyperGrid,
    ece_bins: int = 10,
    seed: int = 42,
):
    """
    Train-only tuning of PMC hyperparameters.
    Uses marginal auditing across predefined group indicators.
    """

    # -------------------------
    # 80/20 split INSIDE TRAIN
    # -------------------------
    strat = y_train if len(np.unique(y_train)) == 2 else None
    idx = np.arange(len(df_train))
    idx_fit, idx_val = train_test_split(
        idx, test_size=0.20, random_state=seed, stratify=strat
    )

    # Auditor uses group columns directly
    auditor = Auditor(groups=group_cols, grouping="marginal")

    best = None
    rows = []

    for alpha in grid.alpha:
        for gamma in grid.gamma:
            for eta in grid.eta:
                for iters in grid.max_iters:

                    mc = MultiCalibrator(
                        auditor=auditor,
                        metric="PMC",
                        alpha=float(alpha),
                        gamma=float(gamma),
                        eta=float(eta),
                        max_iters=int(iters),
                        random_state=seed,
                        verbosity=0,
                    )

                    # Fit only on fit subset
                    mc.fit(
                        p_base_train[idx_fit],
                        y_train[idx_fit],
                        df_train.iloc[idx_fit][group_cols]
                    )

                    # Validate on validation subset
                    p_val_mc = mc.predict(
                        p_base_train[idx_val],
                        df_train.iloc[idx_val][group_cols]
                    )

                    val_ece = float(
                        ece_quantile_bins(y_train[idx_val], p_val_mc, n_bins=ece_bins)
                    )

                    rows.append(
                        {
                            "alpha": alpha,
                            "gamma": gamma,
                            "eta": eta,
                            "max_iters": iters,
                            "val_ece": val_ece,
                        }
                    )

                    if best is None or val_ece < best.val_ece:
                        best = PMCSelected(
                            alpha=alpha,
                            gamma=gamma,
                            eta=eta,
                            max_iters=iters,
                            val_ece=val_ece,
                        )

    if best is None:
        raise RuntimeError("PMC tuning failed.")

    # -------------------------
    # Refit on FULL TRAIN
    # -------------------------
    mc_final = MultiCalibrator(
        auditor=auditor,
        metric="PMC",
        alpha=float(best.alpha),
        gamma=float(best.gamma),
        eta=float(best.eta),
        max_iters=int(best.max_iters),
        random_state=seed,
        verbosity=0,
    )

    mc_final.fit(
        p_base_train,
        y_train,
        df_train[group_cols]
    )

    # -------------------------
    # Apply to TEST
    # -------------------------
    p_test_mc = mc_final.predict(
        p_base_test,
        df_test[group_cols]
    )

    tuning_df = pd.DataFrame(rows).sort_values("val_ece").reset_index(drop=True)

    return np.asarray(p_test_mc), best, tuning_df
