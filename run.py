import os
import json
import joblib
import numpy as np
import pandas as pd

from NHS-TT-Baseline-Subgroup-Evaluation-and-Recalibration.config import Config, default_groups
from NHS-TT-Baseline-Subgroup-Evaluation-and-Recalibration.preprocessing import (
    load_and_filter,
    coerce_numeric,
    make_train_test_split,
    attach_split_flags,
    get_Xy,
)
from NHS-TT-Baseline-Subgroup-Evaluation-and-Recalibration.subgrouping import make_gender_intersections, subgroup_masks, list_group_values
from NHS-TT-Baseline-Subgroup-Evaluation-and-Recalibration.metrics import performance_summary
from NHS-TT-Baseline-Subgroup-Evaluation-and-Recalibration.recalibration import global_recalibration, subgroup_recalibration
from NHS-TT-Baseline-Subgroup-Evaluation-and-Recalibration.multicalibration import PMCHyperGrid, tune_and_fit_pmc


def main():
    # --------------------------
    # EDIT THESE PER OUTCOME RUN
    # --------------------------
    cfg = Config(
        seed=42,
        test_size=0.30,
        ece_bins=10,
        min_group_n_report=100,
        min_subgroup_train_n=50,
        min_subgroup_train_pos=5,
        min_subgroup_train_neg=5,

        data_path="data/analysis_table_baseline.parquet",
        model_path="models_from_paper1/phq9_reliable_improvement.joblib",
        output_dir="outputs/phq9_reliable_improvement",

        outcome_col="outcome_reliable_impPHQ9",
        predictor_columns=[ 
            # ... paste your predictor columns here ...
        ],

        group_cols=default_groups(),
        gender_col="gender",
        intersection_with=["sexual_orientation", "ethnicity", "employment"],

        # Optional: apply outcome-defined analytic samples here
        # e.g. for Recovery depression case sample:
        # filter_query="baseline_PHQ9case_yes == 1"
        filter_query=None,
    )

    os.makedirs(cfg.output_dir, exist_ok=True)

    # --------------------------
    # Load data (already cohort-filtered to Paper 3 inclusion criteria)
    # --------------------------
    df = load_and_filter(cfg.data_path, cfg.filter_query)
    df = coerce_numeric(df, cfg.predictor_columns + [cfg.outcome_col])

    # Create intersectional labels (for tables only)
    df = make_gender_intersections(df, cfg.gender_col, cfg.intersection_with)

    # --------------------------
    # Train/test split
    # --------------------------
    split = make_train_test_split(df, cfg.outcome_col, cfg.test_size, cfg.seed)
    df = attach_split_flags(df, split)
    train_mask = (df["is_global_train"] == 1).to_numpy()
    test_mask = (df["is_global_test"] == 1).to_numpy()

    # --------------------------
    # Load fixed Paper 1 model (pipeline) and generate base probabilities
    # --------------------------
    model = joblib.load(cfg.model_path)

    X_all, y_all = get_Xy(df, cfg.predictor_columns, cfg.outcome_col)
    p_base_all = model.predict_proba(X_all)[:, 1]

    df["pred_base"] = p_base_all

    # Only evaluate on TEST set finally
    y_test = df.loc[test_mask, cfg.outcome_col].to_numpy()
    p_base_test = df.loc[test_mask, "pred_base"].to_numpy()

    # --------------------------
    # (2) Global logistic recalibration (fit on TRAIN only)
    # --------------------------
    p_global_all, global_params = global_recalibration(
        df=df,
        outcome_col=cfg.outcome_col,
        base_pred_col="pred_base",
        train_mask=train_mask,
    )
    df["pred_global_recal"] = p_global_all
    p_global_test = df.loc[test_mask, "pred_global_recal"].to_numpy()

    # --------------------------
    # (3) Subgroup-specific logistic recalibration (fit on subgroup TRAIN only)
    # --------------------------
    # For each group col: create a separate set of recalibrated predictions
    subgroup_pred_cols = []

    subgroup_params_all = {}
    for g in cfg.group_cols:
        p_sub_all, params_by_val = subgroup_recalibration(
            df=df,
            outcome_col=cfg.outcome_col,
            base_pred_col="pred_base",
            train_mask=train_mask,
            group_col=g,
            min_train_n=cfg.min_subgroup_train_n,
            min_train_pos=cfg.min_subgroup_train_pos,
            min_train_neg=cfg.min_subgroup_train_neg,
        )
        col = f"pred_subgroup_recal__{g}"
        df[col] = p_sub_all
        subgroup_pred_cols.append(col)
        subgroup_params_all[g] = {k: vars(v) for k, v in params_by_val.items()}

    # Also do gender-based intersections (tables only)
    intersection_cols = [f"{cfg.gender_col}_X_{c}" for c in cfg.intersection_with]
    intersection_params_all = {}
    for g in intersection_cols:
        p_sub_all, params_by_val = subgroup_recalibration(
            df=df,
            outcome_col=cfg.outcome_col,
            base_pred_col="pred_base",
            train_mask=train_mask,
            group_col=g,
            min_train_n=cfg.min_subgroup_train_n,
            min_train_pos=cfg.min_subgroup_train_pos,
            min_train_neg=cfg.min_subgroup_train_neg,
        )
        col = f"pred_subgroup_recal__{g}"
        df[col] = p_sub_all
        intersection_params_all[g] = {k: vars(v) for k, v in params_by_val.items()}

    # --------------------------
    # (4) PMC multicalibration (tune on TRAIN only; refit on full TRAIN; apply to TEST)
    # --------------------------
    df_train = df.loc[train_mask].copy()
    df_test = df.loc[test_mask].copy()

    y_train = df_train[cfg.outcome_col].to_numpy()
    p_base_train = df_train["pred_base"].to_numpy()

    pmc_grid = PMCHyperGrid(
        alpha=(0.0001, 0.005, 0.01),
        gamma=(0.0001, 0.005, 0.01),
        eta=(0.001, 0.005, 0.01),
        max_iters=(5_000, 10_000, 25_000, 50_000, 100_000),
    )

    p_pmc_test, pmc_best, pmc_tuning = tune_and_fit_pmc(
        df_train=df_train,
        df_test=df_test,
        y_train=y_train,
        y_test=y_test,
        p_base_train=p_base_train,
        p_base_test=p_base_test,
        group_cols=cfg.group_cols + intersection_cols,  # marginal auditing across all indicators
        grid=pmc_grid,
        ece_bins=cfg.ece_bins,
        seed=cfg.seed,
    )

    df.loc[test_mask, "pred_pmc"] = p_pmc_test

    # --------------------------
    # Overall TEST-set performance (baseline vs methods)
    # --------------------------
    methods = {
        "base": p_base_test,
        "global_recal": p_global_test,
        "pmc": df.loc[test_mask, "pred_pmc"].to_numpy(),
    }

    overall_rows = []
    for name, p in methods.items():
        perf = performance_summary(y_test, p, ece_bins=cfg.ece_bins)
        overall_rows.append(
            {
                "method": name,
                "n": perf.n,
                "prevalence": perf.prevalence,
                "auc": perf.auc,
                "brier": perf.brier,
                "ece": perf.ece,
                "cal_intercept": perf.cal_intercept,
                "cal_slope": perf.cal_slope,
            }
        )
    overall_df = pd.DataFrame(overall_rows)
    overall_df.to_csv(os.path.join(cfg.output_dir, "overall_test_performance.csv"), index=False)

    # --------------------------
    # Subgroup tables (single-attribute)
    # Compare baseline vs PMC; subgroup recalibration tables included too
    # --------------------------
    subgroup_rows = []
    for g in cfg.group_cols:
        for v in list_group_values(df_test, g):
            m = (df_test[g].astype(str) == v).to_numpy()
            n_g = int(np.sum(m))
            if n_g < cfg.min_group_n_report:
                continue

            yy = y_test[m]
            p_base_g = p_base_test[m]
            p_pmc_g = df_test.loc[m, "pred_pmc"].to_numpy()

            perf_base = performance_summary(yy, p_base_g, ece_bins=cfg.ece_bins)
            perf_pmc = performance_summary(yy, p_pmc_g, ece_bins=cfg.ece_bins)

            subgroup_rows.append({"group_col": g, "group_value": v, "method": "base", **vars(perf_base)})
            subgroup_rows.append({"group_col": g, "group_value": v, "method": "pmc", **vars(perf_pmc)})

            # subgroup-specific recalibration column for this group (if fitted for that group)
            col_sub = f"pred_subgroup_recal__{g}"
            if col_sub in df.columns:
                p_sub = df_test.loc[m, col_sub].to_numpy()
                # Some groups may be NaN if recalibration not fitted for that subgroup
                if np.isfinite(p_sub).any():
                    ok = np.isfinite(p_sub)
                    perf_sub = performance_summary(yy[ok], p_sub[ok], ece_bins=cfg.ece_bins)
                    subgroup_rows.append({"group_col": g, "group_value": v, "method": "subgroup_recal", **vars(perf_sub)})

    subgroup_df = pd.DataFrame(subgroup_rows)
    subgroup_df.to_csv(os.path.join(cfg.output_dir, "subgroup_test_performance_single_attribute.csv"), index=False)

    # --------------------------
    # Intersectional tables (gender x other) — NO PLOTS (per your methods)
    # --------------------------
    inter_rows = []
    for g in intersection_cols:
        for v in list_group_values(df_test, g):
            m = (df_test[g].astype(str) == v).to_numpy()
            n_g = int(np.sum(m))
            if n_g < cfg.min_group_n_report:
                continue

            yy = y_test[m]
            p_base_g = p_base_test[m]
            p_pmc_g = df_test.loc[m, "pred_pmc"].to_numpy()

            perf_base = performance_summary(yy, p_base_g, ece_bins=cfg.ece_bins)
            perf_pmc = performance_summary(yy, p_pmc_g, ece_bins=cfg.ece_bins)

            inter_rows.append({"group_col": g, "group_value": v, "method": "base", **vars(perf_base)})
            inter_rows.append({"group_col": g, "group_value": v, "method": "pmc", **vars(perf_pmc)})

            col_sub = f"pred_subgroup_recal__{g}"
            if col_sub in df.columns:
                p_sub = df_test.loc[m, col_sub].to_numpy()
                if np.isfinite(p_sub).any():
                    ok = np.isfinite(p_sub)
                    perf_sub = performance_summary(yy[ok], p_sub[ok], ece_bins=cfg.ece_bins)
                    inter_rows.append({"group_col": g, "group_value": v, "method": "subgroup_recal", **vars(perf_sub)})

    inter_df = pd.DataFrame(inter_rows)
    inter_df.to_csv(os.path.join(cfg.output_dir, "subgroup_test_performance_intersectional.csv"), index=False)

    # --------------------------
    # Save params + tuning
    # --------------------------
    with open(os.path.join(cfg.output_dir, "global_recalibration_params.json"), "w") as f:
        json.dump({"intercept": global_params.intercept, "slope": global_params.slope}, f, indent=2)

    with open(os.path.join(cfg.output_dir, "subgroup_recalibration_params_single_attribute.json"), "w") as f:
        json.dump(subgroup_params_all, f, indent=2)

    with open(os.path.join(cfg.output_dir, "subgroup_recalibration_params_intersectional.json"), "w") as f:
        json.dump(intersection_params_all, f, indent=2)

    pmc_tuning.to_csv(os.path.join(cfg.output_dir, "pmc_train_only_tuning.csv"), index=False)
    with open(os.path.join(cfg.output_dir, "pmc_selected_params.json"), "w") as f:
        json.dump(
            {
                "alpha": pmc_best.alpha,
                "gamma": pmc_best.gamma,
                "eta": pmc_best.eta,
                "max_iters": pmc_best.max_iters,
                "val_ece": pmc_best.val_ece,
            },
            f,
            indent=2,
        )

    print("Done.")
    print("Overall performance saved to:", os.path.join(cfg.output_dir, "overall_test_performance.csv"))
    print("Single-attribute subgroup performance saved to:", os.path.join(cfg.output_dir, "subgroup_test_performance_single_attribute.csv"))
    print("Intersectional subgroup performance saved to:", os.path.join(cfg.output_dir, "subgroup_test_performance_intersectional.csv"))


if __name__ == "__main__":
    main()
