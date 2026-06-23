import numpy as np
import pandas as pd
import optuna
import logging
import json
import csv
import os
from datetime import datetime
from model_inference import back_test
from feature_derivation import get_latest_data_folder

# Default to the most recent data folder; pass an explicit `data_folder` to
# `run_hyperparameter_tuning` to override.
DATA_FOLDER = get_latest_data_folder()

# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_distribution_metric(
    preds: pd.DataFrame,
    top_n: int = 20,
    weights: dict | None = None,
) -> tuple[float, pd.DataFrame]:
    """
    From backtest predictions, filter, pick the top-N stocks per calendar
    date, compute distribution stats per date, and return a composite score
    that rewards a right-shifted distribution without chasing outliers.

    Parameters
    ----------
    preds : pd.DataFrame
        Output of ``back_test()`` – must contain columns ``y_true``,
        ``y_pred``, ``close``, ``close_max``, ``marketcap``,
        ``marketcap_quantile`` and a ``calendardate`` level in the index.
    top_n : int
        Number of top-predicted stocks to select per calendar date.
    weights : dict, optional
        Custom percentile weights.  Keys must be a subset of the columns
        returned by ``pd.Series.describe()`` (e.g. ``'25%'``, ``'50%'``).
        Defaults to ``{'25%': 0.35, '50%': 0.40, '75%': 0.25}``.

    Returns
    -------
    score : float
        The composite metric (higher is better).
    monthly_stats : pd.DataFrame
        Per-calendar-date ``describe()`` output for the selected stocks.
    portfolio_per_date : pd.Series
        Mean return per calendar date (matches notebook output).
    """
    if weights is None:
        weights = {"25%": 0.35, "50%": 0.40, "75%": 0.25}

    # Filter: close not too far from all-time high & market cap above quantile
    filtered = preds[
        (preds.close >= 0.25 * preds.close_max)
        & (preds.marketcap >= preds.marketcap_quantile)
    ]

    if filtered.empty:
        return -999.0, pd.DataFrame()

    # Top N per calendar date (data already sorted by y_pred ascending, so
    # .tail(top_n) grabs the highest predicted returns)
    top_picks = filtered.groupby("calendardate").tail(top_n)

    if top_picks.empty:
        return -999.0, pd.DataFrame(), pd.Series(dtype=float)

    # Stock-level: describe within each date, then average across dates
    # (used for scoring — rewards consistent ranking)
    monthly_stats = top_picks.groupby("calendardate")["y_true"].describe()

    score = sum(
        w * monthly_stats[col].mean()
        for col, w in weights.items()
        if col in monthly_stats.columns
    )

    # Portfolio-level: mean return per date (matches notebook output)
    portfolio_per_date = top_picks.groupby("calendardate")["y_true"].mean()

    return float(score), monthly_stats, portfolio_per_date


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def _make_objective(X_train, y_train, top_n, weights, trial_log_path, predictions_dir, max_gpu_workers):
    """Return an Optuna objective closure over the training data."""

    fieldnames = [
        "trial", "datetime", "score",
        "n_estimators", "learning_rate", "max_depth", "subsample",
        "colsample_bytree", "colsample_bynode", "min_child_weight", "gamma",
        "reg_alpha", "reg_lambda", "num_parallel_tree",
        "p25_mean", "p50_mean", "p75_mean",
        "n_dates", "mean_mean", "std_mean",
        "port_p25", "port_p50", "port_p75", "port_mean", "port_std",
    ]

    # Initialise CSV log (write header if new)
    if not os.path.exists(trial_log_path):
        with open(trial_log_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    def objective(trial: optuna.trial.Trial) -> float:
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 1500),
            "learning_rate":    trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "max_depth":        trial.suggest_int("max_depth", 3, 12),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "colsample_bynode": trial.suggest_float("colsample_bynode", 0.3, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "num_parallel_tree": trial.suggest_int("num_parallel_tree", 1, 20),
        }

        logger = logging.getLogger("hptuning")
        logger.info(
            "Trial %d | params: %s",
            trial.number,
            json.dumps({k: round(v, 6) if isinstance(v, float) else v
                        for k, v in params.items()}),
        )

        preds, _ = back_test(
            X_train, y_train,
            whole_back_test=True,
            marketcap_quantile=0.25,
            max_gpu_workers=max_gpu_workers,
            **params,
        )

        # Save full predictions for this trial
        preds_path = os.path.join(predictions_dir, f"trial_{trial.number:04d}_predictions.parquet")
        preds.to_parquet(preds_path)
        logger.info("Trial %d | predictions saved → %s", trial.number, preds_path)

        score, monthly_stats, portfolio_per_date = compute_distribution_metric(
            preds, top_n=top_n, weights=weights,
        )

        # Collect summary numbers for the log
        # Stock-level stats (used for scoring)
        if not monthly_stats.empty:
            p25 = monthly_stats["25%"].mean()
            p50 = monthly_stats["50%"].mean()
            p75 = monthly_stats["75%"].mean()
            mean_val = monthly_stats["mean"].mean()
            std_val = monthly_stats["std"].mean()
            n_dates = len(monthly_stats)
        else:
            p25 = p50 = p75 = mean_val = std_val = float("nan")
            n_dates = 0

        # Portfolio-level stats (notebook-style: describe of per-date means)
        if not portfolio_per_date.empty:
            port_desc = portfolio_per_date.describe()
            port_p25 = port_desc["25%"]
            port_p50 = port_desc["50%"]
            port_p75 = port_desc["75%"]
            port_mean = port_desc["mean"]
            port_std = port_desc["std"]
        else:
            port_p25 = port_p50 = port_p75 = port_mean = port_std = float("nan")

        logger.info(
            "Trial %d | score=%.6f  p25=%.4f  p50=%.4f  p75=%.4f  "
            "mean=%.4f  std=%.4f  dates=%d",
            trial.number, score, p25, p50, p75,
            mean_val, std_val, n_dates,
        )
        logger.info(
            "Trial %d | portfolio: p25=%.4f  p50=%.4f  p75=%.4f  "
            "mean=%.4f  std=%.4f",
            trial.number, port_p25, port_p50, port_p75,
            port_mean, port_std,
        )

        # Append to CSV
        row = {
            "trial": trial.number,
            "datetime": datetime.now().isoformat(),
            "score": round(score, 6),
            **{k: round(v, 6) if isinstance(v, float) else v
               for k, v in params.items()},
            "p25_mean": round(p25, 6),
            "p50_mean": round(p50, 6),
            "p75_mean": round(p75, 6),
            "n_dates": n_dates,
            "mean_mean": round(mean_val, 6),
            "std_mean": round(std_val, 6),
            "port_p25": round(port_p25, 6),
            "port_p50": round(port_p50, 6),
            "port_p75": round(port_p75, 6),
            "port_mean": round(port_mean, 6),
            "port_std": round(port_std, 6),
        }
        with open(trial_log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

        return score

    return objective


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def run_hyperparameter_tuning(
    X_train: pd.DataFrame | None = None,
    y_train: pd.Series | None = None,
    n_trials: int = 50,
    top_n: int = 20,
    weights: dict | None = None,
    study_name: str = "xgb_portfolio_opt",
    log_dir: str | None = None,
    data_folder: str | None = None,
    n_startup_trials: int = 10,
    max_gpu_workers: int = 1,
) -> optuna.study.Study:
    """
    Run Bayesian hyperparameter optimisation for the XGBoost portfolio model.

    The optimiser maximises a weighted-percentile composite score across
    calendar-date cohorts of the top-N predicted stocks from a full
    rolling backtest.  The default weights (p25×0.35 + p50×0.40 + p75×0.25)
    are chosen to shift the *entire* return distribution rightward rather
    than chasing mean/outlier gains.

    Parameters
    ----------
    X_train, y_train : DataFrame / Series, optional
        If not supplied they are loaded from the preprocessed parquet files.
    n_trials : int
        Number of Bayesian optimisation trials.
    top_n : int
        How many top-predicted stocks to pick per calendar date.
    weights : dict, optional
        Percentile weights for the composite score.
    study_name : str
        Optuna study name (also used for file naming).
    log_dir : str, optional
        Directory for log files; defaults to ``<DATA_FOLDER>/results``.
    data_folder : str, optional
        Data folder override.
    n_startup_trials : int
        Number of initial random-exploration trials before TPE takes over.
        ``TPESampler`` samples these uniformly at random to seed the search
        space, then switches to model-guided sampling for the remaining
        ``n_trials - n_startup_trials`` trials (the standard random-then-TPE
        pattern).
    max_gpu_workers : int
        Number of backtest folds trained concurrently inside each trial's
        ``back_test`` call.  Each fold materialises its own full copy of the
        training/test data, so higher values multiply peak RAM (and share the
        GPU).  Defaults to 1 (sequential folds) to keep memory bounded; raise
        only if you have headroom.

    Returns
    -------
    study : optuna.study.Study
        The completed Optuna study.  Access ``study.best_params``,
        ``study.best_value``, and ``study.trials_dataframe()`` for results.

    Notes
    -----
    You can interrupt the loop at any time with Ctrl-C / KeyboardInterrupt.
    All completed trials are preserved in the study and in the CSV log.
    """
    if data_folder is None:
        data_folder = DATA_FOLDER
    if log_dir is None:
        log_dir = f"{data_folder}/results"
    os.makedirs(log_dir, exist_ok=True)

    # ---- logging setup ----------------------------------------------------
    logger = logging.getLogger("hptuning")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{study_name}_{ts}.log")
    trial_csv = os.path.join(log_dir, f"{study_name}_{ts}_trials.csv")
    predictions_dir = os.path.join(log_dir, f"{study_name}_{ts}_predictions")
    os.makedirs(predictions_dir, exist_ok=True)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)

    # ---- load data --------------------------------------------------------
    if X_train is None or y_train is None:
        logger.info("Loading data from %s/preprocessed_data …", data_folder)
        X_train = pd.read_parquet(f"{data_folder}/preprocessed_data/X_train.parquet")
        y_train = pd.read_parquet(f"{data_folder}/preprocessed_data/y_train.parquet")["close_log_return"]

    logger.info("X_train shape: %s | y_train length: %d", X_train.shape, len(y_train))
    logger.info("top_n=%d | weights=%s", top_n, weights or {"25%": 0.35, "50%": 0.40, "75%": 0.25})
    logger.info("Trials CSV    → %s", trial_csv)
    logger.info("Predictions   → %s", predictions_dir)
    logger.info("Full log      → %s", log_file)

    # ---- optuna study -----------------------------------------------------
    # Standard random-then-TPE search: the first ``n_startup_trials`` trials are
    # sampled uniformly at random to explore the space, after which TPE switches
    # to model-guided sampling using those observations.
    sampler = optuna.samplers.TPESampler(
        seed=42,
        n_startup_trials=n_startup_trials,
        multivariate=True,     # model parameter correlations jointly
        constant_liar=True,    # avoid clustering while trials are in-flight
    )
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
    )

    logger.info(
        "TPE search: %d random exploration trials first, then TPE for the "
        "remaining %d trials.",
        min(n_startup_trials, n_trials),
        max(0, n_trials - n_startup_trials),
    )

    objective = _make_objective(X_train, y_train, top_n, weights, trial_csv, predictions_dir, max_gpu_workers)

    try:
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    except KeyboardInterrupt:
        logger.info("Optimisation interrupted by user after %d trials.", len(study.trials))

    # ---- summary ----------------------------------------------------------
    logger.info("=" * 70)
    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    logger.info("Optimisation complete – %d trials finished.", len(completed))
    if completed:
        logger.info("Best score : %.6f  (trial %d)", study.best_value, study.best_trial.number)
        logger.info("Best params: %s", json.dumps(
            {k: round(v, 6) if isinstance(v, float) else v
             for k, v in study.best_params.items()},
            indent=2,
        ))
    else:
        logger.info("No trials completed successfully.")
    logger.info("Trials CSV    → %s", trial_csv)
    logger.info("Predictions   → %s", predictions_dir)
    logger.info("Full log      → %s", log_file)
    logger.info("=" * 70)

    return study
