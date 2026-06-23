import numpy as np
import pandas as pd
import os
import warnings
from xgboost import XGBRegressor
from typing import Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from data_extraction import reconcile_trading212_to_sharadar
from feature_derivation import get_latest_data_folder

# Default to the most recent data folder; pass an explicit `data_folder` to
# override on any function that accepts it.
DATA_FOLDER = get_latest_data_folder()

def build_predictions_and_metrics(
    model: XGBRegressor,
    X_train_fold: pd.DataFrame,
    y_train_fold: pd.Series,
    X_test_fold: pd.DataFrame,
    y_test_fold: pd.Series,
    marketcap_quantile: float = 0.25,
    fold_label: str | int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model.fit(X_train_fold, y_train_fold)
    y_pred_fold_log = model.predict(X_test_fold)

    # The model is trained on log returns; convert both predictions and targets
    # back to simple % price change so every surfaced value (and all downstream
    # metrics / stock selection) is directly comparable to raw returns.
    y_pred_fold = np.expm1(y_pred_fold_log)
    y_true_fold = np.expm1(y_test_fold.values)

    predictions_fold = pd.DataFrame(
        {
            "y_true": y_true_fold,
            "y_pred": y_pred_fold,
        },
        index=X_test_fold.index,
    )
    predictions_fold = predictions_fold.join(
        X_test_fold[["close", "close_max", "marketcap"]]
    )
    predictions_fold['marketcap_quantile'] = predictions_fold\
        .groupby('calendardate')\
            .marketcap.transform(
                lambda x: x.quantile(marketcap_quantile)
            )

    rmse = float(np.sqrt(mean_squared_error(y_true_fold, y_pred_fold)))
    mape = float(
        np.mean(
            np.abs((y_true_fold - y_pred_fold) / np.clip(np.abs(y_true_fold), 1e-8, None))
        ) * 100
    )
    metrics_fold = pd.DataFrame(
        {
            "metric": ["n_train", "n_test", "MAE", "RMSE", "R2", "MAPE_pct"],
            "value": [
                int(len(X_train_fold)),
                int(len(X_test_fold)),
                float(mean_absolute_error(y_true_fold, y_pred_fold)),
                rmse,
                float(r2_score(y_true_fold, y_pred_fold)),
                mape,
            ],
        }
    )

    if fold_label is not None:
        predictions_fold['backtest_year'] = fold_label
        metrics_fold['backtest_year'] = fold_label

    return predictions_fold, metrics_fold

def back_test(
    X_train,
    y_train,
    split_date: str = "2023-03-31",
    marketcap_quantile: float = 0.25,
    whole_back_test: bool = False,
    max_gpu_workers: int = 4,
    **kwargs
) -> pd.DataFrame:
    """
    Train an XGBoost model on historical rows and evaluate it on a holdout set.

    Splits the input features and targets by `split_date`, fits the model on
    rows before that date, and scores predictions on rows on or after it.
    When `whole_back_test` is True, the function performs a rolling yearly
    backtest from 2010 onward, using each year as the test set and all prior
    years as the training set.

    Args:
        X_train: Feature matrix indexed by `(ticker, calendardate)`.
        y_train: Target series aligned to `X_train`.
        split_date: First date to include in the backtest holdout period.
        marketcap_quantile: Quantile used to compute the cross-sectional market
            cap threshold for each calendar date in the test predictions.
        whole_back_test: Whether to run a rolling yearly backtest from 2010 to
            the latest year in the dataset instead of a single holdout split.
        max_gpu_workers: Maximum number of parallel fold-training threads when
            ``whole_back_test`` is True.  Each worker trains an independent
            model on GPU, so this should be tuned to available VRAM (default 3
            is safe for a 24 GB GPU).
        **kwargs: Additional keyword arguments forwarded to `XGBRegressor`.

    Returns:
        A tuple containing:
        - `test_predictions`: Holdout predictions with true values and selected
          market data columns.
        - `metrics_df`: Summary evaluation metrics for the holdout period or,
          for whole-dataset backtests, yearly metrics plus an overall summary.
    """

    xgboost_args = {
        "n_estimators": 600,
        "learning_rate": 0.015,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bynode": 0.8,
        "objective": "reg:squarederror",
        "random_state": 42,
        "n_jobs": -1,
        "device": "cuda"
    } | kwargs

    if whole_back_test:
        date_index = pd.to_datetime(X_train.index.get_level_values(1))
        years = sorted(year for year in date_index.year.unique() if year >= 2010)

        def _train_fold(year):
            """Train a single fold — each thread gets its own model."""
            model = XGBRegressor(**xgboost_args)
            train_mask = date_index.year < year
            test_mask = date_index.year == year

            X_train_fold = X_train[train_mask]
            X_test_fold = X_train[test_mask]
            y_train_fold = y_train[train_mask]
            y_test_fold = y_train[test_mask]

            if X_train_fold.empty or X_test_fold.empty:
                return None

            return build_predictions_and_metrics(
                X_train_fold=X_train_fold,
                y_train_fold=y_train_fold,
                X_test_fold=X_test_fold,
                y_test_fold=y_test_fold,
                fold_label=year,
                model=model,
                marketcap_quantile=marketcap_quantile,
            )

        test_prediction_folds = []
        metric_folds = []

        with ThreadPoolExecutor(max_workers=max_gpu_workers) as executor:
            futures = {executor.submit(_train_fold, y): y for y in years}
            with tqdm(total=len(years)) as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        test_prediction_folds.append(result[0])
                        metric_folds.append(result[1])
                    pbar.update(1)

        if not test_prediction_folds:
            return pd.DataFrame(), pd.DataFrame(columns=['metric', 'value', 'backtest_year'])

        test_predictions = pd.concat(test_prediction_folds).sort_values(by=['calendardate', 'y_pred'])
        metrics_df = pd.concat(metric_folds, ignore_index=True)

        overall_rmse = float(np.sqrt(mean_squared_error(test_predictions['y_true'], test_predictions['y_pred'])))
        overall_mape = float(
            np.mean(
                np.abs(
                    (test_predictions['y_true'] - test_predictions['y_pred']) /
                    np.clip(np.abs(test_predictions['y_true']), 1e-8, None)
                )
            ) * 100
        )
        overall_metrics = pd.DataFrame(
            {
                'metric': ['n_train', 'n_test', 'MAE', 'RMSE', 'R2', 'MAPE_pct'],
                'value': [
                    int(len(X_train[date_index.year < 2010])),
                    int(len(test_predictions)),
                    float(mean_absolute_error(test_predictions['y_true'], test_predictions['y_pred'])),
                    overall_rmse,
                    float(r2_score(test_predictions['y_true'], test_predictions['y_pred'])),
                    overall_mape,
                ],
                'backtest_year': ['overall'] * 6,
            }
        )
        metrics_df = pd.concat([metrics_df, overall_metrics], ignore_index=True)
        return test_predictions, metrics_df
    else:
        model = XGBRegressor(**xgboost_args)
        mask = X_train.index.get_level_values(1) >= split_date
        X_train_fold, X_test_fold = X_train[~mask], X_train[mask]
        y_train_fold, y_test_fold = y_train[~mask], y_train[mask]

        test_predictions, metrics_df = build_predictions_and_metrics(
            X_train_fold=X_train_fold,
            y_train_fold=y_train_fold,
            X_test_fold=X_test_fold,
            y_test_fold=y_test_fold,
            model=model,
            marketcap_quantile=marketcap_quantile
        )
        test_predictions = test_predictions.sort_values(by=['calendardate', 'y_pred'])
        return test_predictions, metrics_df

def fit_and_predict(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_inference: pd.DataFrame,
    marketcap_quantile: float = 0.25,
    data_folder: str = None,
    filter_to_trading212: bool = True,
    **kwargs
) -> pd.DataFrame:
    """
    Fit an XGBoost model on the full training set and score inference rows.

    Args:
        X_train: Training feature matrix.
        y_train: Training targets aligned to `X_train`.
        X_inference: Feature matrix to score after training.
        marketcap_quantile: Quantile used to compute the per-date market cap
            threshold for the inference output.
        data_folder: Optional override for the data directory containing the
            Trading 212 instruments / Sharadar tickers used by the tradability
            filter.
        filter_to_trading212: When True, restrict the predictions to stocks that
            were tradable on Trading 212 at the time of inference. Filtering is
            applied only after prediction, so the model still scores the full
            inference set and only the returned rows are limited to tradable
            stocks. Requires `trading212_instruments.csv` and
            `SHARADAR_TICKERS.parquet` in `data_folder`; if the instruments file
            is absent, filtering is skipped with a warning.
        **kwargs: Additional keyword arguments forwarded to `XGBRegressor`.

    Returns:
        A DataFrame with predicted returns, identifiers, selected market data
        columns, and the per-date market cap quantile.
    """

    if data_folder is None:
        data_folder = DATA_FOLDER

    xgboost_args = {
        "n_estimators": 1000,
        "learning_rate": 0.015,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "colsample_bynode": 0.8,
        "min_child_weight": 1,
        "gamma": 0.0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "num_parallel_tree": 10,
        "objective": "reg:squarederror",
        "random_state": 42,
        "n_jobs": -1,
        "device": "cuda"
    } | kwargs

    model = XGBRegressor(**xgboost_args)
    model.fit(X_train, y_train)
    market_cap_quantile = X_inference.groupby('calendardate').marketcap.quantile(marketcap_quantile).reset_index()
    # Model predicts log returns; convert back to simple % price change so the
    # output `y_pred` is comparable to realized returns.
    y_pred_log = model.predict(X_inference)
    inference_predictions = pd.DataFrame(np.expm1(y_pred_log), columns=['y_pred'], index=X_inference.index).reset_index()\
        .merge(X_inference.reset_index()[['ticker', 'calendardate', 'close', 'close_max', 'marketcap']])
    inference_predictions = inference_predictions.merge(market_cap_quantile.rename(columns={"marketcap":'marketcap_quantile'}))

    if filter_to_trading212:
        inference_predictions = _filter_to_trading212_tradable(
            inference_predictions, data_folder
        )

    return inference_predictions

def obtain_inference_performance_to_date(
    inference_predictions: pd.DataFrame,
    marketcap_quantile: float = 0.25,
    data_folder: str = None,
    filter_to_trading212: bool = True,
) -> pd.DataFrame:
    """
    Evaluate inference predictions against the latest available adjusted close.

    For each predicted `(ticker, calendardate)` row, the function matches the
    adjusted close at the prediction date and the adjusted close at the latest
    available price date, then computes the realized percentage change.

    Args:
        inference_predictions: Output from `fit_and_predict` containing at least
            `ticker`, `calendardate`, `y_pred`, `marketcap`, `close`, and
            `close_max`.
        marketcap_quantile: Quantile used to compute the per-date market cap
            threshold in the evaluation output.
        data_folder: Optional override for the data directory containing the
            SEP price parquet file.
        filter_to_trading212: When True, restrict the output to stocks that were
            tradable on Trading 212 at the time of inference. Each prediction is
            mapped to a Trading 212 instrument via
            `reconcile_trading212_to_sharadar`, and rows are dropped unless the
            matched instrument's `addedOn` date is on or before `calendardate`.
            Requires `trading212_instruments.csv` and `SHARADAR_TICKERS.parquet`
            in `data_folder`; if the instruments file is absent, filtering is
            skipped with a warning.

    Returns:
        A DataFrame with predicted return, realized return to the latest price
        date, selected market fields, and the per-date market cap quantile,
        sorted by realized return descending.
    """

    if data_folder is None:
        data_folder = DATA_FOLDER

    price_data = pd.read_parquet(f"{data_folder}/SHARADAR_SEP.parquet")

    test_eval = inference_predictions.rename(columns={'y_pred': 'pct_change'})[["ticker", "calendardate", "pct_change", "marketcap", "close", "close_max"]].copy()
    test_eval["ticker"] = test_eval["ticker"].astype("string")
    test_eval["calendardate"] = pd.to_datetime(test_eval["calendardate"]).astype("datetime64[ns]")

    px = price_data[["ticker", "date", "closeadj"]].copy()
    px["ticker"] = px["ticker"].astype("string")
    px["date"] = pd.to_datetime(px["date"]).astype("datetime64[ns]")

    start_px = px.rename(columns={"date": "calendardate", "closeadj": "closeadj_at_calendardate"})
    start_px["calendardate"] = start_px["calendardate"].astype("datetime64[ns]")
    start_px = start_px.sort_values(["calendardate", "ticker"]).reset_index(drop=True)

    test_eval = test_eval.sort_values(["calendardate", "ticker"]).reset_index(drop=True)
    test_eval = pd.merge_asof(
        test_eval,
        start_px,
        on="calendardate",
        by="ticker",
        direction="backward",
        tolerance=pd.Timedelta("4 days"),
    )

    ticker_max = px.groupby("ticker", as_index=False)["date"].max().rename(columns={"date": "max_price_date"})
    ticker_max["max_price_date"] = ticker_max["max_price_date"].astype("datetime64[ns]")

    end_px = px.rename(columns={"date": "max_price_date", "closeadj": "closeadj_at_max_date"})
    end_px["max_price_date"] = end_px["max_price_date"].astype("datetime64[ns]")
    end_px = end_px.sort_values(["max_price_date", "ticker"]).reset_index(drop=True)

    test_eval = test_eval.merge(ticker_max, on="ticker", how="left")
    test_eval = test_eval.dropna(subset=["max_price_date"])
    test_eval["max_price_date"] = pd.to_datetime(test_eval["max_price_date"]).astype("datetime64[ns]")
    test_eval = test_eval.sort_values(["max_price_date", "ticker"]).reset_index(drop=True)
    test_eval = pd.merge_asof(
        test_eval,
        end_px,
        on="max_price_date",
        by="ticker",
        direction="backward",
    )

    test_eval["pct_change_at_max_date"] = test_eval["closeadj_at_max_date"] / test_eval["closeadj_at_calendardate"] - 1
    test_eval['marketcap_quantile'] = test_eval.groupby('calendardate').marketcap.transform(
        lambda x: x.quantile(marketcap_quantile)
    )

    if filter_to_trading212:
        test_eval = _filter_to_trading212_tradable(test_eval, data_folder)

    return test_eval[[
        "ticker",
        "calendardate",
        "pct_change",
        "pct_change_at_max_date",
        "close",
        "close_max",
        "marketcap",
        "marketcap_quantile"
    ]].sort_values("pct_change_at_max_date", ascending=False)


def _filter_to_trading212_tradable(
    test_eval: pd.DataFrame,
    data_folder: str,
) -> pd.DataFrame:
    """Drop predictions for stocks not listed on Trading 212 at `calendardate`.

    Each Sharadar `ticker` in `test_eval` is mapped to a Trading 212 instrument
    via `reconcile_trading212_to_sharadar`, and a row is kept only if the
    matched instrument's `addedOn` date is on or before its `calendardate`
    (i.e. the stock was actually tradable on Trading 212 at inference time).
    Where several Trading 212 listings map to the same Sharadar ticker, the
    earliest `addedOn` is used.

    If `trading212_instruments.csv` is missing from `data_folder`, the input is
    returned unchanged with a warning.
    """
    instruments_path = f"{data_folder}/trading212_instruments.csv"
    if not os.path.exists(instruments_path):
        warnings.warn(
            f"'{instruments_path}' not found; skipping Trading 212 tradability "
            "filter.",
            stacklevel=2,
        )
        return test_eval

    t212 = pd.read_csv(instruments_path)
    sharadar_tickers = pd.read_parquet(f"{data_folder}/SHARADAR_TICKERS.parquet")
    reconciled = reconcile_trading212_to_sharadar(t212, sharadar_tickers)

    # Earliest Trading 212 listing date per matched Sharadar ticker.
    added_on = (
        reconciled.dropna(subset=["sharadar_ticker"])
        .assign(addedOn=lambda d: pd.to_datetime(d["addedOn"], utc=True).dt.tz_localize(None))
        .groupby("sharadar_ticker", as_index=False)["addedOn"]
        .min()
        .rename(columns={"sharadar_ticker": "ticker", "addedOn": "t212_added_on"})
    )
    added_on["ticker"] = added_on["ticker"].astype("string")

    filtered = test_eval.copy()
    filtered["ticker"] = filtered["ticker"].astype("string")
    filtered["calendardate"] = pd.to_datetime(filtered["calendardate"]).astype("datetime64[ns]")
    filtered = filtered.merge(added_on, on="ticker", how="left")
    # Keep only stocks matched to a Trading 212 listing that existed at inference.
    keep = filtered["t212_added_on"].notna() & (
        filtered["t212_added_on"] <= filtered["calendardate"]
    )
    return filtered[keep].drop(columns="t212_added_on")
