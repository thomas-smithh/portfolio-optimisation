import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import gc
from sklearn import set_config
from ta import add_all_ta_features
from tqdm import tqdm
from joblib import Parallel, delayed
from datetime import timedelta, datetime
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics
from sklearn.ensemble import BaggingRegressor
from sklearn.decomposition import PCA
from typing import List
from pandas import Timedelta
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from xgboost import XGBRegressor
import xgboost as xgb
import seaborn as sns
import pandas as pd
from pandas.tseries.offsets import BMonthEnd, CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar

warnings.filterwarnings('ignore')  # re-apply after imports
np.seterr(all='ignore')  # suppress numpy RuntimeWarnings (e.g. divide-by-zero in ta lib)
set_config(transform_output = "pandas")

# Optional memory-profiling hooks. No-ops unless a MemoryMonitor is running
# (see run_feature_derivation.py / _memlog.py), so importing is always safe.
try:
    from _memlog import set_phase as _mem_phase, note as _mem_note
except Exception:  # pragma: no cover - profiler is optional
    def _mem_phase(name):
        pass

    def _mem_note(msg):
        pass

DATA_FOLDER = "Data_01_04_2026"

def _downcast_floats(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Downcast all float64 columns to float32 in place to roughly halve memory.

    This is loss-free for modelling purposes: XGBoost internally casts its
    inputs to float32 when building the DMatrix, so the model sees identical
    values either way. Non-float columns (ticker, datetimes, ints) are left
    untouched.

    Args:
        df: DataFrame to downcast.

    Returns:
        The same DataFrame with float64 columns converted to float32.
    """
    float_cols = df.select_dtypes(include=['float64']).columns
    if len(float_cols):
        df[float_cols] = df[float_cols].astype('float32')
    return df

def check_quarterly_stock(
    data: pd.DataFrame
) -> bool:
    """
    Determine whether a stock reports on a quarterly cadence.

    Args:
        data: Stock-level fundamentals data containing a `calendardate` column.

    Returns:
        True if every reporting interval is between 90 and 92 days,
        otherwise False.
    """

    days_dif = data.calendardate.diff().dt.days.dropna()
    return ((days_dif >= 90) & (days_dif <= 92)).all()

def get_technical_indicators(
    data: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate technical analysis indicators from price action data.

    Each ticker is processed independently so that a failure on one ticker
    (e.g. ``add_all_ta_features`` raising on a pathological price series) only
    drops that single ticker rather than discarding the entire parallel batch.
    The previous implementation wrapped a whole-batch ``groupby.apply`` in one
    ``try/except`` returning ``None``; because ``pd.concat`` silently skips
    ``None``, a single bad ticker silently removed every ticker that shared its
    batch — losing ~36% of the universe.

    Args:
        data: Price history containing `ticker`, OHLC, and `volume` columns.

    Returns:
        A DataFrame with technical indicators for every ticker whose TA
        calculation succeeds. Returns None only if no ticker in the batch
        could be processed.
    """

    frames = []
    for _, group in data.groupby('ticker', sort=False):
        try:
            frames.append(
                add_all_ta_features(
                    group, open="open", high="high", low="low",
                    close="close", volume="volume", fillna=True,
                )
            )
        except Exception:
            # Skip only this ticker; keep the rest of the batch.
            continue

    if not frames:
        return None

    # Downcast to float32 immediately so the parallel workers return (and
    # pickle back to the parent) half-size frames.
    return _downcast_floats(pd.concat(frames, ignore_index=True))

def _carry_forward_quarterly_grid(
    df: pd.DataFrame,
    value_cols: list,
    max_carry_forward_quarters: int,
) -> pd.DataFrame:
    """
    Reindex each ticker onto an unbroken quarter-end grid and forward-fill its
    fundamentals, capping how many quarters a value may be carried.

    Filing cadence is irregular (late filings, more than one filing landing in a
    single calendar quarter, slower annual 10-Ks), which otherwise leaves holes
    that make tickers drop out of and re-enter the dataset quarter-on-quarter.
    Each ticker's grid spans from its first quarter to
    ``max_carry_forward_quarters`` quarters beyond its last quarter, and the
    forward fill uses the same cap, so a value is never carried more than that
    many quarters past the filing that produced it.  Holes longer than the cap
    stay NaN and are dropped by the caller's market-cap filter.  The fill only
    propagates past -> future, so it introduces no look-ahead.

    Args:
        df: Fundamentals keyed by ``ticker`` and a quarter-end ``calendardate``.
        value_cols: Columns to carry forward.
        max_carry_forward_quarters: Maximum quarters to carry a value forward.

    Returns:
        The reindexed, forward-filled fundamentals on a continuous grid.
    """
    n = int(max_carry_forward_quarters)
    df = df.sort_values(['ticker', 'calendardate'])
    spans = df.groupby('ticker')['calendardate'].agg(['min', 'max']).reset_index()

    # Don't fabricate quarters beyond the dataset's latest quarter: trailing
    # carry-forward is bounded by the global max so we never emit future rows
    # (a delisted ticker still gets its full historical trailing fill).
    global_max_p = pd.Timestamp(df['calendardate'].max()).to_period('Q')

    tickers_out = []
    dates_out = []
    for ticker, lo, hi in spans.itertuples(index=False):
        end_p = min(pd.Timestamp(hi).to_period('Q') + n, global_max_p)
        qs = pd.period_range(
            pd.Timestamp(lo).to_period('Q'),
            end_p,
            freq='Q',
        ).to_timestamp(how='end').normalize()
        tickers_out.append(np.repeat(ticker, len(qs)))
        dates_out.append(qs.values)

    full_idx = pd.DataFrame({
        'ticker': np.concatenate(tickers_out),
        'calendardate': np.concatenate(dates_out),
    })
    full_idx['ticker'] = full_idx['ticker'].astype(df['ticker'].dtype)
    full_idx['calendardate'] = full_idx['calendardate'].astype('datetime64[ns]')

    merged = full_idx.merge(df, on=['ticker', 'calendardate'], how='left')
    merged[value_cols] = merged.groupby('ticker', sort=False)[value_cols].ffill(limit=n)
    return merged

def get_stock_fundamentals(
    market_cap_lower_limit: float = 0.25,
    max_carry_forward_quarters: int = 4,
) -> pd.DataFrame:
    """
    Load and preprocess quarterly stock fundamentals.

    Args:
        market_cap_lower_limit: Quantile threshold used to filter out stocks
            below the minimum market capitalization for each calendar date.
        max_carry_forward_quarters: Maximum number of quarters a ticker's
            last-known fundamentals are carried forward onto a continuous
            quarterly grid, giving consistent quarter-on-quarter membership.

    Returns:
        A fundamentals DataFrame filtered to quarterly filings, reindexed onto a
        continuous quarterly grid with capped carry-forward, and restricted to
        stocks above the date-specific market cap threshold.
    """
    stock_fundamentals = pd.read_parquet(f'{DATA_FOLDER}/SHARADAR_SF1.parquet')
    stock_fundamentals = stock_fundamentals[stock_fundamentals.dimension == 'ARQ'].copy()
    stock_fundamentals.datekey = pd.to_datetime(stock_fundamentals.datekey)
    stock_fundamentals.calendardate = pd.to_datetime(stock_fundamentals.calendardate)
    stock_fundamentals.reportperiod = pd.to_datetime(stock_fundamentals.reportperiod)

    stock_fundamentals['datekey_q'] = pd.to_datetime(stock_fundamentals['datekey'].dt.to_period('Q').dt.end_time.dt.date)
    stock_fundamentals["fiscalperiod"] = (
        pd.PeriodIndex(stock_fundamentals["fiscalperiod"], freq="Q")
        .to_timestamp(how="end")
        .normalize()
    )

    stock_fundamentals = stock_fundamentals.loc[stock_fundamentals.groupby(['ticker', 'datekey_q']).datekey.idxmax()]
    stock_fundamentals['time_since_reporting_period'] = (
        stock_fundamentals['datekey'] - stock_fundamentals['fiscalperiod']
    ).dt.days
    # Drop the original report-period calendardate; the filing-quarter
    # (datekey_q) becomes the canonical quarter. Keep `datekey` for now so
    # staleness can be recomputed after the carry-forward reindex below.
    stock_fundamentals = stock_fundamentals.drop(['calendardate'], axis=1)
    stock_fundamentals = stock_fundamentals.rename(columns={'datekey_q': 'calendardate'})
    stock_fundamentals = stock_fundamentals.sort_values(["ticker", "calendardate"]).reset_index(drop=True)
    cols_to_fill = [c for c in stock_fundamentals.columns if c != "ticker"]
    stock_fundamentals[cols_to_fill] = (
        stock_fundamentals.groupby("ticker")[cols_to_fill].ffill()
    )
    stock_fundamentals = stock_fundamentals[~stock_fundamentals.isna().all(axis=1)]

    # Reindex each ticker onto a continuous quarterly grid and carry its
    # last-known fundamentals forward (capped) so membership is consistent
    # quarter-on-quarter instead of dropping out and re-entering when filings
    # arrive irregularly.
    value_cols = [c for c in stock_fundamentals.columns if c not in ('ticker', 'calendardate')]
    stock_fundamentals = _carry_forward_quarterly_grid(
        stock_fundamentals, value_cols, max_carry_forward_quarters
    )

    # Recompute staleness against the (possibly carried-forward) calendar quarter
    # so filled quarters correctly reflect ageing fundamentals.
    stock_fundamentals['time_since_reported'] = (
        stock_fundamentals['calendardate'] - stock_fundamentals['datekey']
    ).dt.days

    stock_fundamentals = stock_fundamentals[stock_fundamentals.marketcap.notna() & stock_fundamentals.ticker.notna()]
    stock_fundamentals = stock_fundamentals.drop(
        [
            'datekey',
            'lastupdated', 
            'reportperiod', 
            'dimension',
            'fiscalperiod'
        ], 
        axis=1
    )
    stock_fundamentals = stock_fundamentals.sort_values(
        [
            'ticker', 
            'calendardate'
        ]
    )

    stock_fundamentals = stock_fundamentals.merge(stock_fundamentals.groupby('calendardate').marketcap.quantile(market_cap_lower_limit).to_frame('average_market_cap').reset_index())
    stock_fundamentals = stock_fundamentals[stock_fundamentals.marketcap >= stock_fundamentals.average_market_cap].drop('average_market_cap', axis=1)
    stock_fundamentals = stock_fundamentals.sort_values(['ticker', 'calendardate'])
    return stock_fundamentals

def derive_fundamental_momentum(
    stock_fundamentals: pd.DataFrame,
) -> pd.DataFrame:
    """
    Derive QoQ changes, YoY changes, and acceleration for all numeric
    fundamental columns.

    Operates per-ticker on the quarterly time series.  QoQ is the 1-period
    diff, YoY is the 4-period diff (removes seasonality), and acceleration
    is the diff-of-the-QoQ-diff (second derivative).

    For ratio/margin columns (bounded metrics like ROA, ROE, margins) the
    raw arithmetic difference is used.  For level columns (revenue, assets,
    etc.) percentage change is used so the features are scale-invariant.

    Args:
        stock_fundamentals: Output of ``get_stock_fundamentals()``, sorted
            by ``[ticker, calendardate]``.

    Returns:
        The input DataFrame with QoQ, YoY, and acceleration columns appended.
    """
    exclude_cols = {'ticker', 'calendardate', 'time_since_reported',
                    'time_since_reporting_period'}
    numeric_cols = [
        c for c in stock_fundamentals.select_dtypes(include='number').columns
        if c not in exclude_cols
    ]

    # Columns that are already ratios / percentages — use arithmetic diff
    ratio_cols = {
        'assetturnover', 'currentratio', 'de', 'divyield', 'ebitdamargin',
        'grossmargin', 'netmargin', 'payoutratio', 'pb', 'pe', 'pe1', 'ps',
        'ps1', 'roa', 'roe', 'roic', 'ros', 'evebit', 'evebitda', 'sharefactor',
    }
    level_cols = [c for c in numeric_cols if c not in ratio_cols]
    ratio_cols_present = [c for c in numeric_cols if c in ratio_cols]

    grouped = stock_fundamentals.groupby('ticker', sort=False)

    # --- QoQ (1-period) ---------------------------------------------------
    # Percentage change for level columns
    qoq_pct = grouped[level_cols].pct_change(periods=1)
    qoq_pct.columns = [f"{c}_qoq_pct" for c in level_cols]
    # Arithmetic diff for ratio columns
    qoq_diff = grouped[ratio_cols_present].diff(periods=1)
    qoq_diff.columns = [f"{c}_qoq_diff" for c in ratio_cols_present]

    # --- YoY (4-period) ---------------------------------------------------
    yoy_pct = grouped[level_cols].pct_change(periods=4)
    yoy_pct.columns = [f"{c}_yoy_pct" for c in level_cols]
    yoy_diff = grouped[ratio_cols_present].diff(periods=4)
    yoy_diff.columns = [f"{c}_yoy_diff" for c in ratio_cols_present]

    # --- Acceleration (diff of QoQ) ----------------------------------------
    # Re-group the QoQ values to take the second derivative
    qoq_pct_grouped = qoq_pct.copy()
    qoq_pct_grouped['ticker'] = stock_fundamentals['ticker'].values
    accel_pct = qoq_pct_grouped.groupby('ticker', sort=False)[
        qoq_pct.columns.tolist()
    ].diff(periods=1)
    accel_pct.columns = [c.replace('_qoq_pct', '_accel_pct') for c in accel_pct.columns]

    qoq_diff_grouped = qoq_diff.copy()
    qoq_diff_grouped['ticker'] = stock_fundamentals['ticker'].values
    accel_diff = qoq_diff_grouped.groupby('ticker', sort=False)[
        qoq_diff.columns.tolist()
    ].diff(periods=1)
    accel_diff.columns = [c.replace('_qoq_diff', '_accel_diff') for c in accel_diff.columns]

    # Clip extreme pct_change values to avoid inf / extreme outliers
    for df in [qoq_pct, yoy_pct, accel_pct]:
        df.clip(lower=-10, upper=10, inplace=True)

    result = pd.concat(
        [stock_fundamentals.reset_index(drop=True),
         qoq_pct.reset_index(drop=True),
         qoq_diff.reset_index(drop=True),
         yoy_pct.reset_index(drop=True),
         yoy_diff.reset_index(drop=True),
         accel_pct.reset_index(drop=True),
         accel_diff.reset_index(drop=True)],
        axis=1,
    )

    del qoq_pct, qoq_diff, yoy_pct, yoy_diff, accel_pct, accel_diff
    del qoq_pct_grouped, qoq_diff_grouped
    gc.collect()

    return result

def get_stock_technicals(
    stock_fundamentals: pd.DataFrame,
    tmp_dir: str,
    half_idx: int = 0,
    n_chunks: int = 16,
) -> list:
    """
    Load price history, derive ticker-level technical features, and stream the
    result to disk one ticker-chunk at a time.

    Rather than returning a single ~50 GB daily frame (561 columns over ~22M
    rows per half) and holding every chunk in a list before concatenating
    (which doubles peak memory to ~100 GB and OOM-kills the process), each
    ticker chunk's full feature set is written to its own parquet file and
    freed immediately.  This bounds peak memory to a single chunk and lets the
    caller merge each chunk from disk lazily.

    Args:
        stock_fundamentals: Fundamentals data used to limit the technical
            dataset to the relevant ticker universe.
        tmp_dir: Directory to stream per-chunk parquet files into.
        half_idx: Index of the ticker half being processed (used in filenames).
        n_chunks: Number of ticker chunks to split the aggregation into. More
            chunks => lower peak memory per chunk.

    Returns:
        A list of parquet file paths, one per ticker chunk, each containing the
        raw technical indicators plus expanding aggregate statistics.
    """

    _mem_phase(f"tech_load_h{half_idx}")
    stock_technicals = pd.read_parquet(f'{DATA_FOLDER}/SHARADAR_SEP.parquet')
    stock_technicals = stock_technicals[stock_technicals.ticker.isin(stock_fundamentals.ticker)]
    stock_technicals.drop(
        [
            'closeadj', 
            'closeunadj', 
            'lastupdated'
        ], 
        axis=1, 
        inplace=True
    )
    stock_technicals.date = pd.to_datetime(stock_technicals.date)
    stock_technicals = stock_technicals.sort_values(['ticker', 'date'])

    _mem_phase(f"tech_ta_h{half_idx}")
    technical_indicators = Parallel(n_jobs=8)(delayed(get_technical_indicators)(stock_technicals[stock_technicals.ticker.isin(x)]) for x in np.array_split(stock_technicals.ticker.unique(), 24))
    # Drop any wholly-failed batches (None) explicitly rather than relying on
    # pd.concat silently skipping them, and surface how many tickers were lost.
    technical_indicators = [t for t in technical_indicators if t is not None]
    technical_indicators = pd.concat(technical_indicators, ignore_index=True)
    n_in = stock_technicals.ticker.nunique()
    n_out = technical_indicators.ticker.nunique()
    if n_out < n_in:
        print(f"        ! technicals: {n_in - n_out} of {n_in} tickers dropped during TA computation")
    technical_indicators.rename(columns={'date': 'calendardate'}, inplace=True)
    technical_indicators = technical_indicators.reset_index(drop=True)
    # Downcast the base technical frame to float32 up front so every chunk and
    # expanding aggregate below is half the size (the main OOM driver).
    technical_indicators = _downcast_floats(technical_indicators)
    del stock_technicals
    gc.collect()

    # Process expanding aggregations in ticker chunks, streaming each chunk's
    # result straight to disk so only one chunk lives in memory at a time.
    unique_tickers = technical_indicators['ticker'].unique()
    ticker_chunks = np.array_split(unique_tickers, n_chunks)
    chunk_paths = []

    for i, chunk_tickers in enumerate(ticker_chunks):
        if len(chunk_tickers) == 0:
            continue
        print(f"       Aggregating chunk {i + 1}/{n_chunks} ({len(chunk_tickers)} tickers)...")
        _mem_phase(f"tech_agg_h{half_idx}_c{i + 1}")
        chunk = technical_indicators[technical_indicators['ticker'].isin(chunk_tickers)].copy()
        # Sort by [ticker, calendardate] so the expanding() outputs (which are
        # returned in group-sorted order) line up with the chunk['ticker']
        # reassignment below. Without this the *_mean/_median/_std columns can
        # be silently misaligned across tickers.
        chunk = chunk.sort_values(['ticker', 'calendardate'])
        chunk_indexed = chunk.set_index('calendardate')
        grouped = chunk_indexed.groupby('ticker', as_index=False)

        agg1 = grouped.cummax().reset_index()
        agg1['ticker'] = chunk['ticker'].values
        agg1 = agg1.rename(columns={x: f"{x}_max" for x in agg1.columns if x not in ['ticker', 'calendardate']})

        agg2 = grouped.cummin().reset_index()
        agg2['ticker'] = chunk['ticker'].values
        agg2 = agg2.rename(columns={x: f"{x}_min" for x in agg2.columns if x not in ['ticker', 'calendardate']})

        agg3 = grouped.expanding().mean().reset_index()
        agg3['ticker'] = chunk['ticker'].values
        agg3 = agg3.rename(columns={x: f"{x}_mean" for x in agg3.columns if x not in ['ticker', 'calendardate']})

        agg4 = grouped.expanding().median().reset_index()
        agg4['ticker'] = chunk['ticker'].values
        agg4 = agg4.rename(columns={x: f"{x}_median" for x in agg4.columns if x not in ['ticker', 'calendardate']})

        agg5 = grouped.expanding().std().reset_index()
        agg5['ticker'] = chunk['ticker'].values
        agg5 = agg5.rename(columns={x: f"{x}_std" for x in agg5.columns if x not in ['ticker', 'calendardate']})

        chunk = chunk.set_index(['ticker', 'calendardate'])
        agg1 = agg1.set_index(['ticker', 'calendardate'])
        agg2 = agg2.set_index(['ticker', 'calendardate'])
        agg3 = agg3.set_index(['ticker', 'calendardate'])
        agg4 = agg4.set_index(['ticker', 'calendardate'])
        agg5 = agg5.set_index(['ticker', 'calendardate'])

        chunk_full = pd.concat([chunk, agg1, agg2, agg3, agg4, agg5], axis=1).reset_index()
        chunk_full = _downcast_floats(chunk_full)

        path = f"{tmp_dir}/_tech_h{half_idx}_c{i}.parquet"
        chunk_full.to_parquet(path, index=False)
        chunk_paths.append(path)

        del chunk, chunk_indexed, grouped, agg1, agg2, agg3, agg4, agg5, chunk_full
        gc.collect()

    del technical_indicators
    gc.collect()

    return chunk_paths

def get_sector_fundamentals(
    ticker_data: pd.DataFrame,
    stock_fundamentals: pd.DataFrame
) -> pd.DataFrame():
    """
    Build sector-level median fundamentals time series.

    Args:
        ticker_data: Ticker metadata containing `ticker` and `sector` columns.
        stock_fundamentals: Preprocessed stock fundamentals to aggregate by
            sector and calendar date.

    Returns:
        A wide DataFrame keyed by `calendardate` where each sector's median
        fundamentals are expanded into sector-prefixed feature columns.
    """

    sector_fundamentals_median = stock_fundamentals.merge(ticker_data[['ticker', 'sector']])
    sector_fundamentals_median = sector_fundamentals_median.drop('ticker', axis=1).groupby(['sector', 'calendardate'], as_index=False).median()
    sector_fundamentals_median = sector_fundamentals_median.rename(columns={'sector': 'ticker'})
    sector_fundamentals_median = sector_fundamentals_median.sort_values(['ticker', 'calendardate'])

    sector_fundamentals_median_list = []

    for ticker in sector_fundamentals_median.ticker.unique():
        sector_specific_fundamentals = sector_fundamentals_median[sector_fundamentals_median.ticker == ticker].set_index('calendardate').drop('ticker', axis=1)
        sector_specific_fundamentals.columns = [f"{ticker}_{x}" for x in sector_specific_fundamentals.columns]
        sector_fundamentals_median_list.append(sector_specific_fundamentals)
        
    sector_fundamentals_median = pd.concat(sector_fundamentals_median_list, axis=1).reset_index()
    return sector_fundamentals_median

def derive_target_variables(
    features: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute forward one-year return targets for each ticker-date pair.

    For each `(ticker, calendardate)`, the end date is the earlier of one year
    after `calendardate` or the ticker's last available price date.

    Uses asof merges so non-trading days are matched to the most recent prior
    trading session.

    Args:
        features: Input features containing at least `ticker` and
            `calendardate` columns.

    Returns:
        A DataFrame containing start and end prices, the effective target
        horizon, and the forward percentage close return.

    Raises:
        ValueError: If the required columns are missing from the input features
            or the underlying price data.
    """

    price_movement = pd.read_parquet(f'{DATA_FOLDER}/SHARADAR_SEP.parquet')
    price_movement = price_movement[price_movement.ticker.isin(features.ticker)]
    price_movement = price_movement.drop(['lastupdated'], axis=1)
    price_movement = price_movement.sort_values(['ticker', 'date'])

    req_feature_cols = {"ticker", "calendardate"}
    req_price_cols = {"ticker", "date", "closeadj"}

    if not req_feature_cols.issubset(features.columns):
        raise ValueError(f"features must include columns: {req_feature_cols}")
    if not req_price_cols.issubset(price_movement.columns):
        raise ValueError(f"price_movement must include columns: {req_price_cols}")

    feat = features[["ticker", "calendardate"]].copy()
    px = price_movement[["ticker", "date", "closeadj"]].copy()

    # Force identical dtypes for merge keys
    feat["ticker"] = feat["ticker"].astype("string")
    px["ticker"] = px["ticker"].astype("string")
    feat["calendardate"] = pd.to_datetime(feat["calendardate"]).astype("datetime64[ns]")
    px["date"] = pd.to_datetime(px["date"]).astype("datetime64[ns]")

    feat = feat.dropna(subset=["ticker", "calendardate"])
    px = px.dropna(subset=["ticker", "date", "closeadj"])

    feat = feat.sort_values(["calendardate", "ticker"]).reset_index(drop=True)
    px = px.sort_values(["date", "ticker"]).reset_index(drop=True)

    px_start = px.rename(columns={"date": "calendardate", "closeadj": "close_start"}).copy()
    px_start["calendardate"] = px_start["calendardate"].astype("datetime64[ns]")
    px_start = px_start.sort_values(["calendardate", "ticker"]).reset_index(drop=True)

    # Start close at/just before calendardate (handles weekends/holidays)
    out = pd.merge_asof(
        feat,
        px_start,
        on="calendardate",
        by="ticker",
        direction="backward",
    )

    # End date: min(calendardate + 1Y, ticker max available date)
    ticker_max = px.groupby("ticker", as_index=False)["date"].max().rename(columns={"date": "ticker_max_date"})
    ticker_max["ticker_max_date"] = ticker_max["ticker_max_date"].astype("datetime64[ns]")

    out = out.merge(ticker_max, on="ticker", how="left")
    out["target_date"] = (out["calendardate"] + pd.DateOffset(years=1)).astype("datetime64[ns]")
    out["effective_end_date"] = out[["target_date", "ticker_max_date"]].min(axis=1).astype("datetime64[ns]")

    px_end = px.rename(columns={"date": "effective_end_date", "closeadj": "close_end"}).copy()
    px_end["effective_end_date"] = px_end["effective_end_date"].astype("datetime64[ns]")
    px_end = px_end.sort_values(["effective_end_date", "ticker"]).reset_index(drop=True)

    out = out.sort_values(["effective_end_date", "ticker"]).reset_index(drop=True)
    out = pd.merge_asof(
        out,
        px_end,
        on="effective_end_date",
        by="ticker",
        direction="backward",
    )
    del feat
    del px
    del px_start
    del px_end
    del ticker_max
    gc.collect()

    out["close_pct_change"] = out["close_end"] / out["close_start"] - 1
    # Log-return target with a one-sided floor at -99.9% so total losses
    # (delistings/bankruptcies, where pct_change == -1) stay finite instead of
    # mapping to -inf. Log compresses the extreme upper tail (e.g. +20,900%)
    # so the squared-error objective no longer chases a handful of microcap
    # outliers. Outputs are converted back to simple % change at prediction time.
    out["close_log_return"] = np.log1p(out["close_pct_change"].clip(lower=-0.999))
    out["horizon_days"] = (out["effective_end_date"] - out["calendardate"]).dt.days

    return out

def main(
    min_market_cap: float = 0,
    inference_batch_min_date: str = None,
    data_folder: str = None
):
    """
    Build training and inference datasets for the portfolio model.

    Args:
        inference_batch_min_date: Boundary date used to split historical rows
            into training and inference sets.
        min_market_cap: Quantile threshold passed into the fundamentals filter
            to remove smaller-cap stocks on each reporting date.

    Returns:
        A tuple of `(X_train, y_train, X_inference)` where `X_train` contains
        historical features, `y_train` contains the aligned target returns, and
        `X_inference` contains features for rows on or after the inference
        boundary date.
    """

    if data_folder:
        global DATA_FOLDER
        DATA_FOLDER = data_folder

    if inference_batch_min_date is None:
        base_date = pd.Timestamp(datetime.today()).normalize() - pd.DateOffset(years=1)
        quarter_start_month = ((base_date.month - 1) // 3) * 3 + 1
        current_quarter_start = pd.Timestamp(base_date.year, quarter_start_month, 1)
        current_quarter_end = current_quarter_start + pd.DateOffset(months=3) - pd.Timedelta(days=1)
        inference_batch_min_date = current_quarter_end.strftime('%Y-%m-%d')

    import os
    tmp_dir = f'{DATA_FOLDER}/_tmp_chunks'
    os.makedirs(tmp_dir, exist_ok=True)

    # ── Phase 1: Load fundamentals & sector data (shared across chunks) ──
    print("[1/11] Loading ticker data...")
    _mem_phase("load_tickers")
    ticker_data = pd.read_parquet(f'{DATA_FOLDER}/SHARADAR_TICKERS.parquet')

    print("[2/11] Building stock fundamentals...")
    _mem_phase("stock_fundamentals")
    stock_fundamentals = get_stock_fundamentals(min_market_cap)
    print(f"        -> {stock_fundamentals.shape[0]:,} rows, {stock_fundamentals.shape[1]} cols")

    print("[3/11] Building sector fundamentals...")
    _mem_phase("sector_fundamentals")
    sector_fundamentals = get_sector_fundamentals(
        ticker_data=ticker_data,
        stock_fundamentals=stock_fundamentals
    )
    print(f"        -> {sector_fundamentals.shape[0]:,} rows, {sector_fundamentals.shape[1]} cols")
    del ticker_data
    gc.collect()

    print("[4/11] Deriving fundamental momentum features...")
    _mem_phase("fundamental_momentum")
    stock_fundamentals = derive_fundamental_momentum(stock_fundamentals)
    print(f"        -> {stock_fundamentals.shape[0]:,} rows, {stock_fundamentals.shape[1]} cols")

    stock_fundamentals["calendardate"] = pd.to_datetime(stock_fundamentals["calendardate"]).astype("datetime64[ns]")
    sector_fundamentals["calendardate"] = pd.to_datetime(sector_fundamentals["calendardate"]).astype("datetime64[ns]")
    stock_fundamentals = stock_fundamentals.dropna(subset=["calendardate", "ticker"])
    sector_fundamentals = sector_fundamentals.dropna(subset=["calendardate"])
    stock_fundamentals = stock_fundamentals.sort_values(["calendardate", "ticker"]).reset_index(drop=True)
    sector_fundamentals = sector_fundamentals.sort_values("calendardate").reset_index(drop=True)

    # Split all tickers into 2 halves
    all_tickers = stock_fundamentals['ticker'].unique()
    ticker_halves = np.array_split(all_tickers, 2)

    # ── Phase 2: Process each half independently ──
    for half_idx, half_tickers in enumerate(ticker_halves):
        half_label = f"Half {half_idx + 1}/2"
        print(f"\n{'='*60}")
        print(f"  {half_label}: {len(half_tickers)} tickers")
        print(f"{'='*60}")

        fund_chunk = stock_fundamentals[stock_fundamentals['ticker'].isin(half_tickers)].copy()

        print(f"[5/11] {half_label} - Building stock technicals...")
        tech_paths = get_stock_technicals(
            fund_chunk, tmp_dir=tmp_dir, half_idx=half_idx, n_chunks=16
        )
        print(f"        -> {len(tech_paths)} technical chunk files streamed to disk")

        print(f"[6/11] {half_label} - Merging fundamentals, technicals & sector data...")
        _mem_phase(f"merge_h{half_idx}")
        merged = []
        for tpath in tqdm(tech_paths, desc=f"  {half_label} merge"):
            right = pd.read_parquet(tpath)
            right["calendardate"] = pd.to_datetime(right["calendardate"]).astype("datetime64[ns]")
            right = right.dropna(subset=["calendardate", "ticker"])
            chunk_tickers = right["ticker"].unique()

            left = fund_chunk[fund_chunk.ticker.isin(chunk_tickers)].sort_values(["calendardate", "ticker"])
            right = right.assign(_technical_match_flag=1).sort_values(["calendardate", "ticker"])

            if left.empty or right.empty:
                del left, right
                gc.collect()
                continue

            temp = pd.merge_asof(
                left, right,
                on="calendardate", by="ticker",
                direction="backward", tolerance=Timedelta("4 days"),
            )
            temp = temp[temp["_technical_match_flag"].notna()].drop(columns="_technical_match_flag")
            if temp.empty:
                del left, right, temp
                gc.collect()
                continue

            temp = pd.merge_asof(
                temp.sort_values("calendardate"),
                sector_fundamentals,
                on="calendardate", direction="backward",
                tolerance=Timedelta("4 days"),
            )
            merged.append(temp)
            del left, right, temp
            gc.collect()

        # Technical chunk files have served their purpose; reclaim the disk.
        for tpath in tech_paths:
            try:
                os.remove(tpath)
            except OSError:
                pass

        del fund_chunk
        gc.collect()

        if not merged:
            print(f"        -> {half_label}: no merged rows, skipping")
            continue

        data_chunk = pd.concat(merged, ignore_index=True)
        del merged
        gc.collect()
        print(f"        -> {data_chunk.shape[0]:,} rows, {data_chunk.shape[1]} cols")

        data_chunk = data_chunk.loc[data_chunk.groupby(['ticker', 'calendardate']).close.idxmax()].copy()
        data_chunk = data_chunk.set_index(['ticker', 'calendardate']).reset_index()

        print(f"[7/11] {half_label} - Deriving target variables...")
        _mem_phase(f"targets_h{half_idx}")
        targets_chunk = derive_target_variables(data_chunk[["ticker", "calendardate"]])

        print(f"[8/11] {half_label} - Splitting train / inference & writing to disk...")
        _mem_phase(f"split_write_h{half_idx}")
        Xy_train = data_chunk[data_chunk.calendardate < inference_batch_min_date]\
            .merge(targets_chunk[['ticker', 'calendardate', 'close_log_return']])\
            .set_index(['ticker', 'calendardate'])\
            .dropna(subset=['close_log_return'])\
            .drop_duplicates()

        Xy_inference = data_chunk[data_chunk.calendardate >= inference_batch_min_date]\
            .merge(targets_chunk[['ticker', 'calendardate', 'close_log_return']])\
            .set_index(['ticker', 'calendardate'])\
            .drop_duplicates()

        X_train_c = _downcast_floats(Xy_train.drop('close_log_return', axis=1))
        y_train_c = Xy_train.close_log_return.astype('float32')
        X_inf_c = _downcast_floats(Xy_inference.drop('close_log_return', axis=1))

        print(f"        -> X_train: {X_train_c.shape}, X_inference: {X_inf_c.shape}")

        X_train_c.to_parquet(f'{tmp_dir}/X_train_{half_idx}.parquet')
        y_train_c.to_frame().to_parquet(f'{tmp_dir}/y_train_{half_idx}.parquet')
        X_inf_c.to_parquet(f'{tmp_dir}/X_inference_{half_idx}.parquet')

        del data_chunk, targets_chunk, Xy_train, Xy_inference, X_train_c, y_train_c, X_inf_c
        gc.collect()
        print(f"        -> {half_label} written to disk and freed from memory.")

    del stock_fundamentals, sector_fundamentals
    gc.collect()

    # ── Phase 3: Consolidate both halves ──
    print(f"\n{'='*60}")
    print("[9/11] Consolidating halves from disk...")
    _mem_phase("consolidate")
    X_train = pd.concat([
        pd.read_parquet(f'{tmp_dir}/X_train_{i}.parquet') for i in range(2)
    ])
    y_train = pd.concat([
        pd.read_parquet(f'{tmp_dir}/y_train_{i}.parquet') for i in range(2)
    ]).squeeze()
    X_inference = pd.concat([
        pd.read_parquet(f'{tmp_dir}/X_inference_{i}.parquet') for i in range(2)
    ])

    print(f"[10/11] Final shapes -> X_train: {X_train.shape}, y_train: {y_train.shape}, X_inference: {X_inference.shape}")

    out_dir = f'{DATA_FOLDER}/preprocessed_data'
    os.makedirs(out_dir, exist_ok=True)
    print(f"[11/11] Saving parquets to {out_dir}/ ...")
    X_train.to_parquet(f'{out_dir}/X_train.parquet')
    y_train.to_frame().to_parquet(f'{out_dir}/y_train.parquet')
    X_inference.to_parquet(f'{out_dir}/X_inference.parquet')
    print("Done.")

    # Clean up temp files
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return X_train, y_train, X_inference

if __name__ == "__main__":
    main()