import requests
import nasdaqdatalink
import zipfile
import os
import re
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import json
from feature_derivation import main as feature_derivation

def normalize_parquet_name(file_name: str) -> str:
    base_name, ext = os.path.splitext(file_name)
    # Handles both ..._<hash32> and ..._<number>_<hash32> suffixes.
    normalized = re.sub(r'(?:_\d+)?_[0-9a-fA-F]{32}$', '', base_name)
    return f'{normalized}{ext}'


def get_trading212_instruments(
    api_key: str | None = None,
    api_secret: str | None = None,
    timeout: int = 30,
) -> pd.DataFrame:
    """Pull the list of tradable instruments from the live Trading 212 API.

    Returns one row per instrument currently offered on Trading 212, including
    the ``addedOn`` date (when the instrument was added to their universe).
    Note: the API only exposes the *current* universe + add date — it does not
    provide a delisting/removal history.

    Authentication uses the Trading 212 key *pair* (API Key + API Secret) via
    HTTP Basic Auth: the request sends ``Authorization: Basic
    base64(api_key:api_secret)``. The key must be generated from a live
    Invest / Stocks ISA account.

    Parameters
    ----------
    api_key:
        Trading 212 API Key (the "API Key ID"). If ``None``, read from
        ``api_keys.json`` under ``api_keys[0]['trading212']['key']``.
    api_secret:
        Trading 212 API Secret Key. If ``None``, read from ``api_keys.json``
        under ``api_keys[0]['trading212']['secret']``.
    timeout:
        Per-request timeout in seconds.

    Returns
    -------
    pandas.DataFrame
        Columns typically include ``ticker``, ``type``, ``isin``,
        ``currencyCode``, ``name``, ``shortName``, ``maxOpenQuantity``,
        ``workingScheduleId`` and ``addedOn`` (parsed to datetime).
    """
    if api_key is None or api_secret is None:
        with open("api_keys.json", "r", encoding="utf-8") as f:
            creds = json.load(f)['api_keys'][0]['trading212']
        api_key = api_key or creds['key']
        api_secret = api_secret or creds['secret']

    url = "https://live.trading212.com/api/v0/equity/metadata/instruments"
    response = requests.get(
        url,
        auth=(api_key, api_secret),
        timeout=timeout,
    )
    if not response.ok:
        # T212 returns a useful body on auth/permission errors; surface it so a
        # 401/403 tells you *why* (wrong account, missing scope, etc.).
        raise requests.HTTPError(
            f"{response.status_code} {response.reason} for {url}. "
            f"Response body: {response.text!r}",
            response=response,
        )

    instruments = pd.DataFrame(response.json())

    if "addedOn" in instruments.columns:
        instruments["addedOn"] = pd.to_datetime(
            instruments["addedOn"], errors="coerce"
        )

    return instruments

def reconcile_trading212_to_sharadar(
    t212_instruments: pd.DataFrame,
    sharadar_tickers: pd.DataFrame,
) -> pd.DataFrame:
    """Map Trading 212 instruments to Sharadar tickers.

    Two complementary keys are used, in priority order:

    1. **ISIN -> CUSIP** (primary, most reliable). For US/Canada securities the
       ISIN embeds the 9-character CUSIP in positions 3-11, so we extract it
       from the T212 ``isin`` and match against Sharadar's ``cusips`` column
       (which may list several space-separated CUSIPs per row). This survives
       ticker-symbol changes (e.g. SPAC de-mergers) because identifiers are
       permanent.
    2. **shortName -> ticker** (fallback). T212's ``shortName`` is the current
       exchange symbol, matched to Sharadar's ``ticker``. Used only for T212
       rows the ISIN match could not resolve.

    Parameters
    ----------
    t212_instruments:
        DataFrame from :func:`get_trading212_instruments`.
    sharadar_tickers:
        DataFrame loaded from ``SHARADAR_TICKERS.parquet``.

    Returns
    -------
    pandas.DataFrame
        One row per input T212 instrument, with the original T212 columns plus:
        ``sharadar_ticker``, ``sharadar_permaticker``, ``sharadar_isdelisted``
        and ``match_method`` (``"isin_cusip"``, ``"shortname"`` or ``None`` for
        unmatched rows, e.g. non-US instruments Sharadar does not cover).
    """
    t212 = t212_instruments.copy()

    # --- Build CUSIP -> Sharadar lookup (explode multi-CUSIP rows) ----------
    shar = sharadar_tickers.copy()
    shar_cols = ["ticker", "permaticker", "isdelisted"]
    cusip_exploded = (
        shar.assign(_cusip=shar["cusips"].astype("string").str.split())
        .explode("_cusip")
        .dropna(subset=["_cusip"])
    )
    cusip_exploded["_cusip"] = cusip_exploded["_cusip"].str.strip()
    # Keep the first (active-preferred) Sharadar row per CUSIP.
    cusip_exploded = cusip_exploded.sort_values(
        "isdelisted"  # 'N' (active) sorts before 'Y'
    ).drop_duplicates("_cusip", keep="first")
    cusip_to_shar = cusip_exploded.set_index("_cusip")[shar_cols]

    # First active-preferred Sharadar row per ticker symbol (fallback key).
    ticker_to_shar = (
        shar.sort_values("isdelisted")
        .drop_duplicates("ticker", keep="first")
        .set_index("ticker")[["permaticker", "isdelisted"]]
    )

    # --- Key 1: extract CUSIP from US/CA ISINs ------------------------------
    isin = t212["isin"].astype("string").str.strip()
    is_us_ca = isin.str.match(r"^(US|CA)[0-9A-Z]{9}[0-9]$").fillna(False)
    t212["_cusip"] = isin.where(is_us_ca).str.slice(2, 11)

    isin_match = t212["_cusip"].map(cusip_to_shar["ticker"])
    isin_perma = t212["_cusip"].map(cusip_to_shar["permaticker"])
    isin_delisted = t212["_cusip"].map(cusip_to_shar["isdelisted"])

    # --- Key 2: shortName -> ticker (only where ISIN did not resolve) -------
    short = t212["shortName"].astype("string").str.strip()
    short_match = short.where(short.isin(ticker_to_shar.index))
    short_perma = short.map(ticker_to_shar["permaticker"])
    short_delisted = short.map(ticker_to_shar["isdelisted"])

    use_short = isin_match.isna() & short_match.notna()

    t212["sharadar_ticker"] = isin_match.where(~use_short, short_match)
    t212["sharadar_permaticker"] = isin_perma.where(~use_short, short_perma)
    t212["sharadar_isdelisted"] = isin_delisted.where(~use_short, short_delisted)

    method = pd.Series(pd.NA, index=t212.index, dtype="string")
    method[isin_match.notna()] = "isin_cusip"
    method[use_short] = "shortname"
    t212["match_method"] = method

    return t212.drop(columns="_cusip")

def main():
    with open("api_keys.json", "r", encoding="utf-8") as f:
        api_keys = json.load(f)['api_keys']

    nasdaqdatalink.ApiConfig.api_key = api_keys[0]['nasdaq_data_link']

    date_str = datetime.today().strftime('%d_%m_%Y')
    output_dir = f'Data_{date_str}'
    os.makedirs(output_dir, exist_ok=True)

    tables = [
        'ACTIONS',
        'DAILY',
        'EVENTS',
        'INDICATORS',
        'METRICS',
        'SEP',
        'SF1',
        'SF2',
        'SF3',
        'SF3A',
        'SF3B',
        'SFP',
        'SP500',
        'TICKERS'
    ]

    for table in tqdm(tables):

        list_dir = [os.path.join(output_dir, x) for x in os.listdir(output_dir)]
        to_delete = [x for x in list_dir if table in x]
        for file_to_delete in to_delete:
            os.remove(file_to_delete)

        zip_file_path = os.path.join(output_dir, f"{table}.zip")
        nasdaqdatalink.export_table(f'SHARADAR/{table}', filename=zip_file_path)

        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            extracted_csvs = [
                member for member in zip_ref.namelist() if member.lower().endswith('.csv')
            ]
            zip_ref.extractall(output_dir)

        for csv_member in extracted_csvs:
            csv_path = os.path.join(output_dir, csv_member)
            if os.path.exists(csv_path):
                parquet_name = normalize_parquet_name(
                    os.path.splitext(os.path.basename(csv_path))[0] + '.parquet'
                )
                parquet_path = os.path.join(output_dir, parquet_name)
                df = pd.read_csv(csv_path)
                df.to_parquet(parquet_path, index=False)
                os.remove(csv_path)

        os.remove(zip_file_path)

    trading212_instruments = get_trading212_instruments(
        api_key=api_keys[0]['trading212']['key'],
        api_secret=api_keys[0]['trading212']['secret'],
    )
    trading212_instruments.to_csv(
        os.path.join(output_dir, 'trading212_instruments.csv'),
        index=False,
    )

    return output_dir

if __name__ == "__main__":
    data_dir = main()