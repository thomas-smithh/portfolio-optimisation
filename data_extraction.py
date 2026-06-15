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

    return output_dir

if __name__ == "__main__":
    data_dir = main()