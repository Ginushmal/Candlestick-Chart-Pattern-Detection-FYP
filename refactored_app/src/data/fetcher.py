import os
import json
import subprocess
from urllib.parse import urlencode
from datetime import datetime
import pandas as pd
import yfinance as yf
from tqdm import tqdm

import logging

logger = logging.getLogger(__name__)

def download_ohlc_yfinance(symbols: list, output_dir: str = 'Datasets/OHLC data', start_date: str = '2019-01-01', end_date: str = None) -> set:
    """Downloads OHLC data for a list of symbols using yfinance."""
    os.makedirs(output_dir, exist_ok=True)
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')
        
    error_symbols = set()
    for symbol in tqdm(symbols, desc="Downloading Data", unit="symbol"):
        try:
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if data.empty:
                error_symbols.add(symbol)
                continue
                
            # Flatten column multi-index if necessary (happens in newer yfinance versions)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] for col in data.columns]
                
            data.to_csv(os.path.join(output_dir, f'{symbol}.csv'))
        except Exception as e:
            # logger.error(f"Failed to download {symbol}: {str(e)}")
            error_symbols.add(symbol)
            
    return error_symbols

def find_empty_files(directory: str = 'Datasets/OHLC data') -> list:
    """Returns a list of symbols whose CSV files are empty or missing data."""
    if not os.path.exists(directory):
        return []
        
    empty_files = []
    for file in os.listdir(directory):
        if file.endswith('.csv'):
            try:
                data = pd.read_csv(os.path.join(directory, file))
                if data.empty:
                    empty_files.append(file)
            except pd.errors.EmptyDataError:
                empty_files.append(file)
                
    return [f.replace('.csv', '').replace('.CSV', '').strip().upper() for f in empty_files]

def fetch_investing_data(stock_id: str, start_date: str = '2020-07-01', end_date: str = '2024-08-31') -> dict:
    """Fetches data from investing.com API using curl to bypass some basic blocking."""
    try:
        url = f'https://api.investing.com/api/financialdata/historical/{stock_id}'
        params = {
            'start-date': start_date,
            'end-date': end_date,
            'time-frame': 'Daily',
            'add-missing-rows': 'false'
        }

        cnfg = ['curl', '-A', 'Chrome/128.0.0.0', '-H', 'domain-id: www', '-G', url, '-d', urlencode(params)]
        output = subprocess.run(cnfg, capture_output=True).stdout.decode()
        return json.loads(output)
    except Exception as e:
        logger.error(f"Error fetching data for {stock_id}: {str(e)}")
        return None

def save_investing_data_to_csv(ticker: str, stock_data: dict, output_dir: str = "Datasets/OHLC data"):
    """Saves the JSON data from investing.com to a CSV format matching yfinance."""
    os.makedirs(output_dir, exist_ok=True)
    
    columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    data_rows = []

    for entry in stock_data.get('data', []):
        date = datetime.utcfromtimestamp(entry['rowDateRaw']).strftime('%Y-%m-%d %H:%M:%S+00:00')
        data_rows.append({
            'Date': date,
            'Open': entry['last_openRaw'],
            'High': entry['last_maxRaw'],
            'Low': entry['last_minRaw'],
            'Close': entry['last_closeRaw'],
            'Adj Close': entry['last_closeRaw'],
            'Volume': entry['volumeRaw']
        })

    df = pd.DataFrame(data_rows, columns=columns)
    file_path = os.path.join(output_dir, f"{ticker}.csv")
    df.to_csv(file_path, index=False)
    
def clean_ohlc_files(directory: str = 'Datasets/OHLC data'):
    """Filters only required columns and sorts data by date."""
    required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    
    for file in os.listdir(directory):
        if not file.endswith('.csv'):
            continue
            
        filepath = os.path.join(directory, file)
        try:
            data = pd.read_csv(filepath)
            
            # Keep only the columns that exist
            cols_to_keep = [c for c in required_columns if c in data.columns]
            data = data[cols_to_keep]
            
            if 'Date' in data.columns:
                data['Date'] = pd.to_datetime(data['Date'])
                data.sort_values('Date', inplace=True)
                data.reset_index(drop=True, inplace=True)
            
            data.to_csv(filepath, index=False)
        except Exception as e:
            logger.error(f"Error cleaning {filepath}: {e}")
