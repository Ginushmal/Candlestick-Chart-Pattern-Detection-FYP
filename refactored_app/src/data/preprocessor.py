import pandas as pd
import numpy as np
import math
import os
import logging
from tqdm import tqdm
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

PATTERN_ENCODING = {
    'Double Top, Adam and Adam': 0, 
    'Triangle, symmetrical': 1, 
    'Double Bottom, Eve and Adam': 2, 
    'Head-and-shoulders top': 3, 
    'Double Bottom, Adam and Adam': 4, 
    'Head-and-shoulders bottom': 5, 
    'Flag, high and tight': 6, 
    'Cup with handle': 7
}

def width_augment_patterns(pattern_df: pd.DataFrame, ohlc_dir: str = 'Datasets/OHLC data') -> pd.DataFrame:
    """Augments the pattern dataset by creating randomly padded variations of the patterns."""
    augmented_df = pd.DataFrame(columns=pattern_df.columns)
    pattern_counts = pattern_df['Chart Pattern'].value_counts()
    max_count = pattern_counts.max()
    
    for index, row in tqdm(pattern_df.iterrows(), total=len(pattern_df), desc="Width Augmentation"):
        symbol = row['Symbol']
        start_date = pd.to_datetime(row['Start']).tz_localize(None)
        end_date = pd.to_datetime(row['End']).tz_localize(None)
        pattern = row['Chart Pattern']
        
        ohlc_path = os.path.join(ohlc_dir, f'{symbol}.csv')
        if not os.path.exists(ohlc_path):
            continue
            
        ohlc_df = pd.read_csv(ohlc_path)
        ohlc_df['Date'] = pd.to_datetime(ohlc_df['Date']).dt.tz_localize(None)
        
        ohlc_of_interest = ohlc_df[(ohlc_df['Date'] >= start_date) & (ohlc_df['Date'] <= end_date)]
        data_size = len(ohlc_of_interest)
        
        if data_size <= 0:
            continue
            
        start_index = ohlc_of_interest.index[0]
        end_index = ohlc_of_interest.index[-1]
        
        min_possible_index = 0
        max_possible_index = len(ohlc_df) - 1
        
        num_rows_pattern = pattern_counts[pattern]
        num_row_diff = (max_count - num_rows_pattern) * 2
        multiplier = math.ceil(num_row_diff / num_rows_pattern) + 2
        
        m = np.random.randint(1, multiplier) if multiplier > 1 else 1
        
        for i in range(m):
            max_aug_len = max(5, math.ceil(data_size * 0.5))
            aug_len_l = np.random.randint(1, max_aug_len + 1)
            aug_len_r = np.random.randint(1, max_aug_len + 1)
            
            start_index_aug = max(min_possible_index, start_index - aug_len_l)
            end_index_aug = min(max_possible_index, end_index + aug_len_r)
            
            start_date_aug = ohlc_df.iloc[start_index_aug]['Date']
            end_date_aug = ohlc_df.iloc[end_index_aug]['Date']
            
            new_row = row.copy()
            new_row['Start'] = start_date_aug
            new_row['End'] = end_date_aug
            augmented_df = pd.concat([augmented_df, pd.DataFrame([new_row])], ignore_index=True)
            
        # Concat the original row too
        augmented_df = pd.concat([augmented_df, pd.DataFrame([row])], ignore_index=True)
        
    return augmented_df

def normalize_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """Normalizes OHLC and Volume data per instance."""
    min_low = dataset.groupby(level='Instance')['Low'].transform('min')
    max_high = dataset.groupby(level='Instance')['High'].transform('max')
    
    ohlc_columns = ['Open', 'High', 'Low', 'Close']
    dataset_normalized = dataset.copy()
    
    range_val = max_high.values[:, None] - min_low.values[:, None]
    range_val[range_val == 0] = 1e-10  # Avoid division by zero
    
    dataset_normalized[ohlc_columns] = (dataset_normalized[ohlc_columns] - min_low.values[:, None]) / range_val
    
    if 'Volume' in dataset.columns:
        min_volume = dataset.groupby(level='Instance')['Volume'].transform('min')
        max_volume = dataset.groupby(level='Instance')['Volume'].transform('max')
        vol_range = max_volume.values - min_volume.values
        vol_range[vol_range == 0] = 1e-10
        dataset_normalized['Volume'] = (dataset_normalized['Volume'] - min_volume.values) / vol_range
        
    return dataset_normalized

def format_dataset(pattern_df: pd.DataFrame, ohlc_dir: str = 'Datasets/OHLC data') -> pd.DataFrame:
    """Formats the pattern dataframe into a MultiIndex Sktime compatible dataframe."""
    dataset_blocks = []
    instance_counter = 0
    
    for index, row in tqdm(pattern_df.iterrows(), total=len(pattern_df), desc="Formatting Dataset"):
        symbol = row['Symbol']
        start_date = pd.to_datetime(row['Start'])
        end_date = pd.to_datetime(row['End'])
        
        padding = 0 if row['Chart Pattern'] == 'Triangle, symmetrical' else int((end_date - start_date).days * 0.3)
        padded_start_date = start_date - pd.Timedelta(days=padding)
        padded_end_date = end_date + pd.Timedelta(days=padding)
        
        ohlc_path = os.path.join(ohlc_dir, f'{symbol}.csv')
        if not os.path.exists(ohlc_path):
            continue
            
        symbol_df = pd.read_csv(ohlc_path)
        symbol_df['Date'] = pd.to_datetime(symbol_df['Date']).dt.tz_localize(None)
        
        symbol_df_filtered = symbol_df[(symbol_df['Date'] >= padded_start_date) & 
                                       (symbol_df['Date'] <= padded_end_date)].copy()
        
        if symbol_df_filtered.empty:
            continue
            
        # Add MultiIndex
        symbol_df_filtered.reset_index(drop=True, inplace=True)
        time_index = range(len(symbol_df_filtered))
        multi_index = pd.MultiIndex.from_product([[instance_counter], time_index], names=['Instance', 'Time'])
        symbol_df_filtered.index = multi_index
        
        symbol_df_filtered['Pattern'] = row['Chart Pattern']
        dataset_blocks.append(symbol_df_filtered)
        instance_counter += 1
        
    if not dataset_blocks:
        return pd.DataFrame()
        
    dataset = pd.concat(dataset_blocks, axis=0)
    
    # Clean up datatypes and structures
    dataset.index = dataset.index.set_levels(dataset.index.levels[0].astype('int'), level=0)
    dataset.index = dataset.index.set_levels(dataset.index.levels[1].astype('int64'), level=1)
    
    dataset['Pattern'] = dataset['Pattern'].map(PATTERN_ENCODING)
    
    if 'Date' in dataset.columns:
        dataset.drop('Date', axis=1, inplace=True)
    if 'Adj Close' in dataset.columns:
        dataset.drop('Adj Close', axis=1, inplace=True)
        
    dataset['Volume'] = dataset['Volume'].astype('float64')
    
    dataset = normalize_dataset(dataset)
    return dataset

def split_and_save_data(dataset: pd.DataFrame, output_dir: str = 'Datasets/VanilaDataset/X-Y Splitted Data', test_size: float = 0.2, random_state: int = 6699):
    """Splits the multi-index dataset into X and y for train and test, and saves them."""
    os.makedirs(output_dir, exist_ok=True)
    
    X = dataset.drop(columns='Pattern')
    y = dataset['Pattern']
    
    # We need to drop level 1 index to do a grouped split by Instance
    y_first = y.groupby(level='Instance').first()
    
    # Perform a stratified split
    X_train_instances, X_test_instances, y_train_first, y_test_first = train_test_split(
        y_first.index, y_first, test_size=test_size, random_state=random_state, stratify=y_first
    )
    
    X_train = X.loc[X_train_instances]
    X_test = X.loc[X_test_instances]
    
    # For y, we just need the single label per instance
    y_train = y_train_first
    y_test = y_test_first
    
    # Save the splits
    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'))
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'))
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'))
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'))
    
    return X_train, X_test, y_train, y_test

from dataclasses import dataclass
from typing import Any, List, Dict

@dataclass
class Dataset:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test_cropped: pd.DataFrame
    y_test_cropped: pd.Series
    large_test_segments: pd.DataFrame
    large_test_ground_truth: list

def run_scrape(config: dict):
    csv_path = config.get('csv_path', 'Datasets/scraped_blog_tables.csv')
    scrape_cfg = config.get('scrape', {})
    start_year = scrape_cfg.get('start_year', 2020)
    end_year = scrape_cfg.get('end_year', 2024)
    months = scrape_cfg.get('months', None)
    
    logger.info(f"Running scrape for years {start_year}-{end_year}, months: {months}")
    from .scraper import scrape_pattern_tables
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    required_columns = ['Symbol', 'Chart Pattern', 'BullishBearish', 'Start', 'End', 'Industry']
    
    df = scrape_pattern_tables(start_year, end_year, required_columns, months)
    if df.empty:
        raise RuntimeError("Scraping failed or returned no data.")
        
    df.to_csv(csv_path, index=False)
    logger.info(f"Successfully scraped and saved to {csv_path}")

def run_download(config: dict):
    csv_path = config.get('csv_path', 'Datasets/scraped_blog_tables.csv')
    ohlc_dir = config.get('ohlc_dir', 'Datasets/OHLC data')
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}. Run scrape step first.")
        
    df = pd.read_csv(csv_path)
    os.makedirs(ohlc_dir, exist_ok=True)
        
    from .fetcher import download_ohlc_yfinance
    symbols_to_download = df['Symbol'].unique().tolist()
    missing_symbols = [sym for sym in symbols_to_download if not os.path.exists(os.path.join(ohlc_dir, f"{sym}.csv"))]
    
    if missing_symbols:
        logger.info(f"Downloading OHLC data for {len(missing_symbols)} missing symbols...")
        download_ohlc_yfinance(missing_symbols, output_dir=ohlc_dir)
        
        still_missing = [sym for sym in missing_symbols if not os.path.exists(os.path.join(ohlc_dir, f"{sym}.csv"))]
        if still_missing:
            logger.warning(f"Failed to download OHLC data for some symbols: {still_missing}")
    else:
        logger.info("All OHLC data is already downloaded.")

def run_preprocess(config: dict):
    csv_path = config.get('csv_path', 'Datasets/scraped_blog_tables.csv')
    ohlc_dir = config.get('ohlc_dir', 'Datasets/OHLC data')
    output_dir = config.get('output_dir', '../Datasets/VanilaDataset/X-Y Splitted Data')
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}. Run scrape step first.")
        
    df = pd.read_csv(csv_path)
    
    target_patterns = config.get('target_patterns')
    if target_patterns:
        df = df[df['Chart Pattern'].isin(target_patterns)]
        
    max_samples_per_pattern = config.get('max_samples_per_pattern')
    if max_samples_per_pattern:
        df = df.groupby('Chart Pattern').head(max_samples_per_pattern)
        
    max_total_samples = config.get('max_total_samples')
    if max_total_samples:
        df = df.head(max_total_samples)
        
    df.reset_index(drop=True, inplace=True)
    logger.info(f"Filtered to {len(df)} pattern instances.")
    
    if df.empty:
        raise ValueError("No patterns matched the config criteria.")
        
    aug_df = width_augment_patterns(df, ohlc_dir=ohlc_dir)
    dataset_df = format_dataset(aug_df, ohlc_dir=ohlc_dir)
    
    if dataset_df.empty:
        raise ValueError("Dataset is empty after formatting.")
        
    split_and_save_data(dataset_df, output_dir=output_dir, test_size=0.2)
    logger.info(f"Preprocessed and saved splits to {output_dir}")

def load_preprocessed_dataset(config: dict) -> Dataset:
    output_dir = config.get('output_dir', '../Datasets/VanilaDataset/X-Y Splitted Data')
    ohlc_dir = config.get('ohlc_dir', 'Datasets/OHLC data')
    csv_path = config.get('csv_path', 'Datasets/scraped_blog_tables.csv')
    
    req_files = ['X_train.csv', 'y_train.csv', 'X_test.csv', 'y_test.csv']
    for f in req_files:
        if not os.path.exists(os.path.join(output_dir, f)):
            raise FileNotFoundError(f"Missing {f} in {output_dir}. Run preprocess step first.")
            
    # X datasets will have MultiIndex loaded back
    X_train = pd.read_csv(os.path.join(output_dir, 'X_train.csv'), index_col=[0,1])
    X_test = pd.read_csv(os.path.join(output_dir, 'X_test.csv'), index_col=[0,1])
    y_train = pd.read_csv(os.path.join(output_dir, 'y_train.csv'), index_col=0).squeeze("columns")
    y_test = pd.read_csv(os.path.join(output_dir, 'y_test.csv'), index_col=0).squeeze("columns")
    
    first_symbol = None
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        target_patterns = config.get('target_patterns')
        if target_patterns:
            df = df[df['Chart Pattern'].isin(target_patterns)]
        first_symbol = df['Symbol'].iloc[0] if not df.empty else None

    if first_symbol and os.path.exists(os.path.join(ohlc_dir, f"{first_symbol}.csv")):
        large_test_segments = pd.read_csv(os.path.join(ohlc_dir, f"{first_symbol}.csv"))
        if 'Date' in large_test_segments.columns:
            large_test_segments['Date'] = pd.to_datetime(large_test_segments['Date']).dt.tz_localize(None)
        symbol_patterns = df[df['Symbol'] == first_symbol]
        large_test_ground_truth = symbol_patterns.to_dict('records')
    else:
        large_test_segments = pd.DataFrame()
        large_test_ground_truth = []
        
    return Dataset(
        X_train=X_train,
        y_train=y_train,
        X_test_cropped=X_test,
        y_test_cropped=y_test,
        large_test_segments=large_test_segments,
        large_test_ground_truth=large_test_ground_truth
    )

def load_and_preprocess_data(config: dict) -> Dataset:
    """Legacy backward-compatible method for 'all' step."""
    logger.info(f"Running full data pipeline with config: {config}")
    csv_path = config.get('csv_path', 'Datasets/scraped_blog_tables.csv')
    if not os.path.exists(csv_path):
        run_scrape(config)
    run_download(config)
    run_preprocess(config)
    return load_preprocessed_dataset(config)
