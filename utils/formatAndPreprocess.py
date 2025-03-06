# import the necessary libraries
from multiprocessing import Manager, Value
import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed


pattern_encoding_old = {'Double Top, Adam and Adam': 0, 'Triangle, symmetrical': 1, 'Double Bottom, Eve and Adam': 2, 'Head-and-shoulders top': 3, 'Double Bottom, Adam and Adam': 4, 'Head-and-shoulders bottom': 5, 'Flag, high and tight': 6, 'Cup with handle': 7}
pattern_encoding = {'Double Top, Adam and Adam': 0, 'Triangle, symmetrical': 1, 'Double Bottom, Eve and Adam': 2, 'Head-and-shoulders top': 3, 'Double Bottom, Adam and Adam': 4, 'Head-and-shoulders bottom': 5, 'Flag, high and tight': 6, 'Cup with handle': 7 ,'No Pattern':8}
def get_pattern_encoding_old():
    return pattern_encoding_old
def get_reverse_pattern_encoding_old():
    return {v: k for k, v in pattern_encoding_old.items()}
def get_pattern_encoding():
    return pattern_encoding
def get_reverse_pattern_encoding():
    return {v: k for k, v in pattern_encoding.items()}

def normalize_dataset(dataset):
    # calculate the min values from Low column and max values from High column for each instance
    min_low = dataset.groupby(level='Instance')['Low'].transform('min')
    max_high = dataset.groupby(level='Instance')['High'].transform('max')
    
    # OHLC columns to normalize
    ohlc_columns = ['Open', 'High', 'Low', 'Close']
    
    dataset_normalized = dataset.copy()
    
    # Apply the normalization formula to all columns in one go
    dataset_normalized[ohlc_columns] = (dataset_normalized[ohlc_columns] - min_low.values[:, None]) / (max_high.values[:, None] - min_low.values[:, None])
    
    # if there is a Volume column normalize it
    if 'Volume' in dataset.columns:
        # calculate the min values from Volume column and max values from Volume column for each instance
        min_volume = dataset.groupby(level='Instance')['Volume'].transform('min')
        max_volume = dataset.groupby(level='Instance')['Volume'].transform('max')
        
        # Normalize the Volume column
        dataset_normalized['Volume'] = (dataset_normalized['Volume'] - min_volume.values) / (max_volume.values - min_volume)
    
    
    return dataset_normalized    

def normalize_ohlc_segment(dataset):
    # calculate the min values from Low column and max values from High column for each instance
    min_low = dataset['Low'].min()
    max_high = dataset['High'].max()
    
    # OHLC columns to normalize
    ohlc_columns = ['Open', 'High', 'Low', 'Close']
    
    dataset_normalized = dataset.copy()
    
    if (max_high - min_low) != 0:
        # Apply the normalization formula to all columns in one go
        dataset_normalized[ohlc_columns] = (dataset_normalized[ohlc_columns] - min_low) / (max_high - min_low)
    else :
        print("Error: Max high and min low are equal")
    
    # if there is a Volume column normalize it
    if 'Volume' in dataset.columns:
        # calculate the min values from Volume column and max values from Volume column for each instance
        min_volume = dataset['Volume'].min()
        max_volume = dataset['Volume'].max()
        
        if (max_volume - min_volume) != 0:
            # Normalize the Volume column
            dataset_normalized['Volume'] = (dataset_normalized['Volume'] - min_volume) / (max_volume - min_volume)
        else:
            print("Error: Max volume and min volume are equal")
    
    
    return dataset_normalized
   
def process_row_improved(idx, row, ohlc_df, instance_counter, lock, successful_instances, instance_index_mapping):
    try:
        # Extract info and filter data
        start_date = pd.to_datetime(row['Start'])
        end_date = pd.to_datetime(row['End'])
        
        symbol_df_filtered = ohlc_df[(ohlc_df['Date'] >= start_date) & 
                                    (ohlc_df['Date'] <= end_date)]
        
        if symbol_df_filtered.empty:
            print(f"Empty result for {row['Symbol']} from {start_date} to {end_date}")
            return None
        
        # Get unique instance ID
        with lock:
            unique_instance = instance_counter.value
            instance_counter.value += 1
            
            # Explicitly add to instance_index_mapping using string key conversion
            instance_index_mapping[unique_instance] = idx
            
            # Track successful instances
            successful_instances.append(unique_instance)
        
        # Setup MultiIndex
        symbol_df_filtered = symbol_df_filtered.reset_index(drop=True)
        multi_index = pd.MultiIndex.from_arrays(
            [[unique_instance] * len(symbol_df_filtered), range(len(symbol_df_filtered))],
            names=["Instance", "Time"]
        )
        symbol_df_filtered.index = multi_index
        
        # Set index levels to proper types
        symbol_df_filtered.index = symbol_df_filtered.index.set_levels(
            symbol_df_filtered.index.levels[0].astype('int'), level=0
        )
        symbol_df_filtered.index = symbol_df_filtered.index.set_levels(
            symbol_df_filtered.index.levels[1].astype('int64'), level=1
        )
        
        # Add pattern and clean up
        symbol_df_filtered['Pattern'] = pattern_encoding[row['Chart Pattern']]
        symbol_df_filtered.drop('Date', axis=1, inplace=True)
        if 'Adj Close' in symbol_df_filtered.columns:
            symbol_df_filtered.drop('Adj Close', axis=1, inplace=True)
        
        # Normalize
        symbol_df_filtered = normalize_ohlc_segment(symbol_df_filtered)
        
        return symbol_df_filtered
    
    except Exception as e:
        print(f"Error processing {row['Symbol']}: {str(e)}")
        return None

def dataset_format(filteredPatternDf, give_instance_index_mapping=False):
    """
    Formats and preprocesses the dataset with better tracking of successful instances.
    """
    # Get symbol list from files
    folder_path = 'Datasets/OHLC data/'
    file_list = os.listdir(folder_path)
    symbol_list = [file[:-4] for file in file_list if file.endswith('.csv')]
    
    # Check for missing symbols
    symbols_in_df = filteredPatternDf['Symbol'].unique()
    missing_symbols = set(symbols_in_df) - set(symbol_list)
    if missing_symbols:
        print("Missing symbols: ", missing_symbols)
    
    # Create a list of tasks (symbol, row pairs)
    tasks = []
    for symbol in symbols_in_df:
        if symbol in symbol_list:  # Skip missing symbols
            filteredPatternDf_for_symbol = filteredPatternDf[filteredPatternDf['Symbol'] == symbol]
            file_path = os.path.join(folder_path, f"{symbol}.csv")
            
            # Pre-load symbol data
            try:
                symbol_df = pd.read_csv(file_path)
                symbol_df['Date'] = pd.to_datetime(symbol_df['Date'])
                symbol_df['Date'] = symbol_df['Date'].dt.tz_localize(None)
                
                for idx, row in filteredPatternDf_for_symbol.iterrows():
                    tasks.append((idx, row, symbol_df))
            except Exception as e:
                print(f"Error loading {symbol}: {str(e)}")
    
    print(f"Processing {len(tasks)} tasks in parallel...")
    
    # Process all tasks with instance tracking
    with Manager() as manager:
        instance_counter = manager.Value('i', 0)
        lock = manager.Lock()
        successful_instances = manager.list()  # Track which instances succeed
        instance_index_mapping = manager.dict()  # Mapping from instance ID to index
        
        results = Parallel(n_jobs=-1, verbose=1)(
            delayed(process_row_improved)(task_idx, row, df, instance_counter, lock, successful_instances, instance_index_mapping) 
            for task_idx, row, df in tasks
        )
        
        # Filter out None results
        results = [result for result in results if result is not None]
        
        print(f"Total tasks: {len(tasks)}, Successful: {len(results)}")
        print(f"Instance counter final value: {instance_counter.value}")
        print(f"Number of successful instances: {len(successful_instances)}")
        
        # Debug print for mapping
        print("Debug - Instance Index Mapping:")
        for k, v in instance_index_mapping.items():
            print(f"Key: {k}, Value: {v}")
        
        if len(successful_instances) < instance_counter.value:
            print("Warning: Some instances were assigned but their tasks failed")
        
        # Concatenate results and renumber instances if needed
        if results:
            dataset = pd.concat(results)
            dataset = dataset.sort_index(level=0)
            
            # Replace inf/nan values
            dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
            dataset.fillna(method='ffill', inplace=True)
            
            if give_instance_index_mapping:
                # Convert manager.dict to a regular dictionary
                instance_index_mapping_dict = dict(instance_index_mapping)
                
                print("Converted Mapping:", instance_index_mapping_dict)
                return dataset, instance_index_mapping_dict
            else:
                return dataset
        else:
            return pd.DataFrame()