# import the necessary libraries
import pandas as pd

pattern_encoding = {'Double Top, Adam and Adam': 0, 'Triangle, symmetrical': 1, 'Double Bottom, Eve and Adam': 2, 'Head-and-shoulders top': 3, 'Double Bottom, Adam and Adam': 4, 'Head-and-shoulders bottom': 5, 'Flag, high and tight': 6, 'Cup with handle': 7}

def indexes_fix(dataset):
    print("Fixing indexes...")
    # Print the data types of two levels of the index
    print(dataset.index.get_level_values(0).dtype, dataset.index.get_level_values(1).dtype)

    # Change the data type of level 0 index to int
    dataset.index = dataset.index.set_levels(
        dataset.index.levels[0].astype('int'), level=0
    )

    # Print the data types after modification
    print(dataset.index.get_level_values(0).dtype, dataset.index.get_level_values(1).dtype)

    # Convert level 1 index to int64
    dataset.index = dataset.index.set_levels(
        dataset.index.levels[1].astype('int64'), level=1
    )


    # Print the data types after modification
    print(dataset.index.get_level_values(0).dtype, dataset.index.get_level_values(1).dtype)
    
    return dataset

def customPatternEncoding (dataset):
    print("Pattern encoding...")
    patterns = dataset['Pattern'].unique()
    # Create a dictionary that maps each unique pattern to a unique integer
    # pattern_encoding = {pattern: idx for idx, pattern in enumerate(patterns)}
    # Print the pattern encoding dictionary
    print("Pattern Encoding Dictionary: ",pattern_encoding)
    
    # Encode the 'Pattern' column using the automatically generated encoding dictionary
    dataset['Pattern'] = dataset['Pattern'].map(pattern_encoding)
    
    # Check for any NaN values in the encoded test dataset (in case there are missing patterns)
    if dataset['Pattern'].isnull().any():
        print("Warning: Some patterns in the test dataset are missing from the training dataset.")
    
    return dataset

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

def add_multi_indexes(data_section , Instance = 0):
     # Reset index to integers (from dates)
    data_section.reset_index(drop=True, inplace=True)
    
    # check if data_section is empty , if so print why
    if data_section.empty:
        # print(f"Symbol {symbol} has no data between {padded_start_date} and {padded_end_date}")
        return None
    
    # Create a MultiIndex for the data_section
    time_index = range(len(data_section))
    
    # Create the MultiIndex where the first level is the unique instance counter
    multi_index = pd.MultiIndex.from_product([[Instance], time_index], names=['Instance', 'Time'])
    
    # Assign the MultiIndex directly to the DataFrame
    data_section.index = multi_index
    
    return data_section
    

def data_section_format(data_section, instance=0):
    """
    Formats and preprocesses the given data section.

    Args:
        data_section (DataFrame): The data section to be formatted and preprocessed.
        instance (int, optional): The instance number. Defaults to 0.

    Returns:
        DataFrame: The formatted and preprocessed data section.
    """
    # set multi index
    data_section = add_multi_indexes(data_section, instance)
    # fix the indexes
    data_section = indexes_fix(data_section)
    # convert the volume column to float64 data type
    data_section['Volume'] = data_section['Volume'].astype('float64')
    # Drop date column
    data_section.drop('Date', axis=1, inplace=True)    
    
    # Drop the 'Adj Close' column ########################
    data_section.drop('Adj Close', axis=1, inplace=True)
    
    data_section = normalize_dataset(data_section)
    
    return data_section
    

def dataset_format(filteredPatternDf):
    """
    Formats and preprocesses the dataset for candlestick chart pattern detection.

    Args:
        filteredPatternDf (DataFrame): The filtered pattern DataFrame containing information about chart patterns.

    Returns:
        DataFrame: The formatted and preprocessed dataset.

    """
    # Create an empty DataFrame for the time series with a MultiIndex for chart patterns and integers as indexes
    Dataset = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Pattern'],
                        index=pd.MultiIndex(levels=[[], []], codes=[[], []], names=['Instance', 'Time']))

    # Initialize a counter for unique instances
    instance_counter = 0

    # Loop through the filtered dataset
    for index, row in filteredPatternDf.iterrows():
        symbol = row['Symbol']
        start_date = pd.to_datetime(row['Start'])
        end_date = pd.to_datetime(row['End'])
        padding=0
        if row['Chart Pattern'] == 'Triangle, symmetrical':
            padding = 0
        else:
            # Calculate the padding for the time range (25% of the time range length)
            padding = int((end_date - start_date).days * 0.3)
        
        # Adjust the date range to include padding
        padded_start_date = start_date - pd.Timedelta(days=padding)
        padded_end_date = end_date + pd.Timedelta(days=padding)
        
        # Read the CSV file containing the OHLC data for the symbol
        symbol_df = pd.read_csv(f'Datasets/OHLC data/{symbol}.csv')
        symbol_df['Date'] = pd.to_datetime(symbol_df['Date'])
        
        # Remove timezone information from the Date column
        symbol_df['Date'] = symbol_df['Date'].dt.tz_localize(None)
        
        # Filter the symbol DataFrame to include only the date range with padding
        symbol_df_filtered = symbol_df[(symbol_df['Date'] >= padded_start_date) & 
                                    (symbol_df['Date'] <= padded_end_date)]
        
        
        symbol_df_filtered=add_multi_indexes(symbol_df_filtered, instance_counter)
        if(symbol_df_filtered is None):
            continue
        
        # # Append the Pattern column to indicate the chart pattern
        # symbol_df_filtered['Pattern'] = row['Chart Pattern']
        if not symbol_df_filtered.empty:
            # Append the Pattern column to indicate the chart pattern
            symbol_df_filtered = symbol_df_filtered.copy()  # Create a copy
            symbol_df_filtered['Pattern'] = row['Chart Pattern']

        
            # Concatenate the filtered DataFrame to the Dataset
            Dataset = pd.concat([Dataset, symbol_df_filtered], axis=0)
        
            # Increment the instance counter for the next occurrence
            instance_counter += 1

   
    
    # fix the indexes
    Dataset=indexes_fix(Dataset)
    
    Dataset = customPatternEncoding(Dataset)
    
    
    # Final Fixes :
    # convert the volume column to float64 data type
    Dataset['Volume'] = Dataset['Volume'].astype('float64')
    print("data types /n",Dataset.dtypes)
    # Drop date column
    Dataset.drop('Date', axis=1, inplace=True)    
    
    # Drop the 'Adj Close' column ########################
    Dataset.drop('Adj Close', axis=1, inplace=True)
    
    # Display the head of the Dataset
    # print(Dataset.head())
    
    Dataset = normalize_dataset(Dataset)
    
    return Dataset
