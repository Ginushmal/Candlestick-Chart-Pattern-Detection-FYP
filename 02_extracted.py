# Load csv as pandas dataframe
import pandas as pd

# Load csv as pandas dataframe
cleanedPatternDf = pd.read_csv('Datasets/scraped_blog_tables.csv')
cleanedPatternDf.head()

# --- CELL ---

cleanedPatternDf['Start'] = pd.to_datetime(cleanedPatternDf['Start'])
cleanedPatternDf['End'] = pd.to_datetime(cleanedPatternDf['End'])

# --- CELL ---

# get the csv file names exept the extension in the folder OHLC data 
import os
from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir('Datasets/OHLC data') if isfile(join('Datasets/OHLC data', f))]
onlyfiles = [f.split('.')[0] for f in onlyfiles]
print(onlyfiles)

#  now remove any row that contain a Symbol that is not in the list of csv file names from the filteredPatternDf dataframe
cleanedPatternDf = cleanedPatternDf[cleanedPatternDf['Symbol'].isin(onlyfiles)]
cleanedPatternDf.head()

# --- CELL ---

#  get all the unique values in the Chart Pattern column 
uniquePatterns=cleanedPatternDf['Chart Pattern'].value_counts()
uniquePatterns.head(30)

# --- CELL ---

filteredPatternDf = cleanedPatternDf[cleanedPatternDf['Chart Pattern'].isin(['Double Bottom, Adam and Adam', 'Triangle, symmetrical', 'Double Top, Adam and Adam', 'Double Bottom, Eve and Adam', 'Head-and-shoulders bottom', 'Head-and-shoulders top', 'Cup with handle','Flag, high and tight'])]

# print the un unique values in the Chart Pattern column
print(filteredPatternDf['Chart Pattern'].unique())

filteredPatternDf.head()

# --- CELL ---

# create a column with the difference between the end and start date of each pattern Pattern_Length
filteredPatternDf['Pattern_Length'] = filteredPatternDf['End'] - filteredPatternDf['Start']
filteredPatternDf['Pattern_Length'] = filteredPatternDf['Pattern_Length'].dt.days
filteredPatternDf

# --- CELL ---

import matplotlib.pyplot as plt
import numpy as np

# Define bin edges for 10-day intervals
min_length = filteredPatternDf['Pattern_Length'].min()
max_length = filteredPatternDf['Pattern_Length'].max()
bin_edges = np.arange(min_length, max_length + 10, 10)  # Bin edges with step size of 10

# Plot histogram with explicit bins
filteredPatternDf['Pattern_Length'].hist(bins=bin_edges, edgecolor='black')
plt.xlabel('Pattern Length (10-day bins)')
plt.ylabel('Frequency')
plt.title('Pattern Length Frequency')
plt.show()


# --- CELL ---

# draw a box plot for length of each pattern
filteredPatternDf.boxplot('Pattern_Length', by='Chart Pattern', figsize=(10, 6))
# rotate the x-axis labels 
plt.xticks(rotation=90)


# --- CELL ---

# get the number of patterns that has a Pattern_Length less than 100 days and the number of all patterns
num_short_patterns = filteredPatternDf[filteredPatternDf['Pattern_Length'] < 100].shape[0]
num_total_patterns = filteredPatternDf.shape[0]
print(f'Number of patterns with length less than 100 days: {num_short_patterns}')
print(f'Total number of patterns: {num_total_patterns}')

# --- CELL ---

print(filteredPatternDf['Chart Pattern'].value_counts())

# --- CELL ---

# number of each pattern where the Pattern_Length is less than 100 days
short_patterns = filteredPatternDf[filteredPatternDf['Pattern_Length'] < 100]
short_patterns['Chart Pattern'].value_counts()
print(short_patterns['Chart Pattern'].value_counts())

# --- CELL ---

filteredPatternDf

# --- CELL ---

# get celi of 0.8
import math
math.ceil(0.8)


# --- CELL ---

import math
from tqdm import tqdm

filteredPattern_width_aug_df = pd.DataFrame(columns=filteredPatternDf.columns)

# loop through the rows of filteredPatternDf
for index, row in tqdm(filteredPatternDf.iterrows(), total=len(filteredPatternDf), desc="Processing"):

    symbol = row['Symbol']
    start_date = row['Start']
    end_date = row['End']
    pattern = row['Chart Pattern']
    
    ohlc_df = pd.read_csv(f'Datasets/OHLC data/{symbol}.csv')
    # Ensure all datetime objects are timezone-naive
    ohlc_df['Date'] = pd.to_datetime(ohlc_df['Date']).dt.tz_localize(None)

    # Convert start_date and end_date to timezone-naive if they have a timezone
    start_date = pd.to_datetime(start_date).tz_localize(None)
    end_date = pd.to_datetime(end_date).tz_localize(None)

    ohlc_of_interest = ohlc_df[(ohlc_df['Date'] >= start_date) & (ohlc_df['Date'] <= end_date)]
    data_size = len(ohlc_of_interest)
    
    if data_size <= 0:
        continue
    
    # index of ohlc data on the start date and end date
    start_index = ohlc_of_interest.index[0]
    end_index = ohlc_of_interest.index[-1]
    
    min_possible_index = 0
    max_possible_index = len(ohlc_df) - 1
    
    number_of_rows_for_pattern= filteredPatternDf['Chart Pattern'].value_counts()[pattern]
    max_num_of_rows_for_pattern = filteredPatternDf['Chart Pattern'].value_counts().max()
    
    num_row_diff = (max_num_of_rows_for_pattern - number_of_rows_for_pattern)*2
    
    multiplier = math.ceil(num_row_diff / number_of_rows_for_pattern) +2
    # print ('Pattern :', pattern , 'Multiplier :' , multiplier , 'Number of rows for pattern :', number_of_rows_for_pattern)
    # get a random mvalue between 1 to multiplier
    m = np.random.randint(1, multiplier)
    for i in range(m):
        max_aug_len = math.ceil(data_size * 0.5)
        if max_aug_len < 5:
            max_aug_len = 5
        aug_len_l = np.random.randint(1, max_aug_len)
        aug_len_r = np.random.randint(1, max_aug_len)
        
        # get the start and end index of the augmented data
        start_index_aug = start_index - aug_len_l
        end_index_aug = end_index + aug_len_r
        
        if start_index_aug < min_possible_index:
            start_index_aug = min_possible_index
        if end_index_aug > max_possible_index:
            end_index_aug = max_possible_index
        
        # get the date of the start and end index of the augmented data
        start_date_aug = ohlc_df.iloc[start_index_aug]['Date']
        end_date_aug = ohlc_df.iloc[end_index_aug]['Date']
        
        # create a new row for the augmented data
        new_row = row.copy()
        new_row['Start'] = start_date_aug
        new_row['End'] = end_date_aug
        filteredPattern_width_aug_df = pd.concat([filteredPattern_width_aug_df, pd.DataFrame([new_row])], ignore_index=True)
        
    # concat the original row too
    filteredPattern_width_aug_df = pd.concat([filteredPattern_width_aug_df, pd.DataFrame([row])], ignore_index=True)

# --- CELL ---

filteredPatternDf

# --- CELL ---

filteredPattern_width_aug_df

# --- CELL ---

import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import matplotlib.dates as mdates

def plot_ohlc_for_pattern (pattern_data_row,actual_start_date=None,actual_end_date=None):
    symbol = pattern_data_row['Symbol']
    start_date = pattern_data_row['Start']
    end_date = pattern_data_row['End']
    ohlc_df = pd.read_csv(f'Datasets/OHLC data/{symbol}.csv')
    ohlc_df['Date'] = pd.to_datetime(ohlc_df['Date']).dt.tz_localize(None)
    start_date = pd.to_datetime(start_date).tz_localize(None)
    end_date = pd.to_datetime(end_date).tz_localize(None)
    ohlc_of_interest = ohlc_df[(ohlc_df['Date'] >= start_date) & (ohlc_df['Date'] <= end_date)]
    ohlc_for_mpf = ohlc_of_interest[['Open', 'High', 'Low', 'Close']].copy()

    ohlc_for_mpf.index = pd.to_datetime(ohlc_of_interest['Date'])

    # Re-plot with proper date formatting
    fig, axes = mpf.plot(ohlc_for_mpf, type='candle', style='charles', 
                        datetime_format='%Y-%m-%d', returnfig=True)
    ax = axes[0]
    if actual_start_date is not None and actual_end_date is not None:

        pattern_start = pd.to_datetime(actual_start_date).tz_localize(None)  # Ensure it's a datetime object
        pattern_end = pd.to_datetime(actual_end_date).tz_localize(None)  # Ensure it's a datetime object

        pattern_start_date = pd.to_datetime(actual_start_date).tz_localize(None)  # Ensure it's a datetime object
        pattern_end_date = pd.to_datetime(actual_end_date).tz_localize(None)  # Ensure it's a datetime object



        num_of_OHLC_data_points_from_seg_start_to_pattern_start = len(ohlc_of_interest[ohlc_of_interest['Date'] < pattern_start_date])

        pattern_start = num_of_OHLC_data_points_from_seg_start_to_pattern_start

        num_of_OHLC_data_points_from_pattern_start_to_pattern_end = len(ohlc_of_interest[(ohlc_of_interest['Date'] >= pattern_start_date) & (ohlc_of_interest['Date'] <= pattern_end_date)])

        pattern_end = pattern_start + num_of_OHLC_data_points_from_pattern_start_to_pattern_end

        pattern_lable = row['Chart Pattern']
        
        # Add a vertical span (highlight the pattern) to the chart
        ax.axvspan(pattern_start, pattern_end,color='red', alpha=0.2, label=pattern_lable)

# --- CELL ---

plot_ohlc_for_pattern(filteredPatternDf.iloc[0])

# --- CELL ---

plot_ohlc_for_pattern(filteredPattern_width_aug_df.iloc[0],filteredPatternDf.iloc[0]['Start'],filteredPatternDf.iloc[0]['End'])
plot_ohlc_for_pattern(filteredPattern_width_aug_df.iloc[1],filteredPatternDf.iloc[0]['Start'],filteredPatternDf.iloc[0]['End'])

# --- CELL ---

# number of rows of each pattern in the filteredPattern_width_aug_df
filteredPatternDf['Chart Pattern'].value_counts()

# --- CELL ---

# number of rows of each pattern in the filteredPattern_width_aug_df
filteredPattern_width_aug_df['Chart Pattern'].value_counts()

# --- CELL ---

# get the value of largest number of rows of a pattern in the filteredPattern_width_aug_df
filteredPattern_width_aug_df['Chart Pattern'].value_counts().max()

# --- CELL ---

#  split a 20% of each class in the Dataset Dataframe for testing  and 80% for training
from sklearn.model_selection import train_test_split

# Split the Dataset into training and testing sets
train, test = train_test_split(filteredPatternDf, test_size=0.2,random_state =6699, stratify=filteredPatternDf['Chart Pattern'])

# Display the shape of the training and testing sets
print(f'Training Set Shape: {train.shape}')
print(f'Testing Set Shape: {test.shape}')

# display the amount of data in each class in the train and test data
print("train data",train['Chart Pattern'].value_counts())
print("test data",test['Chart Pattern'].value_counts())



# --- CELL ---

test.to_csv('Datasets/VanilaDataset/test_patterns_with_symbols.csv', index=False)
train.to_csv('Datasets/VanilaDataset/train_patterns_with_symbols.csv', index=False)

# --- CELL ---

#  split a 20% of each class in the Dataset Dataframe for testing  and 80% for training
from sklearn.model_selection import train_test_split

# Split the Dataset into training and testing sets
train_w_aug, test_w_aug = train_test_split(filteredPattern_width_aug_df, test_size=0.2,random_state =6699, stratify=filteredPattern_width_aug_df['Chart Pattern'])

# Display the shape of the training and testing sets
print(f'Training Set Shape: {train_w_aug.shape}')
print(f'Testing Set Shape: {test_w_aug.shape}')

# display the amount of data in each class in the train and test data
print("train data",train_w_aug['Chart Pattern'].value_counts())
print("test data",test_w_aug['Chart Pattern'].value_counts())



# --- CELL ---

test_w_aug.to_csv('Datasets/VanilaDataset/test_w_aug_patterns_with_symbols.csv', index=False)
train_w_aug.to_csv('Datasets/VanilaDataset/train_w_aug_patterns_with_symbols.csv', index=False)

# --- CELL ---

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
        
        # Filter the symbol DataFrame to include only the date range with padding
        symbol_df_filtered = symbol_df[(symbol_df['Date'] >= padded_start_date) & 
                                    (symbol_df['Date'] <= padded_end_date)]
        
        # ---------------------------------
        # # Reset index to integers (from dates)
        # symbol_df_filtered.reset_index(drop=True, inplace=True)
        
        # # check if symbol_df_filtered is empty , if so print why
        # if symbol_df_filtered.empty:
        #     # print(f"Symbol {symbol} has no data between {padded_start_date} and {padded_end_date}")
        #     continue
        
        # # Create a MultiIndex for the symbol_df_filtered
        # time_index = range(len(symbol_df_filtered))
        
        # # Create the MultiIndex where the first level is the unique instance counter
        # multi_index = pd.MultiIndex.from_product([[instance_counter], time_index], names=['Instance', 'Time'])
        
        # # Assign the MultiIndex directly to the DataFrame
        # symbol_df_filtered.index = multi_index
        # ---------------------------------
        
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


# --- CELL ---

# import formatAndPreprocess.py in utils folder to use the functions
from utils.formatAndPreprocess import dataset_format

#  create formatted data set for the train and test data
trainDataset = dataset_format(train)
testDataset = dataset_format(test)

# --- CELL ---

import mplfinance as mpf
import pandas as pd

def plot_csgraph(filtered_data):
    # Assuming filtered_data is your DataFrame with OHLC data
    # Ensure the DataFrame has the required columns: 'Open', 'High', 'Low', 'Close'
    required_columns = ['Open', 'High', 'Low', 'Close']
    if all(column in filtered_data.columns for column in required_columns):
        # Set the index to a datetime column if not already set
        if not pd.api.types.is_datetime64_any_dtype(filtered_data.index):
            filtered_data.index = pd.to_datetime(filtered_data.index)

        # Plot the candlestick chart
        mpf.plot(filtered_data, type='candle', style='charles', title='Candlestick Chart', ylabel='Price')
    else:
        print("The DataFrame does not contain the required OHLC columns.")

# --- CELL ---

# Plot the candlestick chart for the filtered data and the normalized data
# plot_csgraph(filtered_data)

# --- CELL ---

# save test and train data to csv
trainDataset.to_csv('Datasets/VanilaDataset/trainDataset.csv')
testDataset.to_csv('Datasets/VanilaDataset/testDataset.csv')

# --- CELL ---

# import formatAndPreprocess.py in utils folder to use the functions
from utils.formatAndPreprocess import dataset_format

#  create formatted data set for the train and test data
trainDataset_w_aug = dataset_format(train_w_aug)
testDataset_w_aug = dataset_format(test_w_aug)

# --- CELL ---

# save test and train data to csv
trainDataset_w_aug.to_csv('Datasets/VanilaDataset/trainDataset_w_aug.csv')
testDataset_w_aug.to_csv('Datasets/VanilaDataset/testDataset_w_aug.csv')

# --- CELL ---

# Create a temporary DataFrame with level 0 indexes and the 'Pattern' column
temp_df = trainDataset.reset_index(level=0)  # Resetting level 1 index to create a flat DataFrame
# drop all the columns except the 'Pattern' column and the Level_0 column
temp_df = temp_df[['Instance','Pattern']]

# drop all the duplicate rows in the temp_df dataframe
temp_df = temp_df.drop_duplicates()

# get the number of each unique value in the 'Pattern' column
pattern_counts = temp_df['Pattern'].value_counts()
print(pattern_counts)

# --- CELL ---

# split the trainDataset_aug_encoded dataframe into X_train and y_train and the testDataset_encoded dataframe into X_test and y_test
X_train = trainDataset.drop(columns='Pattern')
y_train = trainDataset['Pattern']

X_test = testDataset.drop(columns='Pattern')
y_test = testDataset['Pattern']

# drop level 1 indexes from y_train and y_test
y_train = y_train.droplevel(1)
y_test = y_test.droplevel(1)
# now group the y_train and y_test by their indexes
y_train = y_train.groupby(y_train.index).first()
y_test = y_test.groupby(y_test.index).first()

# Display the shapes of the training and testing sets
print(f"X_train Shape: {X_train.shape}, y_train Shape: {y_train.shape}")
print(f"X_test Shape: {X_test.shape}, y_test Shape: {y_test.shape}")

# --- CELL ---

#  save test and train data to csv
X_train.to_csv('Datasets/VanilaDataset/X-Y Splitted Data/X_train.csv')
y_train.to_csv('Datasets/VanilaDataset/X-Y Splitted Data/y_train.csv')
X_test.to_csv('Datasets/VanilaDataset/X-Y Splitted Data/X_test.csv')
y_test.to_csv('Datasets/VanilaDataset/X-Y Splitted Data/y_test.csv')



# --- CELL ---

from sktime.datatypes import check_is_mtype

check_is_mtype(X_train, mtype="pd-multiindex", return_metadata=True)

# --- CELL ---

X_train

# --- CELL ---

from sktime.classification.kernel_based import RocketClassifier
from sktime.transformations.panel.padder import PaddingTransformer
from sktime.datasets import load_unit_test
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

my_padded_multi_rocket = PaddingTransformer() * RocketClassifier(rocket_transform='multirocket')
my_padded_multi_rocket.fit(X_train, y_train)
y_pred_multi = my_padded_multi_rocket.predict(X_test) 

# calculate the accuracy of the model
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred_multi)
print(f"Accuracy: {accuracy:.2f}")

# Create the confusion matrix
cm3 = confusion_matrix(y_test, y_pred_multi)

# Create a DataFrame from the confusion matrix
cm_df3 = pd.DataFrame(cm3, index=pattern_encoding.keys(), columns=pattern_encoding.keys())

# Create the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm_df3, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# --- CELL ---

bjjbjbj

# --- CELL ---

import os
import time
import joblib
from sktime.classification.kernel_based import RocketClassifier
from sktime.classification.shapelet_based import ShapeletLearningClassifierTslearn
from sktime.classification.shapelet_based import ShapeletTransformClassifier
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.classification.dictionary_based import BOSSEnsemble
from sktime.classification.deep_learning import InceptionTimeClassifier
from sktime.classification.hybrid import HIVECOTEV2
from sktime.classification.interval_based import DrCIF
from sktime.transformations.panel.padder import PaddingTransformer
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define the classifiers to test (with correct MiniRocket and MultiRocket setup)
classifiers = {
    "ROCKET": RocketClassifier(rocket_transform='rocket'),
    "MiniROCKET": RocketClassifier(rocket_transform='minirocket'),
    "MultiROCKET": RocketClassifier(rocket_transform='multirocket'),  # MultiRocket definition
    # "InceptionTime": InceptionTimeClassifier(),
    # 'ShapeletLearningTslearn': ShapeletLearningClassifierTslearn(),
    "HIVE-COTE": HIVECOTEV2(),
    "DrCIF": DrCIF()
}

# Initialize a dictionary to store results
results = {}

# Create folder for saving trained models
model_save_dir = 'test_models'
os.makedirs(model_save_dir, exist_ok=True)

# Iterate over classifiers
for name, clf in classifiers.items():
    print(f"Training and testing classifier: {name}")
    
    # Initialize a Padding Transformer
    padder = PaddingTransformer()

    # Apply padding and fit the classifier
    pipeline = padder * clf
    
    # Fit the classifier and save the trained model
    start_time_train = time.time()
    pipeline.fit(X_train, y_train)
    end_time_train = time.time()
    train_duration = end_time_train - start_time_train
    print(f"{name} Training Time: {train_duration:.2f} seconds")
    
    # Save the trained model
    joblib.dump(pipeline, os.path.join(model_save_dir, f"{name}_model.pkl"))
    
    # Measure prediction time
    start_time_predict = time.time()
    y_pred = pipeline.predict(X_test)
    end_time_predict = time.time()
    predict_duration = end_time_predict - start_time_predict
    print(f"{name} Prediction Time: {predict_duration:.2f} seconds")

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.2f}")
    
    # Store results
    results[name] = {
        "accuracy": accuracy,
        "y_pred": y_pred,
        "train_duration": train_duration,
        "predict_duration": predict_duration
    }

    # Calculate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Create a DataFrame from the confusion matrix (adjusting to your 'pattern_encoding' if needed)
    cm_df = pd.DataFrame(cm, index=pattern_encoding.keys(), columns=pattern_encoding.keys())

    # Plot confusion matrix heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, cmap='Blues', fmt='g')
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# Display final accuracy comparison
accuracy_results = {name: result["accuracy"] for name, result in results.items()}
accuracy_df = pd.DataFrame(list(accuracy_results.items()), columns=["Classifier", "Accuracy"])

# Plotting the accuracy comparison
plt.figure(figsize=(10, 6))
sns.barplot(x="Classifier", y="Accuracy", data=accuracy_df)
plt.title("Classifier Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()

# Print detailed timing results
for name, result in results.items():
    print(f"\n{name} Results:")
    print(f"Accuracy: {result['accuracy']:.2f}")
    print(f"Training Time: {result['train_duration']:.2f} seconds")
    print(f"Prediction Time: {result['predict_duration']:.2f} seconds")


# --- CELL ---

import numpy as np
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt



# Define a linear trend function (y = a*x + b)
def linear_trend(x, slope=0.1, intercept=0):
    return slope * x + intercept

# Define a non-linear (quadratic) trend function (y = a*x^2 + b*x + c)
def quadratic_trend(x, a=0.001, b=0.01, c=0):
    return a * x**2 + b * x + c

# Define a sine wave trend function
def sine_trend(x, amplitude=0.02, frequency=0.05 ):
    return amplitude * np.sin(frequency * x)

def traindata_augment(Dataset):
    # Define the number of samples to generate
    n_total_samples = 500
    # Create a counter for generating new first-level indices
    new_first_level_counter = Dataset.index.get_level_values(0).max() + 1

    # Loop through the unique chart patterns in the Dataset
    for pattern in Dataset['Pattern'].unique():
        # Filter the Dataset for the current pattern
        pattern_data = Dataset[Dataset['Pattern'] == pattern]

        # Get the unique values from the first level of the multi-index
        unique_first_level_index = pattern_data.index.get_level_values(0).unique()

        # Set the number of augmented samples to create
        n_samples = n_total_samples - len(unique_first_level_index)

        # Loop through the number of samples to generate
        for i in range(n_samples):
            # Randomly select a section of the DataFrame based on the first-level index
            random_first_level_value = np.random.choice(unique_first_level_index)
            random_section = pattern_data.loc[(random_first_level_value, slice(None)), :]

            # Get the difference between max and min of Close column
            diff = random_section['Close'].max() - random_section['Close'].min()
            adjusted_diff = np.log(1 + diff)  # Adding 1 to avoid log(0)

            # Introduce randomness into the noise level
            noise_level = adjusted_diff * (0.08 + np.random.uniform(-0.01, 0.01))  # Adding random factor
            sub_noise_level = adjusted_diff * (0.01 + np.random.uniform(-0.004, 0.004))  # Adding random factor

            x_values = np.arange(len(random_section))

            # Randomly select a trend function
            trend_function = np.random.choice([linear_trend, quadratic_trend, sine_trend])

            print(f"\nSelected Chart Pattern: {pattern}")
            print(f"\nSelected Trend Function: {trend_function.__name__}")

            # Modify the trend function parameters based on price range and random factors
            if trend_function == linear_trend:
                slope = np.random.uniform(0.005, 0.05) * adjusted_diff * (0.4 + np.random.uniform(-0.01, 0.01))   # Slope depends on price difference
                trend = linear_trend(x_values, slope=slope)
                print('slope:', slope)
            elif trend_function == quadratic_trend:
                a = np.random.uniform(0.00001, 0.0005) * adjusted_diff * (0.4 + np.random.uniform(-0.01, 0.01))   # Quadratic coefficient scaled by diff
                b = np.random.uniform(0.001, 0.01) * adjusted_diff
                trend = quadratic_trend(x_values, a=a, b=b)
                print('a:', a, 'b:', b)
            else:  # sine_trend
                amplitude = np.random.uniform(0.05, 1.2) * adjusted_diff * (0.4 + np.random.uniform(-0.01, 0.01))   # Amplitude depends on price difference
                frequency = np.random.uniform(0.01, 1)
                trend = sine_trend(x_values, amplitude=amplitude, frequency=frequency)
                print('amplitude:', amplitude, 'frequency:', frequency)

            # Plot the trend function
            plt.figure(figsize=(3, 2))
            plt.plot(x_values, trend, label=f'{trend_function.__name__}')
            plt.title('Trend Function Plot')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend()
            plt.grid(True)
            plt.show()

            # Add random noise (minor noise level) for variation in individual OHLC points
            noise = np.random.normal(0, sub_noise_level, random_section[['Open', 'High', 'Low', 'Close', 'Volume']].shape)
            noisy_data = random_section[['Open', 'High', 'Low', 'Close', 'Volume']] + noise

            # Add consistent noise across the same row (major noise level)
            row_noise = np.random.normal(0, noise_level, random_section[['Open']].shape)
            noisy_data['Open'] = random_section['Open'] + row_noise.squeeze()
            noisy_data['High'] = random_section['High'] + row_noise.squeeze()
            noisy_data['Low'] = random_section['Low'] + row_noise.squeeze()
            noisy_data['Close'] = random_section['Close'] + row_noise.squeeze()
            # noisy_data['Adj Close'] = random_section['Adj Close'] + row_noise.squeeze()
            noisy_data['Volume'] = random_section['Volume'] + row_noise.squeeze()

            # Add the trend equally to all OHLC columns
            noisy_data['Open'] += trend
            noisy_data['High'] += trend
            noisy_data['Low'] += trend
            noisy_data['Close'] += trend
            # noisy_data['Adj Close'] += trend
            noisy_data['Volume'] += trend  # You can adjust the impact of the trend on Volume if needed

            # Assign new first-level index to the noisy_data
            new_first_level_index = pd.MultiIndex.from_product([[new_first_level_counter], random_section.index.get_level_values(1)], names=['Index', 'Date'])
            noisy_data.index = new_first_level_index

            # Increment the first-level counter for the next sample
            new_first_level_counter += 1

            # Visualize the original and noisy data using candlestick charts
            
            # Prepare original and noisy data for plotting (OHLC only)
            original_data_for_plot = random_section[['Open', 'High', 'Low', 'Close']].reset_index(drop=True)
            augmented_data_for_plot = noisy_data[['Open', 'High', 'Low', 'Close']].reset_index(drop=True)

            # Create two subplots to compare original and noisy data
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            # Create a DatetimeIndex for the plots (ensure it matches the number of rows in the data)
            date_range = pd.date_range(start='2024-01-01', periods=len(original_data_for_plot), freq='D')

            # Assign the index to the original and noisy data for plotting
            original_data_for_plot.index = date_range
            augmented_data_for_plot.index = date_range

            # Plot the original data
            mpf.plot(original_data_for_plot, type='candle', ax=axes[0], style='yahoo', volume=False)
            axes[0].grid(True)  # Enable gridlines
            axes[0].set_title("Original OHLC Candlestick Chart")

            # Plot the noisy data (augmented with trend)
            mpf.plot(augmented_data_for_plot, type='candle', ax=axes[1], style='yahoo', volume=False)
            axes[1].grid(True)  # Enable gridlines
            axes[1].set_title("Augmented OHLC Candlestick Chart with Trend")

            # Show the comparison plots
            plt.tight_layout()
            plt.show()
            
            # add Pattern column to the noisy data
            noisy_data['Pattern'] = pattern

            # Concatenate the noisy data to the original Dataset
            Dataset = pd.concat([Dataset, noisy_data], axis=0)

    return Dataset



# --- CELL ---

# augment train dataset
trainDataset_aug = traindata_augment(trainDataset)

# shape of the augmented train dataset
print(f'Training Set Shape: {trainDataset_aug.shape}')

# --- CELL ---

# Create a temporary DataFrame with level 0 indexes and the 'Pattern' column
temp_df = trainDataset_aug.reset_index(level=0)  # Resetting level 1 index to create a flat DataFrame
# drop all the columns except the 'Pattern' column and the Level_0 column
temp_df = temp_df[['level_0','Pattern']]

# drop all the duplicate rows in the temp_df dataframe
temp_df = temp_df.drop_duplicates()

# get the number of each unique value in the 'Pattern' column
pattern_counts = temp_df['Pattern'].value_counts()
print(pattern_counts)

# --- CELL ---

X_train_aug = trainDataset_aug.drop(columns='Pattern')
y_train_aug = trainDataset_aug['Pattern']

# drop level 1 indexes from y_train_aug
y_train_aug = y_train_aug.droplevel(1)
# now group the y_train_aug by their indexes
y_train_aug = y_train_aug.groupby(y_train_aug.index).first()

# Display the shapes of the training and testing sets
print(f"X_train_aug Shape: {X_train_aug.shape}, y_train_aug Shape: {y_train_aug.shape}")

# check if the data types are correct
check_is_mtype(X_train_aug, mtype="pd-multiindex", return_metadata=True)


# --- CELL ---

from sktime.classification.kernel_based import RocketClassifier
from sktime.transformations.panel.padder import PaddingTransformer
from sktime.datasets import load_unit_test
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

my_padded_multi_rocket_aug = PaddingTransformer() * RocketClassifier(rocket_transform='multirocket')
my_padded_multi_rocket_aug.fit(X_train_aug, y_train_aug)
y_pred_multi_aug = my_padded_multi_rocket_aug.predict(X_test) 

# calculate the accuracy of the model
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred_multi_aug)
print(f"Accuracy: {accuracy:.2f}")

# Create the confusion matrix
cm3 = confusion_matrix(y_test, y_pred_multi_aug)

# Create a DataFrame from the confusion matrix
cm_df3 = pd.DataFrame(cm3, index=pattern_encoding.keys(), columns=pattern_encoding.keys())

# Create the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm_df3, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
