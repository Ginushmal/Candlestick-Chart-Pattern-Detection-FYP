# Load csv as pandas dataframe
import pandas as pd


# --- CELL ---

import requests
from bs4 import BeautifulSoup
import pandas as pd

# Set headers to mimic a browser request
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:91.0) Gecko/20100101 Firefox/91.0',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'DNT': '1',  # Do Not Track request header
}

# Function to extract tables from a given URL
def extract_table_from_url(url, required_columns, headers):
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find all tables with the target class
    tables = soup.find_all('table', class_='BlogTableTiling MyTable')
    
    print(f"Found {len(tables)} tables on {url}")
    
    tablesdf_combined = pd.DataFrame()  # Initialize empty DataFrame for all tables
    i = 0  # Counter to track number of matching tables
    
    # Iterate over found tables
    for table in tables:
        # Get the first row (headers)
        headers_row = table.find_all('tr')[0]
        headers = [td.get_text(strip=True) for td in headers_row.find_all('td')]
        
        # Normalize and remove extra spaces
        headers = [header.replace('\n', ' ').strip() for header in headers]

        # print(f"Extracted headers: {headers}")
        
        # Check if headers match the required columns
        if headers == required_columns:
            print(f"Found matching table {i + 1} on {url}")
            i += 1
            
            # If headers match, convert the table to a pandas DataFrame
            rows = []
            for row in table.find_all('tr')[1:]:
                row_data = []
                for idx, td in enumerate(row.find_all('td')):
                    # Check if it's the "BullishBearish" column (assuming it's the 3rd column based on the example)
                    if idx == 2:
                        bgcolor = td.get('bgcolor', '').lower()  # Get the background color and normalize case
                        if bgcolor == '#ff0000':
                            row_data.append(-1)  # Red color, mark as -1
                        elif bgcolor == '#008000':
                            row_data.append(1)  # Green color, mark as 1
                        else:
                            row_data.append(0)  # No color or other color, mark as 0 (or leave as is)
                    else:
                        row_data.append(td.get_text(strip=True))  # For all other columns
                rows.append(row_data)

            tabledf = pd.DataFrame(rows, columns=headers)
            
            # Concatenate the current DataFrame with the previously combined ones
            tablesdf_combined = pd.concat([tablesdf_combined, tabledf], ignore_index=True)
            
            # print(f"Table {i} extracted: \n{tabledf.head()}")  # Print the first few rows of the extracted table
            
    return tablesdf_combined




# Function to loop through months and years, extracting the required table
def scrape_tables(start_year, end_year, required_columns, headers):
    df_list = []
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    for year in range(start_year, end_year + 1):
        for month in months:
            url = f"https://thepatternsite.com/Blog-{month}{str(year)[-2:]}.html"
            print(f"Scraping {url}")

            # Extract table from the URL
            table_df = extract_table_from_url(url, required_columns, headers)
            
            if table_df is not None:
                df_list.append(table_df)
            else:
                print(f"No matching table found for {url}")
    
    # Concatenate all DataFrames into one
    if df_list:
        final_df = pd.concat(df_list, ignore_index=True)
        return final_df
    else:
        return pd.DataFrame()  # Return empty DataFrame if no tables were found

# Define the required columns for the table
required_columns = ['Symbol', 'Chart Pattern', 'BullishBearish', 'Start', 'End', 'Industry']

# Scrape tables from the blog pages between 2020 and 2024
final_data = scrape_tables(2020, 2025, required_columns, headers)

# Save the final DataFrame to a CSV file
final_data.to_csv('Datasets/scraped_blog_tables.csv', index=False)
print("Tables successfully scraped and saved to 'scraped_blog_tables.csv'")

# extract_table_from_url("https://thepatternsite.com/Blog-Jan20.html", required_columns, headers)


# --- CELL ---

final_data

# --- CELL ---



# Load csv as pandas dataframe
# cleanedPatternDf = pd.read_csv('Extracted Cleaned data of 2019 - 2024 chart patterns by Mr. Bulkowski.csv')
cleanedPatternDf = pd.read_csv('Datasets/scraped_blog_tables.csv')
cleanedPatternDf

# --- CELL ---

# create a new df with the xolumn number of days from start to end of the pattern
patternDf = cleanedPatternDf.copy()
patternDf['Start'] = pd.to_datetime(patternDf['Start'])
patternDf['End'] = pd.to_datetime(patternDf['End'])
patternDf['Days'] = (patternDf['End'] - patternDf['Start']).dt.days
patternDf.head()


# --- CELL ---

# get the min max and mean of the days
minDays = patternDf['Days'].min()
print("Min Days: ", minDays)

maxDaysDays = patternDf['Days'].max()
print("Max Days: ", maxDaysDays)

meanDays = patternDf['Days'].mean()
print("Mean Days: ", meanDays)

# --- CELL ---

# number of rows in the dataframe
numRows = len(patternDf)
print("Number of rows: ", numRows)

# --- CELL ---

import pandas as pd
import matplotlib.pyplot as plt

# Assuming you have loaded your DataFrame as patternDf
# patternDf = pd.read_csv('your_data.csv') # Example of loading data if needed

# Plot a histogram to visualize the distribution of 'Days' data
plt.figure(figsize=(10, 6))
plt.hist(patternDf['Days'], bins=30, color='blue', alpha=0.7, edgecolor='black')
plt.title('Distribution of Days')
plt.xlabel('Days')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)

plt.show()


# --- CELL ---

# get the number of occurances of each values in the Symbol column
numberofTimes=cleanedPatternDf['Symbol'].value_counts()
numberofTimes

# --- CELL ---

#  get all the unique values in the Chart Pattern column 
uniquePatterns=cleanedPatternDf['Chart Pattern'].value_counts()
uniquePatterns.head(30)

# --- CELL ---

#  number of occurances of the Cup with handle pattern  
cupWithHandle=cleanedPatternDf['Chart Pattern'].str.contains('Cup with handle').sum()
cupWithHandle

# --- CELL ---

# search for the row with the symbol 'AAPL' in numberofTimes
numberofTimes.loc['NVDA']


# --- CELL ---

import os
import yfinance as yf
import pandas as pd
from datetime import datetime
from tqdm import tqdm

# Get today's date
end_date = datetime.today().strftime('%Y-%m-%d')

# Set to store symbols with errors
error_symbols = set()

# Create the "OHLC data" folder if it doesn't exist
output_directory = 'Datasets/OHLC data'
os.makedirs(output_directory, exist_ok=True)  # Creates the directory if it doesn't exist

# Assuming numberofTimes is already defined as a DataFrame
for symbol in tqdm(numberofTimes.index, desc="Downloading Data", unit="symbol"):
    try:
        # Download data with end date set to today's date
        data = yf.download(symbol, start='2019-01-01', end=end_date, progress=False)
        
        # Check if data is empty
        if data.empty:
            # print(f"No data found for {symbol}. It may be delisted or unavailable.")
            error_symbols.add(symbol)  # Add to error set
            continue
        
        # convert the date to datetime
        
        
        # Save data to a CSV file
        data.to_csv(f'{output_directory}/{symbol}.csv')  # Save in the created folder
        
    except Exception as e:
        # print(f"Failed to download {symbol}: {str(e)}")
        error_symbols.add(symbol)  # Add to error set

# Print the symbols that encountered errors
if error_symbols:
    print("Symbols with download errors:")
    print(error_symbols)
else:
    print("All symbols downloaded successfully.")


# --- CELL ---

#  check for the empty csv files in the "OHLC data" folder
import os
import pandas as pd

# get the list of all the files in the "OHLC data" folder
files = os.listdir('Datasets/OHLC data')

emptyFiles = []

# check for the empty csv files in the "OHLC data" folder
for file in files:
    data = pd.read_csv('Datasets/OHLC data/'+file)
    if data.empty:
        # add the symbol of the empty csv file to the emptyFiles list
        emptyFiles.append(file)
        

print(emptyFiles)

# --- CELL ---

# number of empty files
len(emptyFiles)

# --- CELL ---


# Normalize and remove the .CSV extension
emptyFiles = [symbol.strip().upper().replace('.CSV', '').replace('.CSV', '') for symbol in emptyFiles]


# --- CELL ---

import subprocess
import json
import pandas as pd
import os
from urllib.parse import urlencode
from datetime import datetime

def get_data(stock_id):
    try:
        url = f'https://api.investing.com/api/financialdata/historical/{stock_id}'
        params = {
            'start-date': '2020-07-01',
            'end-date': '2024-08-31',
            'time-frame': 'Daily',
            'add-missing-rows': 'false'
        }

        cnfg = ['curl', '-A', 'Chrome/128.0.0.0', '-H', 'domain-id: www', '-G', url, '-d', urlencode(params)]
        output = subprocess.run(cnfg, capture_output=True).stdout.decode()
        return json.loads(output)
    except Exception as e:
        print(f"Error fetching data for {stock_id}: {str(e)}")
        return None

def save_data_to_csv(stock_id, stock_data, folder_path="Datasets/OHLC data"):
    os.makedirs(folder_path, exist_ok=True)
    
    columns = ['Price', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume', 'Ticker', 'Date']
    data_rows = []

    # Parse and format the data into a list of dictionaries
    for entry in stock_data.get('data', []):
        date = datetime.utcfromtimestamp(entry['rowDateRaw']).strftime('%Y-%m-%d %H:%M:%S+00:00')
        data_rows.append({
            'Price': entry['last_closeRaw'],
            'Adj Close': entry['last_closeRaw'],
            'Close': entry['last_closeRaw'],
            'High': entry['last_maxRaw'],
            'Low': entry['last_minRaw'],
            'Open': entry['last_openRaw'],
            'Volume': entry['volumeRaw'],
            'Ticker': stock_id,
            'Date': date
        })

    df = pd.DataFrame(data_rows, columns=columns)
    
    file_path = os.path.join(folder_path, f"{stock_id}.csv")
    
    df.to_csv(file_path, index=False)


# --- CELL ---

stocks = {
    'WIRE': '17555',   # Encore Wire Corp
    'CSOD': '15833',   # Cornerstone OnDemand Inc
    'BBBY': '6389',   # Bed Bath & Beyond Inc
    'STMP': '17267',   # Stamps.com Inc
    'RE': '20171',     # Everest Re Group Ltd (assuming you meant Everest Re)
    'COG': '13835',    # Cabot Oil & Gas Corporation
    'KBAL': '16443',   # Kimball International, Inc.
    'ADS': '32507',    # Alliance Data Systems Corp
    'ERA': '41207',    # Era Group Inc
    'NWY': '8130',     # New York & Company Inc
    'ANTM': '958110',   # Anthem Inc
    'SLCA': '29673',  # U.S. Silica Holdings Inc
    'ABC': '8060',    # AmerisourceBergen Corp
    'NVTA': '',  # Invitae Corp
    'XLNX': '',   # Xilinx Inc
    'CHS': '',     # Chico's FAS Inc.
    'ASNA': '',    # Ascena Retail Group
    'CLR': '111508',   # Continental Resources Inc.
    'FLIR': '3038',    # FLIR Systems Inc
    'HSC': '20517',    # Harsco Corp
    'AJRD': '17292',   # Aerojet Rocketdyne Holdings Inc
    'AYX': '159875',   # Alteryx, Inc
    'CREE': '3021',    # Cree Inc
    'LAWS': '17022',   # Lawson Products
    'TLRD': '28820',   # Tailored Brands Inc
    'CONN': '2969',    # Conn's Inc
    'REV': '2560',     # Revlon Inc
    'JCP': '20306',    # J.C. Penney Company Inc
    'NBL': '2971',     # Noble Energy Inc.
    'AEL': '20560',    # American Equity Investment Life Holding
    'CY': '20461',     # Cypress Semiconductor
    'IVC': '2888',     # Invacare Corp.
    'FFG': '17260',    # FBL Financial Group
    'POL': '21312',    # PolyOne Corp
    'MDC': '24784',    # MDC Holdings Inc.
    'EIGI': '26314',   # Endurance International Group Holdings
    'NXGN': '115196',  # NextGen Healthcare Inc
    'AXE': '20499',    # Anixter International Inc
    'PICO': '24805',   # PICO Holdings
    'JCOM': '17282',   # J2 Global Inc.
    'MYL': '11510',    # Mylan Laboratories Inc.
    'MLHR': '2704',    # Miller, Herman Inc.
    'FOE': '20616',    # Ferro Corp
    'CLGX': '25655',   # CoreLogic Inc
    'FEYE': '988464'   # FireEye Inc
}


# --- CELL ---

for ticker, stock_id in stocks.items():
    data = get_data(stock_id)
    print (data)
    # if data is not none or null
    if data.get('data'):
        save_data_to_csv(ticker, data)
        
    else:
        print(f"No data found for {ticker} with stock ID {stock_id}")

# --- CELL ---

# coppy the error_symbols set to a new set
all_error_symbols = error_symbols.copy()

# convert the emptyFiles list to a set and append it to the all_error_symbols set
all_error_symbols.update(set(emptyFiles))

# number of symbols with errors
print(len(all_error_symbols))

print("Symbols with download errors:")
print(all_error_symbols)

# --- CELL ---

# loop through cleanedPatternDf rows and check if the data range of each pattern data available in the OHLC data in the "OHLC data" folder files , if not create a list of such symbols
symbols_with_half_data_error = []
for index, row in cleanedPatternDf.iterrows():
    symbol = row['Symbol']
    # check if the symbol is NOT in the all_error_symbols set
    if symbol not in all_error_symbols:
        # get the data of the symbol from the "OHLC data" folder
        data = pd.read_csv(f'Datasets/OHLC data/{symbol}.csv')
        
        data['Date'] = pd.to_datetime(data['Date'])
        row['Start'] = pd.to_datetime(row['Start'])
        row['End'] = pd.to_datetime(row['End'])
        #  check if the start and end date range of the row is available in the data
        if data[(data['Date'] >= row['Start']) & (data['Date'] <= row['End'])].empty:
            symbols_with_half_data_error.append(symbol)
            print(f"Symbol {symbol} has missing data for the range {row['Start']} to {row['End']}")
        


# --- CELL ---

print(len(symbols_with_half_data_error))
print(symbols_with_half_data_error)


# --- CELL ---

patternDf['Symbol'] = patternDf['Symbol'].str.strip().str.upper()
all_error_symbols = [symbol.strip().upper() for symbol in all_error_symbols]

# --- CELL ---

# Get the number of rows for each chart pattern of any symbol that has an empty csv file
pattern_count = patternDf[patternDf['Symbol'].isin(all_error_symbols)]['Chart Pattern'].value_counts()

print("Pattern counts for symbols with empty CSV files:", pattern_count)


# --- CELL ---

import requests
from bs4 import BeautifulSoup
import pandas as pd

# Set headers to mimic a browser request
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:91.0) Gecko/20100101 Firefox/91.0',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'DNT': '1',  # Do Not Track request header
}

# Function to extract full stock names from a given URL based on short names
def extract_full_names_from_url(url, short_names, headers):
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    print(f"Scraping {url}")

    full_names = {}

    # Iterate over the short names and find their full names in the HTML
    for short_name in short_names.copy():  # Copy the list to avoid modifying during iteration
        # Search for the short name in the text
        entry = soup.find('div', text=lambda x: x and short_name in x)
        if entry:
            # Get the bold text (full name)
            bold_text = entry.find('span', style=lambda x: x and 'font-weight: bold;' in x)
            if bold_text:
                full_names[short_name] = bold_text.text.strip()
                print(f"Found full name for {short_name}: {full_names[short_name]}")
                # remove the found short name from the list
                short_names.remove(short_name)

    return full_names

# Function to loop through months and years, extracting full names
def scrape_full_names(start_year, end_year, short_names, headers):
    all_full_names = {}
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    
    short_names_copy = short_names.copy()  # Create a copy of the list to avoid modifying the original

    for year in range(start_year, end_year + 1):
        for month in months:
            url = f"https://thepatternsite.com/Blog-{month}{str(year)[-2:]}.html"
            
            # Check if the short names list is empty, break if it is
            if not short_names_copy:
                break
            
            # Extract full names from the URL
            full_names = extract_full_names_from_url(url, short_names_copy, headers)
            all_full_names.update(full_names)  # Update the dictionary with found names

        # After each month loop, check again if short_names_copy is empty
        if not short_names_copy:
            break

    return all_full_names

# Scrape full names from the blog pages between 2020 and 2024
full_names_data = scrape_full_names(2020, 2024, all_error_symbols, headers)

# Convert the result to a DataFrame
full_names_df = pd.DataFrame(list(full_names_data.items()), columns=['Short Name', 'Full Name'])

# Set the index to be the short names
full_names_df.set_index('Short Name', inplace=True)

# Save the DataFrame to a CSV file
full_names_df.to_csv('Datasets/scraped_full_names.csv', index=True)
print("Full names successfully scraped and saved to 'scraped_full_names.csv'")



# --- CELL ---

full_names_df

# --- CELL ---

# full names of symbols_with_half_data_error 
half_error_full_names = scrape_full_names(2019, 2024, symbols_with_half_data_error, headers)
half_error_full_names

# --- CELL ---

# loop through all the .csv files in the "OHLC data" folder and filter out only Date , Open , High , Low , Close , Volume columns and replace the existing files with the filtered data
import os
import pandas as pd

# get the list of all the files in the "OHLC data" folder
files = os.listdir('Datasets/OHLC data')

# loop through all the .csv files in the "OHLC data" folder
for file in files:
    data = pd.read_csv('Datasets/OHLC data/'+file)
    # filter out only Date , Open , High , Low , Close , Volume columns
    data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    # replace the existing files with the filtered data
    data.to_csv('Datasets/OHLC data/'+file, index=False)
    


# --- CELL ---

# create a copy of cleanedPatternDf where it doesnt have rows with symbols in symbols_with_half_data_error list and all_error_symbols set
cleanedPatternDf_copy = cleanedPatternDf[~cleanedPatternDf['Symbol'].isin(symbols_with_half_data_error)]
cleanedPatternDf_copy = cleanedPatternDf_copy[~cleanedPatternDf_copy['Symbol'].isin(all_error_symbols)]

#  get all the unique values in the Chart Pattern column 
uniquePatterns_without_errors=cleanedPatternDf_copy['Chart Pattern'].value_counts()
uniquePatterns_without_errors.head(30)

# --- CELL ---

# Initialize an empty list to store the data
data = []

for file in emptyFiles:
    # Create a temporary dictionary with the current 'Symbol' and 'Times'
    temp_dict = {
        'Symbol': file[:-4],
        'Times': numberofTimes.loc[file[:-4]]
    }
    # Append the dictionary to the list
    data.append(temp_dict)

# Create the DataFrame from the list of dictionaries
nullfilesPatternTimesDf = pd.DataFrame(data)

nullfilesPatternTimesDf

# --- CELL ---

# #  delete the empty csv files in the "OHLC data" folder and ""OHLC data pattern added" folder
# import os

# # get the list of all the files in the "OHLC data" folder
# files = os.listdir('OHLC data')

# # delete the empty csv files in the "OHLC data" folder
# for file in emptyFiles:
#     os.remove
#     ('OHLC data/'+file)
#     print(file)
#     print('---------------------------------------------')
    


# --- CELL ---

# from the ""OHLC data" folder get te "NVDA.csv" file which has these columns now : Date,Open,High,Low,Close,Adj Close,Volume , and plot the candle stick pattern chart and mark the chart pattern on the chart by using the start and end date of the each chart pattern for the symbol 'NVDA' in the cleanedPatternDf dataframe where it has Symbol,Chart Pattern,Bullish/Bearish,Start,End columns 

import pandas as pd

# Load NVDA CSV file
nvda_df = pd.read_csv("Datasets/OHLC data/NVDA.csv")

# Ensure the Date column is of datetime type
nvda_df['Date'] = pd.to_datetime(nvda_df['Date'])

# Set the Date as the index
nvda_df.set_index('Date', inplace=True)



# --- CELL ---

# loop through all the data files in OHLC data folder and conert the Date column to datetime type and sort the data by Date , then reset the index and save the data to the same file
import os
import pandas as pd

# get the list of all the files in the "OHLC data" folder
files = os.listdir('Datasets/OHLC data')

# loop through all the data files in OHLC data folder
for file in files:
    data = pd.read_csv('Datasets/OHLC data/'+file)
    # convert the Date column to datetime type
    data['Date'] = pd.to_datetime(data['Date'])
    # sort the data by Date
    data.sort_values('Date', inplace=True)
    # reset the index
    data.reset_index(drop=True, inplace=True)
    # save the data to the same file
    data.to_csv('Datasets/OHLC data/'+file, index=False)

# --- CELL ---

cleanedPatternDf

# --- CELL ---

import pandas as pd
import plotly.graph_objects as go

symbol_name = 'GOOGL'

# Step 1: Read CSV Data
nvda_df = pd.read_csv('Datasets/OHLC data/'+symbol_name+'.csv')

# Step 2: Prepare Data
nvda_df['Date'] = pd.to_datetime(nvda_df['Date'])

# Step 3: Plot Candlestick Chart
fig = go.Figure(data=[go.Candlestick(x=nvda_df['Date'],
                                     open=nvda_df['Open'],
                                     high=nvda_df['High'],
                                     low=nvda_df['Low'],
                                     close=nvda_df['Close'])])

# Step 4: Prepare cleanedPatternDf

cleanedPatternDf['Start'] = pd.to_datetime(cleanedPatternDf['Start'])
cleanedPatternDf['End'] = pd.to_datetime(cleanedPatternDf['End'])

# Step 5: Mark Patterns
for index, row in cleanedPatternDf.iterrows():
    if row['Symbol'] == symbol_name:
        start_date = row['Start']
        end_date = row['End']
        pattern = row['Chart Pattern']
        color = 'rgba(0, 0, 255, 0.2)' if row['BullishBearish'] == 'Bullish' else 'rgba(255, 0, 0, 0.2)'
        
        # Add shaded area for pattern range
        fig.add_shape(
            type="rect",
            xref="x",
            yref="paper",
            x0=start_date,
            y0=0,
            x1=end_date,
            y1=1,
            fillcolor=color,
            opacity=0.2,
            layer="below",
            line_width=0,
        )
        
        # Add annotation for pattern
        fig.add_annotation(x=start_date, y=nvda_df['High'].max(), text=pattern, showarrow=True, arrowhead=1)

# Step 6: Adjust Layout for Height and Zoom Controls
fig.update_layout(
    title=symbol_name+' Candlestick Chart with Pattern Annotations',
    height=800,
    autosize=True,
    yaxis={'fixedrange': False},
    xaxis={'fixedrange': False, 'rangeslider': {'visible': True}}
)

# Step 7: Display Chart
fig.show()


# --- CELL ---

# get all the data in cleanedPatternDf where symbol is 'NVDA'
cleanedPatternDf[cleanedPatternDf['Symbol'] == 'BBBY']

# --- CELL ---

import pandas as pd
import plotly.graph_objects as go
import random


pattern = 'Head-and-shoulders top'
n = 15

# Get all the rows with 'Head-and-shoulders top' chart pattern
head_and_shoulders_df = cleanedPatternDf[cleanedPatternDf['Chart Pattern'] == pattern]

# Get a random sample of 10 rows
random_head_and_shoulders_df = head_and_shoulders_df.sample(n=n)

# Initialize the figure
fig = go.Figure()

# Loop through the random sample
for index, row in random_head_and_shoulders_df.iterrows():
    symbol = row['Symbol']
    start_date = row['Start']
    end_date = row['End']
    
    # Adjust the date range to include a 5-day padding
    padded_start_date = start_date - pd.Timedelta(days=0)
    padded_end_date = end_date + pd.Timedelta(days=0)
    
    # Read the CSV file for the symbol but efore check if the csv file exists , if not skip this iteration
    try:
        symbol_df = pd.read_csv(f'Datasets/OHLC data/{symbol}.csv')
    except FileNotFoundError:
        # print(f"Data not found for {symbol}. Skipping...")
        continue

    # convert the data column to datetime type
    symbol_df['Date'] = pd.to_datetime(symbol_df['Date'])

    symbol_df['Date'] = symbol_df['Date'].dt.tz_localize(None)

    # Filter the DataFrame to include only the date range of the pattern with padding
    symbol_df_filtered = symbol_df[(symbol_df['Date'] >= padded_start_date) & (symbol_df['Date'] <= padded_end_date)]
    
    # Add the candlestick chart for the filtered date range
    fig.add_trace(go.Candlestick(x=symbol_df_filtered['Date'],
                                 open=symbol_df_filtered['Open'],
                                 high=symbol_df_filtered['High'],
                                 low=symbol_df_filtered['Low'],
                                 close=symbol_df_filtered['Close'],
                                 name=symbol))
    
    # Add shaded area for pattern range
    fig.add_shape(
        type="rect",
        xref="x",
        yref="paper",
        x0=start_date,
        y0=0,
        x1=end_date,
        y1=1,
        fillcolor='rgba(0, 0, 255, 0.2)',
        opacity=0.2,
        layer="below",
        line_width=0,
    )
    
    # Add annotation for pattern
    fig.add_annotation(x=start_date, y=symbol_df_filtered['High'].max(), text=pattern, showarrow=True, arrowhead=1)

# Update the layout
fig.update_layout(
    title=pattern+' Candlestick Chart with Pattern Annotations',
    height=800,
    autosize=True,
    yaxis={'fixedrange': False},
    xaxis={'fixedrange': False, 'rangeslider': {'visible': True}}
)

# Display the chart
fig.show()


# --- CELL ---

#  display all columns in pandas dataframe
pd.set_option('display.max_columns', None)

# --- CELL ---

# load Datasets/CSE Data/WATCH_SEP2020_SEP2024.xlsx as a pandas dataframe
raw_cse_df = pd.read_excel('Datasets/CSE Data/WATCH_SEP2020_SEP2024.xlsx')
raw_cse_df

# --- CELL ---

#  create the OHLC dartaframe from the raw_cse_df dataframe
cse_df = raw_cse_df[["SECURITYCODE","TRADEDATE","OPENINGPRICE","HIGHPX","LOWPX","CLOSINGPRICE","TOTALVOLUME"]].copy()

# rename the columns to match the OHLC columns
cse_df.rename(columns={"SECURITYCODE":"Symbol", "TRADEDATE":"Date", "OPENINGPRICE":"Open", "HIGHPX":"High", "LOWPX":"Low", "CLOSINGPRICE":"Close", "TOTALVOLUME":"Volume"}, inplace=True)

cse_df

# --- CELL ---

# set NaN values to 0
# cse_df.fillna(0, inplace=True)

# --- CELL ---

# create separate csv files for each symbol in the cse_df dataframe in the "CSE Data/OHLC Data" folder
import os

# Create the "OHLC Data" folder if it doesn't exist
output_directory = 'Datasets/CSE Data/OHLC Data'
os.makedirs(output_directory, exist_ok=True)  # Creates the directory if it doesn't exist

# Get the unique symbols in the DataFrame
symbols = cse_df['Symbol'].unique()

# Save each symbol's data to a separate CSV file
for symbol in symbols:
    symbol_data = cse_df[cse_df['Symbol'] == symbol].copy()
    symbol_data.drop('Symbol', axis =1 , inplace = True)
    symbol_data.to_csv(f'{output_directory}/{symbol}.csv', index=False)
    
print("Data saved successfully.")
