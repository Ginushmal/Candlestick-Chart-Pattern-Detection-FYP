import requests
from bs4 import BeautifulSoup
import pandas as pd

import logging

logger = logging.getLogger(__name__)

DEFAULT_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:91.0) Gecko/20100101 Firefox/91.0',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'DNT': '1',  # Do Not Track request header
}

def extract_table_from_url(url: str, required_columns: list, headers: dict = DEFAULT_HEADERS) -> pd.DataFrame:
    """Extracts the pattern tables from a given Bulkowski's blog URL."""
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch {url}: {e}")
        return pd.DataFrame()

    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find all tables with the target class
    tables = soup.find_all('table', class_='BlogTableTiling MyTable')
    logger.info(f"Found {len(tables)} tables on {url}")
    
    tablesdf_combined = pd.DataFrame()
    i = 0
    
    for table in tables:
        rows = table.find_all('tr')
        if not rows:
            continue
            
        # Get the first row (headers)
        headers_row = rows[0]
        extracted_headers = [td.get_text(strip=True).replace('\n', ' ').strip() for td in headers_row.find_all('td')]
        
        # Check if headers match the required columns
        if extracted_headers == required_columns:
            logger.info(f"Found matching table {i + 1} on {url}")
            i += 1
            
            data_rows = []
            for row in rows[1:]:
                row_data = []
                for idx, td in enumerate(row.find_all('td')):
                    # Check if it's the "BullishBearish" column (assuming it's the 3rd column based on required_columns)
                    if idx == 2:
                        bgcolor = td.get('bgcolor', '').lower()
                        if bgcolor == '#ff0000':
                            row_data.append(-1)  # Red color, mark as -1
                        elif bgcolor == '#008000':
                            row_data.append(1)   # Green color, mark as 1
                        else:
                            row_data.append(0)   # No color or other color, mark as 0
                    else:
                        row_data.append(td.get_text(strip=True))
                data_rows.append(row_data)

            tabledf = pd.DataFrame(data_rows, columns=extracted_headers)
            tablesdf_combined = pd.concat([tablesdf_combined, tabledf], ignore_index=True)
            
    return tablesdf_combined

def scrape_pattern_tables(start_year: int, end_year: int, required_columns: list, months: list = None, headers: dict = DEFAULT_HEADERS) -> pd.DataFrame:
    """Loops through months and years, extracting required tables from Bulkowski's blog."""
    df_list = []
    if not months:
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    for year in range(start_year, end_year + 1):
        for month in months:
            url = f"https://thepatternsite.com/Blog-{month}{str(year)[-2:]}.html"
            logger.info(f"Scraping {url}")
            table_df = extract_table_from_url(url, required_columns, headers)
            if not table_df.empty:
                df_list.append(table_df)
            else:
                logger.info(f"No matching table found for {url}")
    
    if df_list:
        return pd.concat(df_list, ignore_index=True)
    return pd.DataFrame()

def extract_full_names_from_url(url: str, short_names: list, headers: dict = DEFAULT_HEADERS) -> dict:
    """Extracts full stock names from a URL based on short names."""
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch {url}: {e}")
        return {}

    soup = BeautifulSoup(response.text, 'html.parser')
    logger.info(f"Scraping {url} for full names")

    full_names = {}
    for short_name in short_names.copy():
        # Search for the short name in the text
        entry = soup.find('div', string=lambda x: x and short_name in x)
        if entry:
            # Get the bold text (full name)
            bold_text = entry.find('span', style=lambda x: x and 'font-weight: bold;' in x)
            if bold_text:
                full_names[short_name] = bold_text.text.strip()
                logger.info(f"Found full name for {short_name}: {full_names[short_name]}")
                short_names.remove(short_name)

    return full_names

def scrape_full_names(start_year: int, end_year: int, short_names: list, headers: dict = DEFAULT_HEADERS) -> dict:
    """Loops through months and years, extracting full names."""
    all_full_names = {}
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    
    short_names_copy = short_names.copy()

    for year in range(start_year, end_year + 1):
        for month in months:
            url = f"https://thepatternsite.com/Blog-{month}{str(year)[-2:]}.html"
            
            if not short_names_copy:
                break
            
            full_names = extract_full_names_from_url(url, short_names_copy, headers)
            all_full_names.update(full_names)

        if not short_names_copy:
            break

    return all_full_names
