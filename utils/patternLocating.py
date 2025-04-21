import os
from tqdm import tqdm
from utils.formatAndPreprocessNewPatterns import normalize_ohlc_len, normalize_ohlc_segment ,get_pattern_encoding
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
import math
from sklearn.cluster import DBSCAN
from scipy.signal import find_peaks
import mplfinance as mpf
import matplotlib.dates as mdates

path = 'Datasets/OHLC data'


def patter_locate_test_data_create(train_patterns, test_patterns, extra_days,seed=69):
    segment_id = 0
    # set a seed to np.random
    np.random.seed(seed)
    # Create a new empty dataframe with Seg_ID, Seg_Start, Seg_End columns + the same columns as the test_patterns
    columns = ['Seg_ID', 'Seg_Start', 'Seg_End'] + list(test_patterns.columns)
    # Set the path to the folder containing the data
    path = 'Datasets/OHLC data'
    # Create an empty dataframe
    test_pattern_segment_wise = pd.DataFrame(columns=columns)



    # loop through the files in the folder
    for filename in tqdm(os.listdir(path), desc="Processing files"):
            if filename.endswith('.csv'):
            # print(filename)
                OHCL_symbol_df = pd.read_csv(path + '/' + filename)
                # print(df.head())
                
                # convert the date columns to datetime
                OHCL_symbol_df['Date'] = pd.to_datetime(OHCL_symbol_df['Date'])
                
                max_available_date = OHCL_symbol_df['Date'].max()
                min_available_date = OHCL_symbol_df['Date'].min()
                
                name = os.path.splitext(filename)[0]
                
                # get the rows of pattern_labled_df where Symbol == name
                train_data_this_symbol = (train_patterns[train_patterns['Symbol'] == name]).copy()
                test_data_this_symbol = (test_patterns[test_patterns['Symbol'] == name]).copy()
                
                # reset the index of the dataframes
                train_data_this_symbol.reset_index(drop=True, inplace=True)
                test_data_this_symbol.reset_index(drop=True, inplace=True)
                
                # in test data create a new column ID and give it the same value as the index
                test_data_this_symbol['ID'] = test_data_this_symbol.index
                
                # print(test_data_this_symbol)
                
                
                # convert the date columns to datetime
                train_data_this_symbol['Start'] = pd.to_datetime(train_data_this_symbol['Start'])
                train_data_this_symbol['End'] = pd.to_datetime(train_data_this_symbol['End'])
                test_data_this_symbol['Start'] = pd.to_datetime(test_data_this_symbol['Start'])
                test_data_this_symbol['End'] = pd.to_datetime(test_data_this_symbol['End'])
                
                
                # 01. pick one test item
                # 02. get the left cut off date as the max of start date - 50 or the cthe end date of a train data point that has a end date between test start date and test start - 50
                # 03. get the right cut off date as the min of end date + 50 or the cthe start date of a train data point that has a start date between test end date and test end + 50
                # 04. get the tets data points that can fit in between the left and right cut off dates
                # 05. iterate the set of test items that is to the left of the selected one , as the new start date is the start date of the i th items start date and the end date is that + 100 + the length of the selcted item
                # 06. get the number of test items that fit between the segment in each iteration , and select the option that has the max number of test items
                # 07. get the wiggle room by the min start date of the test items that fit in the segment and the max end date of the test items that fit in the segment and try to randomise the start and end dates of the segment
                
                # get a list of randomly ordered test item indexes
                index_touple = list(range(len(test_data_this_symbol)))
                np.random.shuffle(index_touple)
                
                # 01. pick one test item
                
                for test_idx in index_touple:
                    # get the test item
                    test_item = test_data_this_symbol[test_data_this_symbol['ID'] == test_idx]
                    index_touple.remove(test_idx)
                    selected_test_item_length = test_item['End'] - test_item['Start']
                    
                    # 02. get the left  and right min and max possible segment dates 
                    min_possible_seg_start = (test_item['Start'] - pd.to_timedelta(extra_days, unit='D')).iloc[0]
                    max_possible_seg_end = (test_item['End'] + pd.to_timedelta(extra_days, unit='D')).iloc[0]
                    
                    # 03. get the left and right cut off dates avoiding the train data points that are within the possible segment
                    # get the train data points that are within the possible segment
                    train_data_that_fit = train_data_this_symbol[(train_data_this_symbol['Start'] <= max_possible_seg_end) | (train_data_this_symbol['End'] >= min_possible_seg_start)]
                    
                    # get the  maximum of the End dates of train_data_that_fit
                    left_cut_off = min_possible_seg_start
                    right_cut_off = max_possible_seg_end
                    # left_cut_off = min(max(min_possible_seg_start, train_data_that_fit['End'].max()),test_item['Start'].iloc[0])
                    if (((train_data_that_fit['Start'] < test_item['Start'].iloc[0]) & (train_data_that_fit['End'] > test_item['End'].iloc[0])).any()) :
                        left_cut_off = test_item['Start'].iloc[0]
                    elif (((train_data_that_fit['Start'] < test_item['Start'].iloc[0]) & (train_data_that_fit['End'] < test_item['Start'].iloc[0])).any()) :
                        Left_cut_off = train_data_that_fit[((train_data_that_fit['Start'] < test_item['Start'].iloc[0]) & (train_data_that_fit['End'] < test_item['Start'].iloc[0]))]['End'].max()
                    else :
                        left_cut_off = min_possible_seg_start
                    
                    left_cut_off = max(left_cut_off, min_possible_seg_start)
                    
                    # get the  minimum of the Start dates of train_data_that_fit
                    # right_cut_off = max(min(max_possible_seg_end, train_data_that_fit['Start'].min()),test_item['End'].iloc[0])
                    if (((train_data_that_fit['Start'] < test_item['End'].iloc[0]) & (train_data_that_fit['End'] > test_item['End'].iloc[0])).any()) :
                        right_cut_off = test_item['End'].iloc[0]
                    elif (((train_data_that_fit['Start'] > test_item['End'].iloc[0]) & (train_data_that_fit['End'] > test_item['End'].iloc[0])).any()) :
                        right_cut_off = train_data_that_fit[((train_data_that_fit['Start'] > test_item['End'].iloc[0]) & (train_data_that_fit['End'] > test_item['End'].iloc[0]))]['Start'].min()
                    else :
                        right_cut_off = max_possible_seg_end
                    
                    right_cut_off = min(right_cut_off, max_possible_seg_end)
                    
                    # 04. get the test data points that are within the cut off dates
                    test_data_that_fit = test_data_this_symbol[(test_data_this_symbol['Start'] >= left_cut_off) & (test_data_this_symbol['End'] <= right_cut_off)] 
                    
                    # if (len(test_data_that_fit) > 1) :
                    #     print("test_data_that_fit")
                        
                    items_that_fit = pd.DataFrame()
                    if(len(test_data_that_fit) > 1):     
                    
                        # 05. iterate the test_data_that_fit and get the number of test items that fit in the segment
                        number_of_test_items_that_fit = {}
                        test_items_left_to_the_selected = test_data_that_fit[test_data_that_fit['Start'] <= test_item['Start'].iloc[0]]
                        for idx, row in test_items_left_to_the_selected.iterrows():
                            # get the number of test items that fit in the segment
                            number_of_test_items_that_fit[row["ID"]] = len(test_data_that_fit[(test_data_that_fit['Start'] >= row['Start']) & (test_data_that_fit['End'] <= row['Start'] + pd.to_timedelta(extra_days + selected_test_item_length.iloc[0].days, unit='D'))])
                    
                        test_items_right_to_the_selected = test_data_that_fit[test_data_that_fit['Start'] > test_item['Start'].iloc[0]]
                        for idx, row in test_items_right_to_the_selected.iterrows():
                            # get the number of test items that fit in the segment
                            number_of_test_items_that_fit[row["ID"]] = len(test_data_that_fit[(test_data_that_fit['End'] <= max(row['End'],test_item['End'].iloc[0])) & (test_data_that_fit['Start'] >= max(row['End'],test_item['End'].iloc[0]) - pd.to_timedelta(extra_days + selected_test_item_length.iloc[0].days, unit='D'))])
                            
                        # 06. get the segment that has the max number of test items that fit
                        # get the key of the max value in the dictionary
                        max_key = max(number_of_test_items_that_fit, key=number_of_test_items_that_fit.get)
                        max_ancor_test_row = test_data_that_fit[test_data_that_fit['ID'] == max_key]
                        
                        if(max_ancor_test_row['Start'].iloc[0]<= test_item['Start'].iloc[0]):
                            items_that_fit = test_data_that_fit[(test_data_that_fit['Start'] >= max_ancor_test_row['Start'].iloc[0]) & (test_data_that_fit['End'] <= max_ancor_test_row['Start'].iloc[0] + pd.to_timedelta(extra_days + selected_test_item_length.iloc[0].days, unit='D'))]
                        elif(max_ancor_test_row['Start'].iloc[0] > test_item['Start'].iloc[0]):
                            items_that_fit = test_data_that_fit[(test_data_that_fit['End'] <= max(max_ancor_test_row['End'].iloc[0],test_item['End'].iloc[0])) & (test_data_that_fit['Start'] >= max(max_ancor_test_row['End'].iloc[0],test_item['End'].iloc[0]) - pd.to_timedelta(extra_days + selected_test_item_length.iloc[0].days, unit='D'))]
                        
                        # drop the item with the id of test_item from the items_that_fit
                        items_that_fit = items_that_fit[items_that_fit['ID'] != test_item['ID'].iloc[0]]
                        # print(items_that_fit)
                        
                    # 07. get the wiggle room by the min start date of the test items that fit in the segment and the max end date of the test items that fit in the segment and try to randomise the start and end dates of the segment
                    if ('items_that_fit' in locals() and not items_that_fit.empty) :
                        max_seg_start_date = min(items_that_fit['Start'].min(),test_item['Start'].iloc[0])
                        min_seg_end_date = max(items_that_fit['End'].max(),test_item['End'].iloc[0])
                    else :
                        max_seg_start_date = test_item['Start'].iloc[0]
                        min_seg_end_date = test_item['End'].iloc[0]
                        
                    # remove the items that selected from index_touple if it is in index_touple
                    for idx, row in items_that_fit.iterrows():
                        if row['ID'] in index_touple:
                            index_touple.remove(row['ID'])
                    
                    
                    # get the wiggle room
                    wiggle_room = (pd.to_timedelta(extra_days, unit='D')+ selected_test_item_length.iloc[0]) - (min_seg_end_date - max_seg_start_date)
                    
                    if (wiggle_room.days < 0) :
                        print("Error")
                    
                    if ( wiggle_room.days!= 0) :
                        random_days = np.random.randint(0, wiggle_room.days)
                    else:
                        random_days = 0
                    seg_start_date = max_seg_start_date - pd.to_timedelta(random_days, unit='D')
                    seg_end_date = min_seg_end_date + pd.to_timedelta(wiggle_room.days - random_days, unit='D')
                    
                    seg_start_date = max(seg_start_date, left_cut_off)
                    seg_end_date = min(seg_end_date, right_cut_off)
                    
                    test_pattern_segments = pd.concat([items_that_fit, test_item], ignore_index=True)  
                    test_pattern_segments['Seg_ID'] = segment_id
                    test_pattern_segments['Seg_Start'] = seg_start_date
                    test_pattern_segments['Seg_End'] = seg_end_date
                    test_pattern_segment_wise = pd.concat([test_pattern_segment_wise, test_pattern_segments], ignore_index=True)
                    segment_id += 1
                    
    return test_pattern_segment_wise


    
colors = ["blue", "green", "red", "cyan", "magenta", "yellow", "purple", "orange", "brown", "pink", "lime", "teal"]

def plot_patterns_for_segment(segment_id, test_pattern_segment_wise ,ohcl_data_given=None,padding_days=0,same_color = False, color_pattern_wise = False ,color_cluster_wise = False, legend = True , seg_alpha = 0.2, probability = None , save = False,name = ""):
    grouped = test_pattern_segment_wise.groupby('Seg_ID')
    group = grouped.get_group(segment_id)
    ohcl_data = pd.DataFrame()
    
    if (ohcl_data_given is None):
        # get ohlc data for the symbol
        symbol = group['Symbol'].iloc[0]
        ohcl_data = pd.read_csv(path + '/' + symbol + '.csv')
    else:
        ohcl_data = ohcl_data_given

    # convert the date columns to datetime
    ohcl_data['Date'] = pd.to_datetime(ohcl_data['Date'])
    ohcl_data['Date'] = ohcl_data['Date'].dt.tz_localize(None)
    
    group['Seg_Start'] = pd.to_datetime(group['Seg_Start'])
    group['Seg_End'] = pd.to_datetime(group['Seg_End'])

    seg_start = group['Seg_Start'].iloc[0]
    seg_end = group['Seg_End'].iloc[0]
    
    # Define the padding range (before and after the segment)
    seg_start = seg_start - pd.to_timedelta(padding_days, unit='D')
    seg_end = seg_end + pd.to_timedelta(padding_days, unit='D')

    # get the ohlc data that is within the segment
    ohcl_data = ohcl_data[(ohcl_data['Date'] >= seg_start) & (ohcl_data['Date'] <= seg_end)]   

    if (ohcl_data.empty == True):
        print("OHLC Data set is empty ")
    else:
        
        # Create a candlestick plot using mplfinance
        ohlc_for_mpf = ohcl_data[['Open', 'High', 'Low', 'Close']].copy()


        # # Create the base plot (this returns a figure and axes)
        # fig, axes = mpf.plot(ohlc_for_mpf, type='candle', style='charles', title=f'OHLC Chart with Patterns',
        #                         ylabel='Price', figsize=(12, 6), returnfig=True)  # Set figsize here

        ohlc_for_mpf.index = pd.to_datetime(ohcl_data['Date'])

        # Re-plot with proper date formatting
        fig, axes = mpf.plot(ohlc_for_mpf, type='candle', style='charles', 
                            datetime_format='%Y-%m-%d', returnfig=True)


        ax = axes[0]  # Access the first (and only) axis object
        # Loop through the patterns and highlight them on the chart
        color_index = 0
        for index, row in group.iterrows():

            
            pattern_start = pd.to_datetime(row['Start']).tz_localize(None)  # Ensure it's a datetime object
            pattern_end = pd.to_datetime(row['End']).tz_localize(None)  # Ensure it's a datetime object
            
            print('Pattern Name : ', row['Chart Pattern'], 'Pattern Start : ', pattern_start, 'Pattern End : ', pattern_end)

            pattern_start_date = pd.to_datetime(row['Start']).tz_localize(None)  # Ensure it's a datetime object
            pattern_end_date = pd.to_datetime(row['End']).tz_localize(None)  # Ensure it's a datetime object



            num_of_OHLC_data_points_from_seg_start_to_pattern_start = len(ohcl_data[ohcl_data['Date'] < pattern_start_date])

            pattern_start = num_of_OHLC_data_points_from_seg_start_to_pattern_start

            num_of_OHLC_data_points_from_pattern_start_to_pattern_end = len(ohcl_data[(ohcl_data['Date'] >= pattern_start_date) & (ohcl_data['Date'] <= pattern_end_date)])

            pattern_end = pattern_start + num_of_OHLC_data_points_from_pattern_start_to_pattern_end

            pattern_lable = row['Chart Pattern']
            
            pattern_encoding = get_pattern_encoding()
            
            if color_pattern_wise:
                color_index = pattern_encoding[pattern_lable]
            if color_cluster_wise:
                color_index = row['Cluster']
            if same_color:
                color_index = 2
            if save and probability is not None :
                seg_alpha = probability/5
                
            color_index = color_index % len(colors)
            
            # Add a vertical span (highlight the pattern) to the chart
            ax.axvspan(pattern_start, pattern_end,color=colors[color_index], alpha=seg_alpha, label=pattern_lable)
            
            if not color_pattern_wise:
                color_index += 1

        # Customize the chart with grid, labels, and legend
        ax.grid(True)
        if probability is not None:
            ax.set_title(f'Probability of being a {pattern_lable} : {probability}')
        if legend:
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Patterns")

        # Show the chart
        plt.show()
        
        if save:
            fig.savefig(f"Samples/{segment_id}_{name}.png")
            
def get_ohlc_data_segment(test_pattern_segment_wise, test_seg_id, path,group):
    seg_id = group['Seg_ID'].iloc[0]

    seg_start = group['Seg_Start'].iloc[0]
    seg_end = group['Seg_End'].iloc[0]

    # Get OHLC data for the symbol
    symbol = group['Symbol'].iloc[0]
    ohcl_data = pd.read_csv(path + '/' + symbol + '.csv')

    # Convert the date column to datetime
    ohcl_data['Date'] = pd.to_datetime(ohcl_data['Date'])
    ohcl_data['Date'] = ohcl_data['Date'].dt.tz_localize(None)

    # Filter out the original data within the segment (without padding)
    ohlc_data_segment = ohcl_data[(ohcl_data['Date'] >= seg_start) & (ohcl_data['Date'] <= seg_end)]

    # normalize the data segment
    ohlc_data_segment = normalize_ohlc_segment(ohlc_data_segment)

    ohlc_data_segment.drop('Volume', axis=1, inplace=True)
    
    return ohlc_data_segment


def process_window(i, ohlc_data_segment, rocket_model, probability_threshold, pattern_encoding_reversed, seg_id, symbol, seg_start, seg_end, test_seg_id, window_size, padding_proportion,len_norm=False,target_len=30):
    start_index = i - math.ceil(window_size * padding_proportion)
    end_index = start_index + window_size

    if start_index < 0:
        start_index = 0
    if end_index > len(ohlc_data_segment):
        end_index = len(ohlc_data_segment)

    ohlc_segment = ohlc_data_segment[start_index:end_index]
    if len(ohlc_segment) == 0:
        return None  # Skip empty segments
    win_start_date = ohlc_segment['Date'].iloc[0]
    win_end_date = ohlc_segment['Date'].iloc[-1]
    
    if len_norm:
        ohlc_segment = normalize_ohlc_len(ohlc_segment ,target_len=target_len)
    # print("ohlc befor :" , ohlc_segment)
    ohlc_array_for_rocket = ohlc_segment[['Open', 'High', 'Low', 'Close']].to_numpy().reshape(1, len(ohlc_segment), 4)
    ohlc_array_for_rocket = np.transpose(ohlc_array_for_rocket, (0, 2, 1))
    # print( "ohlc for rocket :" , ohlc_array_for_rocket)
    try:
        pattern_probabilities = rocket_model.predict_proba(ohlc_array_for_rocket)
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None
    max_probability = np.max(pattern_probabilities)
    # print(pattern_probabilities)
    # print(f"Predicted Pattern: {pattern_encoding_reversed[np.argmax(pattern_probabilities)]} with probability: {max_probability} in num {i} window")
    # if max_probability > probability_threshold:
    pattern_index = np.argmax(pattern_probabilities)
    new_row = {
        'Seg_ID': seg_id, 'Start': win_start_date, 'End': win_end_date, 
        'Symbol': symbol, 'Chart Pattern': pattern_encoding_reversed[pattern_index], 
        'Seg_Start': seg_start, 'Seg_End': seg_end ,'Probability': max_probability
    }
    # plot_patterns_for_segment(test_seg_id, pd.DataFrame([new_row]), ohlc_data_segment)
    return new_row
    # return None



def parallel_process_sliding_window(ohlc_data_segment, rocket_model, probability_threshold, stride, pattern_encoding_reversed, group, test_seg_id, window_size, padding_proportion,len_norm=False,target_len=30):
    seg_id = group['Seg_ID'].iloc[0]
    seg_start = group['Seg_Start'].iloc[0]
    seg_end = group['Seg_End'].iloc[0]
    symbol = group['Symbol'].iloc[0]

    num_cores = 16  # Use all available cores

    # Use Parallel as a context manager to ensure cleanup
    with Parallel(n_jobs=num_cores,verbose = 1) as parallel:
        results = parallel(
            delayed(process_window)(i, ohlc_data_segment, rocket_model, probability_threshold, pattern_encoding_reversed, seg_id, symbol, seg_start, seg_end, test_seg_id, window_size, padding_proportion)
            for i in range(0, len(ohlc_data_segment), stride)
        )

    # print(f"Finished processing segment {seg_id} for symbol {symbol}")
    # print(results)
    # Filter out None values and create DataFrame
    win_results_df = pd.DataFrame([res for res in results if res is not None])
    
    # #  do the sam e thing without parrellel processing
    # results = []
    # for i in range(0, len(ohlc_data_segment), stride):
    #     res = process_window(i, ohlc_data_segment, rocket_model, probability_threshold, pattern_encoding_reversed, seg_id, symbol, seg_start, seg_end, test_seg_id, window_size, padding_proportion)
    #     if res is not None:
    #         results.append(res)
    # win_results_df = pd.DataFrame(results)

    return win_results_df

def plot_sliding_steps(win_results_df, ohlc_data_segment, probability_threshold, test_seg_id,save = False):
    # loop through each row of the win_results_df and add plot  
    for index, row in win_results_df.iterrows():
        print(f"Predicted Pattern: {row['Chart Pattern']} with probability: {row['Probability']} in num {index} window")
        if row['Probability'] > probability_threshold:
            plot_patterns_for_segment(test_seg_id, pd.DataFrame([row]), ohlc_data_segment, color_pattern_wise=True,probability = row['Probability'], save = save,name = f"{index}")
            
def prepare_dataset_for_cluster(ohlc_data_segment, win_results_df):

    predicted_patterns = win_results_df.copy()
    origin_date = ohlc_data_segment['Date'].min()
    for index, row in predicted_patterns.iterrows():
        pattern_start = row['Start']
        pattern_end = row['End']
        
        #  get the number of OHLC data points from the origin date to the pattern start date
        start_point_index = len(ohlc_data_segment[ohlc_data_segment['Date'] < pattern_start])
        pattern_len = len(ohlc_data_segment[(ohlc_data_segment['Date'] >= pattern_start) & (ohlc_data_segment['Date'] <= pattern_end)])
        
        pattern_mid_index = start_point_index + (pattern_len / 2)
        
        # add the center index to a new column Center in the predicted_patterns current row
        predicted_patterns.at[index, 'Center'] = pattern_mid_index
        predicted_patterns.at[index, 'Pattern_Start_pos'] = start_point_index
        predicted_patterns.at[index, 'Pattern_End_pos'] = start_point_index + pattern_len

    return predicted_patterns
    
def cluster_windows(predicted_patterns , probability_threshold, window_size,eps = 0.05 , min_samples = 2):
    df = predicted_patterns.copy()

    # only get the rows that has a probability greater than the probability threshold
    df = df[df['Probability'] > probability_threshold]

    # Initialize a list to store merged clusters from all groups
    cluster_labled_windows = []
    interseced_clusters = []
    
    min_center = df['Center'].min()
    max_center = df['Center'].max()

    # Group by 'Chart Pattern' and apply clustering to each group
    for pattern, group in df.groupby('Chart Pattern'):
        # print (pattern)
        # print(group)
        # Clustering
        centers = group['Center'].values.reshape(-1, 1)
        
        # centers normalization
        if min_center < max_center:  # Avoid division by zero
            norm_centers = (centers - min_center) / (max_center - min_center)
        else:
            # If all values are the same, set to constant (e.g., 0 or 1)
            norm_centers = np.ones_like(centers)
        
        # eps  =window_size/2 + 4
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(norm_centers)
        group['Cluster'] = db.labels_
        
        cluster_labled_windows.append(group)
        
        # Filter out noise (-1) and group by Cluster
        for cluster_id, cluster_group in group[group['Cluster'] != -1].groupby('Cluster'):

            
            expanded_dates = []
            for _, row in cluster_group.iterrows():
                # Print the start and end dates for debugging
                dates = pd.date_range(row["Start"], row["End"])
                expanded_dates.extend(dates)

            # print("Total expanded dates:", len(expanded_dates))


            # Step 2: Count occurrences of each date
            date_counts = pd.Series(expanded_dates).value_counts().sort_index()

            # Step 3: Identify cluster start and end (where at least 2 windows overlap)
            cluster_start = date_counts[date_counts >= 2].index.min()
            cluster_end = date_counts[date_counts >= 2].index.max()
            
            interseced_clusters.append({
                'Seg_ID' : cluster_group['Seg_ID'].iloc[0],
                'Symbol' : cluster_group['Symbol'].iloc[0],
                'Chart Pattern': pattern,
                'Cluster': cluster_id,
                'Start': cluster_start,
                'End': cluster_end,
                'Seg_Start': cluster_group['Seg_Start'].iloc[0],
                'Seg_End': cluster_group['Seg_End'].iloc[0],
                'Avg_Probability': cluster_group['Probability'].mean(),
            })

    if len(cluster_labled_windows) == 0 or len(interseced_clusters) == 0:
        return None,None
    # # Combine all merged clusters into a final DataFrame
    cluster_labled_windows_df = pd.concat(cluster_labled_windows)
    interseced_clusters_df = pd.DataFrame(interseced_clusters)

    # sort by the index 
    cluster_labled_windows_df = cluster_labled_windows_df.sort_index()
    # print(cluster_labled_windows_df)
    # Display the result
    # print(merged_df)
    return cluster_labled_windows_df,interseced_clusters_df