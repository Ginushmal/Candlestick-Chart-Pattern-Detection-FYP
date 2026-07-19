import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

import logging

logger = logging.getLogger(__name__)

class DBSCANClusterer:
    """
    Clusters overlapping predicted patterns using DBSCAN and resolves boundaries by intersection.
    """
    def __init__(self, eps_base=4, min_samples=2, min_intersection_count=2):
        self.eps_base = eps_base
        self.min_samples = min_samples
        self.min_intersection_count = min_intersection_count
        
    def prepare_dataset(self, ohlc_data_segment, predicted_patterns):
        df = predicted_patterns.copy()
        
        centers = []
        for index, row in df.iterrows():
            pattern_start = row['Start']
            pattern_end = row['End']
            
            # Find the position index of the start and length of the pattern
            start_point_index = len(ohlc_data_segment[ohlc_data_segment['Date'] < pattern_start])
            pattern_len = len(ohlc_data_segment[(ohlc_data_segment['Date'] >= pattern_start) & (ohlc_data_segment['Date'] <= pattern_end)])
            
            # Calculate center relative index
            pattern_mid_index = start_point_index + (pattern_len / 2)
            centers.append(pattern_mid_index)
            
        df['Center'] = centers
        return df

    def cluster(self, ohlc_data_segment, predicted_patterns, default_window_size=30):
        if predicted_patterns is None or len(predicted_patterns) == 0:
            return []
            
        df = self.prepare_dataset(ohlc_data_segment, predicted_patterns)
        interseced_clusters = []
        
        for pattern, group in df.groupby('Chart Pattern'):
            centers = group['Center'].values.reshape(-1, 1)
            
            win_size = group['Window_Size'].mean() if 'Window_Size' in group.columns else default_window_size
            eps = (win_size / 2) + self.eps_base
            
            db = DBSCAN(eps=eps, min_samples=self.min_samples).fit(centers)
            group = group.copy()
            group['Cluster'] = db.labels_
            
            for cluster_id, cluster_group in group[group['Cluster'] != -1].groupby('Cluster'):
                expanded_dates = []
                for _, row in cluster_group.iterrows():
                    # Expand into individual dates
                    dates = pd.date_range(row["Start"], row["End"])
                    expanded_dates.extend(dates)
                    
                # Count occurrences of each date
                date_counts = pd.Series(expanded_dates).value_counts().sort_index()
                
                # Boundary intersection resolution (at least min_intersection_count overlapping windows)
                valid_dates = date_counts[date_counts >= self.min_intersection_count]
                
                if not valid_dates.empty:
                    cluster_start = valid_dates.index.min()
                    cluster_end = valid_dates.index.max()
                    
                    mean_score = cluster_group['Probability'].mean()
                    
                    interseced_clusters.append({
                        'start': cluster_start,
                        'end': cluster_end,
                        'pattern': pattern,
                        'score': mean_score
                    })
                    
        return interseced_clusters
