import numpy as np
import pandas as pd
import math
from joblib import Parallel, delayed

import logging

logger = logging.getLogger(__name__)

class MultiWindowSlidingScanner:
    """
    Scans a large OHLC segment using a sliding window approach.
    Supports multiple window sizes.
    """
    def __init__(self, window_sizes, stride, padding_proportion, probability_threshold, pattern_encoding_reversed, n_jobs=-1):
        self.window_sizes = window_sizes if isinstance(window_sizes, list) else [window_sizes]
        self.stride = stride
        self.padding_proportion = padding_proportion
        self.probability_threshold = probability_threshold
        self.pattern_encoding_reversed = pattern_encoding_reversed
        self.n_jobs = n_jobs

    def scan(self, ohlc_data_segment, classifier):
        all_results = []
        for window_size in self.window_sizes:
            results = self._scan_single_window_size(ohlc_data_segment, classifier, window_size)
            all_results.extend(results)
        
        if not all_results:
            return pd.DataFrame()
            
        return pd.DataFrame(all_results)
        
    def _scan_single_window_size(self, ohlc_data_segment, classifier, window_size):
        def process_window(i):
            start_index = i - math.ceil(window_size * self.padding_proportion)
            end_index = start_index + window_size
            
            if start_index < 0:
                start_index = 0
            if end_index > len(ohlc_data_segment):
                end_index = len(ohlc_data_segment)
            
            ohlc_segment = ohlc_data_segment.iloc[start_index:end_index]
            if len(ohlc_segment) == 0:
                return None
                
            win_start_date = ohlc_segment['Date'].iloc[0]
            win_end_date = ohlc_segment['Date'].iloc[-1]
            
            # Prepare for Rocket (shape: 1, length, 4) -> (1, 4, length)
            ohlc_array = ohlc_segment[['Open', 'High', 'Low', 'Close']].to_numpy().reshape(1, len(ohlc_segment), 4)
            ohlc_array = np.transpose(ohlc_array, (0, 2, 1))
            
            pattern_probabilities = classifier.predict_proba(ohlc_array)
            max_probability = np.max(pattern_probabilities)
            
            if max_probability > self.probability_threshold:
                pattern_index = np.argmax(pattern_probabilities)
                return {
                    'Start': win_start_date,
                    'End': win_end_date,
                    'Chart Pattern': self.pattern_encoding_reversed[pattern_index],
                    'Probability': max_probability,
                    'Window_Size': window_size
                }
            return None

        with Parallel(n_jobs=self.n_jobs) as parallel:
            results = parallel(
                delayed(process_window)(i)
                for i in range(0, len(ohlc_data_segment), self.stride)
            )
            
        return [r for r in results if r is not None]
