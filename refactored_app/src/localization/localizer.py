import pandas as pd
from typing import List, Dict, Optional, Any
from src.models.base import ILocalizer, IClassifier
from src.localization.scanners import MultiWindowSlidingScanner
from src.localization.clusterers import DBSCANClusterer

import logging

logger = logging.getLogger(__name__)

class Localizer(ILocalizer):
    """
    Implements the ILocalizer interface using a sliding window scanner and DBSCAN clusterer.
    """
    def __init__(self, scanner: MultiWindowSlidingScanner, clusterer: DBSCANClusterer):
        self.scanner = scanner
        self.clusterer = clusterer
        
    def _normalize_ohlc_segment(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Normalizes the OHLC data to [0, 1] range as required by the model."""
        min_low = dataset['Low'].min()
        max_high = dataset['High'].max()
        
        ohlc_columns = ['Open', 'High', 'Low', 'Close']
        dataset_normalized = dataset.copy()
        
        if max_high > min_low:
            dataset_normalized[ohlc_columns] = (dataset_normalized[ohlc_columns] - min_low) / (max_high - min_low)
        else:
            dataset_normalized[ohlc_columns] = 0.5
            
        if 'Volume' in dataset.columns:
            min_volume = dataset['Volume'].min()
            max_volume = dataset['Volume'].max()
            if max_volume > min_volume:
                dataset_normalized['Volume'] = (dataset_normalized['Volume'] - min_volume) / (max_volume - min_volume)
            else:
                dataset_normalized['Volume'] = 0.5
                
        return dataset_normalized
        
    def find_patterns(self, ohlc_segment: pd.DataFrame, classifier: Optional[IClassifier] = None) -> List[Dict[str, Any]]:
        if classifier is None:
            raise ValueError("Classifier must be provided for the sliding window localizer.")
            
        # 1. Normalize segment
        normalized_segment = self._normalize_ohlc_segment(ohlc_segment)
        
        # 2. Scan for patterns
        win_results_df = self.scanner.scan(normalized_segment, classifier)
        
        if win_results_df is None or win_results_df.empty:
            return []
            
        # 3. Cluster and resolve boundaries
        default_window_size = self.scanner.window_sizes[0] if self.scanner.window_sizes else 30
        clusters = self.clusterer.cluster(ohlc_segment, win_results_df, default_window_size)
        
        return clusters
