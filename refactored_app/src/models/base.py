from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any

import logging

logger = logging.getLogger(__name__)

class IClassifier(ABC):
    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Trains the model on pre-segmented patterns."""
        pass
        
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns probabilities. Crucial for the Localizer to filter low-confidence windows."""
        pass

class ILocalizer(ABC):
    @abstractmethod
    def find_patterns(self, ohlc_segment: pd.DataFrame, classifier: Optional[IClassifier] = None) -> List[Dict[str, Any]]:
        """
        Scans a large chart and returns boundary coordinates. 
        For a 2-stage pipeline, it uses the injected classifier to score windows.
        For an end-to-end model, the 'classifier' param might be ignored or None.
        Returns a list of dicts, e.g., [{'start': 100, 'end': 150, 'pattern': 'Double Top', 'score': 0.85}]
        """
        pass
