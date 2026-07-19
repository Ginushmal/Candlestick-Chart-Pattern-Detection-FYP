from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import numpy as np
import pandas as pd

import logging

logger = logging.getLogger(__name__)

@dataclass
class PipelineResultDTO:
    # --- Localization Results (Always populated) ---
    predicted_bounds: List[Dict[str, Any]] 
    ground_truth_bounds: List[Dict[str, Any]] 
    
    # --- Intermediate Classification Results (Populated by 2-stage/3-stage, None for YOLO) ---
    clf_y_true: Optional[np.ndarray] = None
    clf_y_pred_proba: Optional[np.ndarray] = None

class IPipeline(ABC):
    @abstractmethod
    def run(self, dataset: Any) -> PipelineResultDTO:
        """Executes the pipeline and returns the standard DTO."""
        pass
