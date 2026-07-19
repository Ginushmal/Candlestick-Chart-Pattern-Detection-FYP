from .base import IPipeline, PipelineResultDTO
from typing import Any

import logging

logger = logging.getLogger(__name__)

class EndToEndPipeline(IPipeline):
    def __init__(self, e2e_model: Any):
        self.e2e_model = e2e_model
        
    def run(self, dataset: Any) -> PipelineResultDTO:
        # YOLO/E2E trains on large bounded charts
        self.e2e_model.fit(dataset.train_segments, dataset.train_ground_truth)
        
        predicted_bounds = self.e2e_model.predict(dataset.large_test_segments)
        
        return PipelineResultDTO(
            predicted_bounds=predicted_bounds,
            ground_truth_bounds=dataset.large_test_ground_truth,
            clf_y_true=None,
            clf_y_pred_proba=None
        )
