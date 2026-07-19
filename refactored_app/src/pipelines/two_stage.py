from .base import IPipeline, PipelineResultDTO
from ..models.base import IClassifier, ILocalizer
from typing import Any

import logging

logger = logging.getLogger(__name__)

class TwoStagePipeline(IPipeline):
    def __init__(self, classifier: IClassifier, localizer: ILocalizer):
        self.classifier = classifier
        self.localizer = localizer
        
    def run(self, dataset: Any) -> PipelineResultDTO:
        # 1. Train classifier
        self.classifier.fit(dataset.X_train, dataset.y_train)
        
        # 2. Get intermediate probabilities
        clf_probs = self.classifier.predict_proba(dataset.X_test_cropped)
        
        # 3. Localize on large segments
        predicted_bounds = self.localizer.find_patterns(dataset.large_test_segments, self.classifier)
        
        return PipelineResultDTO(
            predicted_bounds=predicted_bounds,
            ground_truth_bounds=dataset.large_test_ground_truth,
            clf_y_true=dataset.y_test_cropped,
            clf_y_pred_proba=clf_probs
        )
