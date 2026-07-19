import numpy as np
from ..pipelines.base import PipelineResultDTO
from sklearn.metrics import accuracy_score

import logging

logger = logging.getLogger(__name__)

class Evaluator:
    @staticmethod
    def calculate_iou(box1, box2):
        start_inter = max(box1['start'], box2['start'])
        end_inter = min(box1['end'], box2['end'])
        inter = max(0, end_inter - start_inter)
        union = (box1['end'] - box1['start']) + (box2['end'] - box2['start']) - inter
        return inter / union if union > 0 else 0

    def evaluate(self, result: PipelineResultDTO) -> dict:
        metrics = {}
        
        if result.clf_y_true is not None and result.clf_y_pred_proba is not None:
            preds = np.argmax(result.clf_y_pred_proba, axis=1)
            metrics['classification_accuracy'] = accuracy_score(result.clf_y_true, preds)
            
        matched_ious = []
        for pred in result.predicted_bounds:
            best_iou = 0
            for gt in result.ground_truth_bounds:
                if pred.get('pattern') == gt.get('pattern'):
                    iou = self.calculate_iou(pred, gt)
                    if iou > best_iou:
                        best_iou = iou
            if best_iou > 0.65:
                matched_ious.append(best_iou)
                
        metrics['localization_recall'] = len(matched_ious) / len(result.ground_truth_bounds) if result.ground_truth_bounds else 0
        metrics['localization_precision'] = len(matched_ious) / len(result.predicted_bounds) if result.predicted_bounds else 0
        metrics['avg_iou'] = np.mean(matched_ious) if matched_ious else 0
        
        return metrics
