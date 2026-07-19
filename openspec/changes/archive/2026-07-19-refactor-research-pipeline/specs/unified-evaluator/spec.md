## ADDED Requirements

### Requirement: Conditional Classification Evaluation
The unified evaluator SHALL evaluate intermediate classification metrics only if the pipeline output DTO contains classification predictions.

#### Scenario: Evaluating a two-stage pipeline
- **WHEN** the evaluator receives a DTO from a TwoStagePipeline with classification probabilities
- **THEN** it calculates intermediate classification metrics like accuracy and log-loss alongside localization metrics.

#### Scenario: Evaluating an end-to-end pipeline
- **WHEN** the evaluator receives a DTO from an EndToEndPipeline (like YOLO) with null classification probabilities
- **THEN** it safely skips classification metrics and only calculates localization metrics.

### Requirement: Standardized Localization Metrics
The evaluator SHALL always calculate localization metrics such as Intersection Over Union (IOU) and Mean Absolute Error (MAE) for all pipelines.

#### Scenario: Evaluating final bounds
- **WHEN** the evaluator processes the predicted and ground truth bounds
- **THEN** it outputs a unified scorecard (e.g., CSV or console) with standardized IOU, Precision, and Recall scores.
