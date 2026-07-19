## Why

The current candlestick chart pattern detection research is scattered across multiple Jupyter Notebooks. This makes it difficult to reproducibly test and compare different classifiers, feature extractors, and localization algorithms against the same datasets. We need a modular, object-oriented pipeline to cleanly orchestrate these experiments and support end-to-end models like YOLO natively.

## What Changes

- Replaces standalone Jupyter Notebooks with a configuration-driven Python project architecture.
- Introduces `PipelineResultDTO` to standardize pipeline outputs.
- Adds Abstract Base Classes (`IClassifier`, `ILocalizer`, `IPipeline`) for dependency injection.
- Unifies the evaluation metrics (Precision, Recall, F1, IOU, MAE) for all model architectures.

## Capabilities

### New Capabilities
- `research-pipeline`: The core pipeline execution framework that orchestrates data extraction, model training, and pattern localization.
- `model-components`: The interface contracts and implementations for classifiers and localizers.
- `unified-evaluator`: The evaluation module that calculates standardized metrics for both intermediate classifications and final localization bounds.

### Modified Capabilities

## Impact

- Existing `.ipynb` files will be kept for reference but their core logic will be extracted into the `src/` directory.
- Requires new Python dependencies for configuration (e.g. PyYAML).
- Simplifies testing new algorithms via configuration files instead of code duplication.
