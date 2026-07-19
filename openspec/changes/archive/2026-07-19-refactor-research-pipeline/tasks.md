## 1. Scaffold Core Architecture

- [x] 1.1 Create `src/` directory structure (`data/`, `models/`, `pipelines/`, `localization/`, `evaluation/`).
- [x] 1.2 Define abstract base classes `IClassifier`, `ILocalizer`, `IPipeline` in `src/models/base.py`.
- [x] 1.3 Define `PipelineResultDTO` in `src/pipelines/base.py`.
- [x] 1.4 Setup configuration parser for YAML files.

## 2. Component Logic Extraction (via Subagents)

- [x] 2.1 Delegate data extraction logic (from `01` and `02` notebooks) to a subagent and write to `src/data/`.
- [x] 2.2 Delegate classification logic (MiniRocket/MultiRocket/XGBoost from `03.x` notebooks) to a subagent and write to `src/models/`.
- [x] 2.3 Delegate localization logic (Sliding Window/DBSCAN from `04.2` notebook) to a subagent and write to `src/localization/`.

## 3. Pipeline Construction

- [x] 3.1 Implement `TwoStagePipeline` class to weave `IClassifier` and `ILocalizer`.
- [x] 3.2 Implement `EndToEndPipeline` class to handle single-model bounds output.

## 4. Unified Evaluation

- [x] 4.1 Extract metric calculations (IOU, MAE, Accuracy) and build the Evaluator module in `src/evaluation/`.
- [x] 4.2 Verify evaluator works correctly for both intermediate (Classification) and final (Localization) DTO outputs.

## 5. End-to-End Validation

- [x] 5.1 Run the refactored MiniRocket+XGBoost pipeline via configuration on the existing datasets.
- [x] 5.2 Compare the output metrics against the original notebook results to ensure parity.
