# Model Components

## Purpose
TBD - Defines the standard interfaces and components for models in the research pipeline.

## Requirements

### Requirement: Standardized Classifier Interface
Any classifier injected into a TwoStagePipeline SHALL implement an `IClassifier` interface exposing `fit` and `predict_proba` methods.

#### Scenario: Using a probability-based classifier
- **WHEN** the localizer requests predictions for sliding windows
- **THEN** the classifier provides continuous probabilities allowing the localizer to threshold low-confidence windows.

### Requirement: Standardized Localizer Interface
Any localizer injected into a TwoStagePipeline SHALL implement an `ILocalizer` interface exposing a `find_patterns` method.

#### Scenario: Scanning large charts
- **WHEN** the pipeline calls `find_patterns` on a large continuous chart
- **THEN** the localizer processes the chart and returns discrete bounding coordinates for detected patterns.
