## ADDED Requirements

### Requirement: Configuration-driven pipeline execution
The pipeline SHALL read experiment configurations (e.g., from a YAML file) and dynamically instantiate the correct data pipeline, feature extractors, classifiers, and localization algorithms.

#### Scenario: Running a defined experiment
- **WHEN** the user executes the main runner with a configuration file specifying MiniRocket and DBSCAN
- **THEN** the pipeline instantiates a TwoStagePipeline with those components and processes the data end-to-end.

### Requirement: Unified DTO for Pipeline Results
The pipeline SHALL output results using a unified Data Transfer Object (DTO) containing ground truth bounds, predicted bounds, and optional intermediate classification results.

#### Scenario: Emitting results from a two-stage pipeline
- **WHEN** a TwoStagePipeline completes execution
- **THEN** it returns a DTO populated with both localized bounds and intermediate classification probabilities.
