## Context

Currently, the codebase consists of several exploratory Jupyter notebooks (`01`, `02`, `03.1`, `03.2`, `03.3`, `04.2`) detailing a machine learning pipeline for Candlestick Chart Pattern Detection. These notebooks are difficult to compose and reuse for comparing different combinations (e.g. YOLO vs MiniRocket+XGBoost). The goal is to refactor the logic from these notebooks into a strict Object-Oriented pipeline.

## Goals / Non-Goals

**Goals:**
- Port the exact logic from the notebooks into a modular Python architecture inside a new `src/` directory.
- Introduce `IPipeline`, `IClassifier`, and `ILocalizer` interfaces.
- Standardize the input and output (via `PipelineResultDTO`) for evaluation.
- Enable configuration-driven experiments (e.g., via YAML files).
- Keep the original logic exactly intact.

**Non-Goals:**
- Introducing net-new machine learning logic or algorithms.
- Changing the existing metrics (we will just centralize their calculation).

## Decisions

**1. Dependency Injection via Interfaces**
We will define strict abstract base classes. `TwoStagePipeline` will accept an `IClassifier` and an `ILocalizer`. `EndToEndPipeline` will take a unified model. This allows testing a `MultiWindowLocalizer` with a `HiveCoteClassifier` simply by injecting them, without touching the pipeline code.

**2. Unified DTO for Evaluation**
Since two-stage pipelines output intermediate classification probabilities and YOLO-like pipelines do not, we use a single `PipelineResultDTO` where intermediate fields are `Optional`. The single Evaluator module will skip null intermediate fields, ensuring metrics are calculated consistently across all approaches.

**3. Implementation via Context-Aware Subagents**
To avoid context overload during implementation, the extraction of specific notebook logic (Data prep, Classification, Localization) will be delegated to subagents. The orchestrating agent will provide them with extremely detailed prompts (including required context from the research paper) to ensure they extract the code faithfully. The main agent will double-check their work.

## Risks / Trade-offs

- **Risk:** Subagents may hallucinate missing code when extracting from notebooks.
  **Mitigation:** The main orchestrating agent will review all extracted code and subagent prompts will be heavily detailed with context.
- **Risk:** Tightly coupled notebook variables make extraction hard.
  **Mitigation:** Subagents will be instructed to encapsulate global state into class members or function parameters.
