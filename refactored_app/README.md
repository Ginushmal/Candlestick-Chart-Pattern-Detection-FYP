# Candlestick Chart Pattern Detection Pipeline

This directory contains the cleanly refactored, modularized, and production-ready version of the Candlestick Chart Pattern Detection research project. The codebase has been redesigned from the original Jupyter Notebooks into a structured Object-Oriented pipeline.

## 🏗️ Architecture Overview

The pipeline leverages the **Strategy Design Pattern** through strictly defined interfaces. This ensures that any piece of the puzzle (how data is loaded, how patterns are classified, how localization boundaries are drawn) can be swapped out easily without rewriting the core execution flow.

### Core Interfaces (`src/models/base.py`)
- **`IClassifier`**: Any classification model must implement `fit(X, y)` and `predict_proba(X)`.
- **`ILocalizer`**: Any pattern locator must implement `find_patterns(ohlc_segment, classifier)`.
- **`IPipeline`**: High-level execution flow that ties everything together. Currently implemented as `TwoStagePipeline` (Extraction/Classification -> Localization) and `EndToEndPipeline` (YOLO-style).

## 📂 Folder Structure

```text
refactored_app/
│
├── configs/                  # YAML configuration files for different experiments
│   └── test_config.yaml      
│
├── src/                      # Source code for the pipeline
│   ├── data/                 # Scraping, loading, augmenting, and preprocessing
│   ├── evaluation/           # IOU, Accuracy, Precision/Recall metrics
│   ├── localization/         # Sliding window scanners and DBSCAN clustering
│   ├── models/               # Classifiers (MiniRocket, MultiRocket, XGBoost)
│   ├── pipelines/            # Pipeline orchestration (Two-Stage, End-to-End)
│   └── utils/                # Configuration parsing and logging helpers
│
├── run_experiment.py         # Main execution script
└── README.md                 # You are here
```

## ⚙️ Configuration Guide

The pipeline is entirely driven by YAML configuration files. You do not need to edit code to change experiment parameters.

### Example Configuration Block Explained
```yaml
pipeline_type: "two_stage" # Options: "two_stage" (Extraction/Classification -> Localization) or "end_to_end" (YOLO-style bounding boxes)

data:
  csv_path: "../Datasets/scraped_blog_tables.csv" # Path to save/load the scraped pattern metadata
  ohlc_dir: "../Datasets/OHLC data"               # Directory to save/load raw Yahoo Finance OHLC data
  scrape:
    start_year: 2024                              # Integer year (e.g., 2020) to start scraping Bulkowski's blog
    end_year: 2024                                # Integer year (e.g., 2024) to end scraping
    months: ["Jan", "Feb"]                        # (Optional) List of specific months to target (e.g., ["Jan", "Feb", "Mar"]). Omit to scrape all months.
  target_patterns:                                # (Optional) List of exact pattern strings to filter the dataset by (e.g., ["Double Top, Adam and Adam", "Triangle, symmetrical"]). Omit to use all scraped patterns.
    - "Double Top, Adam and Adam"
    - "Triangle, symmetrical"
  max_samples_per_pattern: 5                      # (Optional) Integer. Limits the number of instances loaded per pattern class.
  max_total_samples: 10                           # (Optional) Integer. Limits the total dataset size.

classifier:
  name: "minirocket"                              # Options: "minirocket", "multirocket"
  params:
    num_kernels: 100                              # Integer. Model hyperparameter defining the number of random convolutional kernels.

localization:
  window_size: 30                                 # Integer or List[Integer]. Width of the MultiWindowSlidingScanner.
  stride: 5                                       # Integer. Number of steps to jump between sliding windows.
  padding_proportion: 0.2                         # Float (0.0 to 1.0). Proportion of edge padding to add to windows for context.
  probability_threshold: 0.5                      # Float (0.0 to 1.0). Minimum classifier confidence required to trigger a bounding box.
  eps_base: 4                                     # Integer/Float. DBSCAN eps baseline offset for clustering overlapping windows.
```

## 🚀 How to Run the Project

To run an experiment, simply point the `run_experiment.py` script to a configuration file.

1. Ensure your `Datasets/` folder is populated in the root directory (one level up from this folder).
2. Ensure you have installed the requirements: 
   ```bash
   pip install pandas numpy scikit-learn xgboost sktime yfinance numba
   ```
### Standard Execution
To run the full end-to-end pipeline simply point the `run_experiment.py` script to a configuration file:
   ```bash
   python run_experiment.py --config configs/test_config.yaml
   ```
The script will automatically execute the entire pipeline (scraping, downloading, preprocessing, training, and evaluation).

### Granular Execution (--step)
The pipeline is designed to be highly modular. You can bypass the entire pipeline and test specific stages independently using the `--step` flag! This is especially useful for quickly iterating on a model's logic without waiting for web scraping or downloading.

Available steps:
- `--step scrape`: Extracts the patterns from Bulkowski's blog based on your config parameters.
- `--step download`: Parses the CSV and downloads the missing historical OHLC stock data via `yfinance`.
- `--step preprocess`: Width-augments the OHLC sequences, formats them into multi-index DataFrames, and saves the `X_train.csv`/`y_test.csv` splits.
- `--step train_eval`: Bypasses all data ingestion and runs the Training and Evaluation logic directly using the saved splits.

**Example**:
```bash
python run_experiment.py --config configs/test_config.yaml --step train_eval
```

## 🛠️ How to Introduce New Approaches

Because of the interface-driven design, testing a completely new research idea is trivial!

### 1. Adding a New Classifier
If you want to try a new Time-Series Classifier (e.g., InceptionTime, HIVE-COTE):
1. Create a new class in `src/models/classifiers.py`.
2. Inherit from `IClassifier`.
3. Implement the `fit` and `predict_proba` methods.
4. Update `run_experiment.py` to recognize your new config `name`.

### 2. Adding a New Localizer
If you want to try a new localization approach (e.g., peak detection instead of sliding window):
1. Create a new class in `src/localization/localizer.py`.
2. Inherit from `ILocalizer`.
3. Implement `find_patterns(ohlc_segment, classifier)`.
4. Plug it into the pipeline initialization in `run_experiment.py`.

### 3. Adding a YOLO-style End-to-End Model
1. Implement your model wrapper inheriting from `IPipeline`.
2. Initialize it in `run_experiment.py` under the `pipeline_type: "end_to_end"` branch.
3. Pass it to the `EndToEndPipeline` class which will evaluate bounding boxes directly without intermediate classification metrics.
