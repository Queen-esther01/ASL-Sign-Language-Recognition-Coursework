# ASL Hand Pose Recognition System

A machine learning system for recognizing American Sign Language (ASL) hand poses (letters A-J) using MediaPipe landmark extraction and multiple classifiers.

## Overview

This project implements a complete ML pipeline for ASL hand gesture classification:

1. **Data Loading** - Load 256x256 images from the dataset
2. **Feature Extraction** - Extract 21 hand landmarks (63 features) using MediaPipe
3. **Preprocessing** - Remove noise (failed extractions) and duplicates
4. **Classification** - Train and compare three classifiers:
   - k-Nearest Neighbors (implemented from scratch using only Python standard libraries)
   - Decision Tree (scikit-learn)
   - Random Forest (scikit-learn)
5. **Clustering Analysis** - K-means and hierarchical clustering comparison
6. **Evaluation** - 5-fold cross-validation, hyperparameter tuning, confusion matrices

## Project Structure

```
├── src/
│   ├── data_loader.py          # Load images and parse filenames
│   ├── landmark_extractor.py   # MediaPipe hand landmark extraction
│   ├── preprocessor.py         # Data cleaning and train/test split
│   ├── knn.py                  # kNN classifier (from scratch)
│   ├── decision_tree.py        # Decision Tree wrapper
│   ├── random_forest.py        # Random Forest wrapper
│   ├── cross_validator.py      # 5-fold cross validation
│   ├── evaluator.py            # Metrics and visualization
│   ├── clustering.py           # K-means and hierarchical clustering
│   └── classifier_comparison.py # Full comparison pipeline
├── CW2_dataset_final/          # Dataset (A-J folders with images)
├── reports/                    # Generated reports and visualizations
├── tests/                      # Unit and property tests
└── requirements.txt            # Python dependencies
```

## Dataset

Images follow the naming convention: `{ASL-sign}_sample_{number}.jpg`

Example: `J_sample_500.jpg` → Sign: J, Sample ID: 500

Classes: A, B, C, D, E, F, G, H, I, J (10 ASL letters)

## Installation

### Prerequisites

- Python 3.12+
- uv package manager (recommended) or pip

### Setup with uv

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync
```

### Setup with pip

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Dependencies

- mediapipe - Hand landmark detection
- numpy - Numerical operations
- pandas - Data manipulation
- scikit-learn - Decision Tree, Random Forest, clustering
- matplotlib - Visualization

**Note:** The kNN classifier uses only Python standard libraries (math, collections) as required.

## Usage

```bash
# Run the full pipeline
python -m src.classifier_comparison

# Or run individual components
python -c "from src.data_loader import DataLoader; dl = DataLoader('CW2_dataset_final'); print(dl.get_total_count())"
```

## Key Features

- **kNN from scratch** - Implements Euclidean distance, neighbor finding, and majority voting without NumPy/scikit-learn
- **5-fold cross validation** - Stratified folds for hyperparameter tuning
- **Comprehensive evaluation** - Accuracy, sensitivity, confusion matrices
- **Clustering comparison** - Compares unsupervised clusters with actual labels and classifier predictions

## Coursework

CMP_6058A 2025/26 Coursework 2 - ASL Hand Pose Recognition
