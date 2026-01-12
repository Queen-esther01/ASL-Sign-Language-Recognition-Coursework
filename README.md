# ASL Hand Pose Recognition System

A machine learning system for recognizing American Sign Language (ASL) hand poses (letters A-J) using MediaPipe landmark extraction and multiple classifiers.

## Overview

This project implements a complete ML pipeline for ASL hand gesture classification:

1. **Data Loading** - Load 256x256 images from the dataset
2. **Feature Extraction** - Extract 21 hand landmarks (63 features: x, y, z per landmark) using MediaPipe
3. **Preprocessing** - Reshape landmarks into feature vectors, apply StandardScaler normalization
4. **Supervised Classification** - Train and compare three classifiers:
    - Support Vector Machine (SVM)
    - Decision Tree
    - k-Nearest Neighbors (KNN with manual implementation)
5. **Unsupervised Clustering** - K-Means and Agglomerative (Hierarchical) clustering
6. **Evaluation** - 5-fold cross-validation, hyperparameter tuning via GridSearchCV, confusion matrices, silhouette analysis

## Project Structure

```
├── src/
│   ├── data_loader.py          # Load images and parse filenames
│   ├── landmark_extractor.py   # MediaPipe hand landmark extraction
│   ├── preprocessor.py         # Data reshaping and feature preparation
│   ├── supervised_models.py    # SVM, Decision Tree, KNN classifiers
│   ├── unsupervised_models.py  # K-Means and Agglomerative clustering
│   └── evaluator.py            # Clustering metrics and visualization
├── data/
│   ├── CW2_dataset_final/      # Original dataset (A-J folders with images)
│   ├── clean_images/           # Preprocessed images with valid landmarks
│   └── clean_dataset/          # Extracted landmarks CSV
├── models/
│   └── mediapipe/              # MediaPipe hand landmarker model
├── reports/                    # Generated visualizations and charts
└── requirements.txt            # Python dependencies
```

## Dataset

Images follow the naming convention: `{ASL-sign}_sample_{number}.jpg`

Example: `J_sample_500.jpg` → Sign: J, Sample ID: 500

Classes: A, B, C, D, E, F, G, H, I, J (10 ASL letters)

## Installation

### Prerequisites

-   Python 3.12+
-   uv package manager (recommended) or pip

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

-   **mediapipe** - Hand landmark detection
-   **numpy** - Numerical operations
-   **pandas** - Data manipulation
-   **scikit-learn** - SVM, Decision Tree, KNN, clustering, metrics
-   **matplotlib** - Visualization
-   **seaborn** - Enhanced heatmaps and confusion matrices
-   **scipy** - Hierarchical clustering dendrograms
-   **opencv-python** - Image processing

## Usage

### Extract Landmarks from Images

```bash
cd src
python landmark_extractor.py
```

### Run Supervised Learning Models

```bash
cd src
python supervised_models.py
```

This will:

-   Train SVM, Decision Tree, and KNN models
-   Perform hyperparameter tuning via GridSearchCV
-   Generate confusion matrices and performance comparison charts

### Run Unsupervised Clustering

```bash
cd src
python unsupervised_models.py
```

This will:

-   Run K-Means clustering on XY and XYZ coordinates
-   Run Agglomerative clustering with dendrogram visualization

### Evaluate Clustering Performance

```bash
cd src
python evaluator.py
```

This will:

-   Compute silhouette scores, Adjusted Rand Index, Normalized Mutual Information
-   Generate cluster-to-class heatmaps
-   Visualize clusters with PCA

## Key Features

### Supervised Learning

-   **SVM** - Linear, polynomial, RBF, and sigmoid kernels with C parameter tuning
-   **Decision Tree** - Max depth and min samples split tuning
-   **KNN (Manual Implementation)** - Euclidean distance, k-neighbors, majority voting
-   **KNN (sklearn)** - Grid search over n_neighbors, weights, metrics, and p values
-   **5-fold Cross Validation** - GridSearchCV for hyperparameter optimization
-   **Confusion Matrices** - Heatmap visualization for each model

### Unsupervised Learning

-   **K-Means Clustering** - On XY coordinates (42 features) and XYZ coordinates (63 features)
-   **Agglomerative Clustering** - Hierarchical clustering with dendrogram
-   **PCA Visualization** - 2D projection of high-dimensional clusters
-   **Cluster-to-Class Mapping** - Majority voting to map clusters to true labels

### Evaluation Metrics

-   **Silhouette Score** - Cluster separation quality
-   **Adjusted Rand Index** - Clustering agreement with true labels
-   **Normalized Mutual Information** - Information shared between clusters and labels
-   **Accuracy** - Classification performance on test set

## Generated Reports

| File                                 | Description                                 |
| ------------------------------------ | ------------------------------------------- |
| `svm_cm.png`                         | SVM confusion matrix                        |
| `dt_cm.png`                          | Decision Tree confusion matrix              |
| `knn_cm.png`                         | KNN confusion matrix                        |
| `svm_grid_search.png`                | SVM hyperparameter tuning results           |
| `dt_grid_search.png`                 | Decision Tree hyperparameter tuning results |
| `knn_grid_search.png`                | KNN hyperparameter tuning results           |
| `best_model_performance.png`         | Model accuracy comparison chart             |
| `best_model_cm.png`                  | Best model confusion matrix                 |
| `kmeans_on_x_y.png`                  | K-Means on XY coordinates                   |
| `kmeans_on_x_y_z.png`                | K-Means on XYZ coordinates                  |
| `kmeans_hierarchical_clustering.png` | Agglomerative clustering with dendrogram    |

## Coursework

CMP_6058A 2025/26 Coursework 2 - ASL Hand Pose Recognition
