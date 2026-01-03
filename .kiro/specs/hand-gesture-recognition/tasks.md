# Implementation Plan: ASL Hand Pose Recognition System

## Overview

This implementation plan breaks down the ASL hand pose recognition system (letters A-J) into discrete coding tasks. Each task builds incrementally on previous work. The focus is on Python implementation using MediaPipe for feature extraction, a custom kNN implementation from scratch (using only Python standard libraries), scikit-learn for Decision Tree and third classifier, and clustering analysis.

## Tasks

- [x] 1. Set up project structure and dependencies
  - Create directory structure: `src/`, `tests/`, `data/`, `models/`, `reports/`
  - Create `requirements.txt` with dependencies: mediapipe, numpy, pandas, scikit-learn, opencv-python, matplotlib, seaborn, hypothesis
  - Create `__init__.py` files for package structure
  - Note: kNN implementation must NOT use numpy/scikit-learn for core algorithm
  - _Requirements: 1.1, 2.1, 6.1_
  - **Why:** This establishes the foundation for all subsequent work. A clean project structure is essential for coursework submission organization and demonstrates professional software engineering practices. The dependency setup ensures reproducibility—markers can run your code. The explicit note about kNN restrictions is critical because the coursework specifically requires implementing kNN from scratch to demonstrate algorithm understanding.

- [x] 2. Implement DataLoader component
  - [ ] 2.1 Create `src/data_loader.py` with DataLoader class
    - Implement `__init__` to accept dataset path
    - Implement `parse_filename()` to extract sign and sample_id from {sign}_sample_{number}.jpg format
    - Implement `load_dataset()` to traverse directory and load 256x256 images
    - Implement `get_class_names()` returning ['A', 'B', ..., 'J']
    - Implement `get_total_count()` method
    - Use OpenCV to read images, preserve image-label-id associations
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_
    - **Why:** The coursework requires you to work with the provided anonymised dataset. This component handles the first step of any ML pipeline—getting data into your system. The filename parsing is crucial because labels are embedded in filenames ({id}_{ASL-sign}_{num}.jpg). Without correct label extraction, your entire classification system fails. Reporting class distribution demonstrates data exploration skills expected in the coursework.

  - [ ]* 2.2 Write property test for data loading
    - **Property 1: Data Loading Preserves Image-Label Associations**
    - **Validates: Requirements 1.1, 1.4, 1.5**
    - **Why:** Ensures correctness of the most fundamental operation—if labels get mixed up with wrong images, all subsequent training and evaluation is meaningless. Property testing catches edge cases that unit tests might miss.

- [ ] 3. Implement LandmarkExtractor component
  - [ ] 3.1 Create `src/landmark_extractor.py` with LandmarkExtractor class
    - Initialize mediapipe.solutions.hands with static_image_mode=True
    - Implement `extract()` to process single image and return 63-dim vector or None
    - Implement `extract_batch()` to process multiple images, tracking failures
    - Flatten 21 landmarks x 3 coordinates into feature vector (wrist first)
    - Track and report extraction failures per class
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_
    - **Why:** This is the core feature engineering step required by the coursework. The coursework explicitly states you must use MediaPipe to extract 21 hand landmarks (63 features total: x, y, z per landmark). This transforms raw images into the numerical feature vectors that classifiers can process. Tracking failures is essential because the coursework requires you to report and handle "noise" (images where MediaPipe fails). The 63-dimensional output is the exact format expected for classification.

  - [ ]* 3.2 Write property test for landmark extraction dimensions
    - **Property 2: Landmark Extraction Produces Correct Dimensions**
    - **Validates: Requirements 2.2, 2.3**
    - **Why:** Guarantees the feature vector is always exactly 63 dimensions. Classifiers expect fixed-size input—wrong dimensions cause crashes or silent errors.

- [ ] 4. Implement NoiseFilter component
  - [ ] 4.1 Create `src/noise_filter.py` with NoiseFilter class
    - Implement `filter()` to remove samples where extraction failed
    - Track and return noise counts per ASL sign class (A-J)
    - Preserve all valid samples
    - Generate noise report with before/after distribution
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_
    - **Why:** The coursework explicitly requires noise removal as part of data preprocessing. "Noise" is defined as images where MediaPipe cannot detect hand landmarks (partial hands, wrong orientation, occlusions). Training on invalid data degrades model performance. The per-class reporting is required for the coursework's data quality documentation—you must show how many samples were removed from each class.

  - [ ]* 4.2 Write property test for noise filtering completeness
    - **Property 3: Noise Filtering Completeness**
    - **Validates: Requirements 3.1, 3.3**
    - **Why:** Ensures no invalid samples leak through to training and no valid samples are accidentally removed. Both scenarios corrupt your results.

- [ ] 5. Implement DuplicateDetector component
  - [ ] 5.1 Create `src/duplicate_detector.py` with DuplicateDetector class
    - Implement `find_duplicates()` to identify identical feature vectors
    - Implement `detect_and_remove()` to retain only unique samples
    - Use exact numerical equality for comparison
    - Track and return duplicate counts per ASL sign class
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_
    - **Why:** The coursework explicitly requires duplicate detection and removal. Duplicates in training data cause overfitting and inflate accuracy metrics artificially. The coursework defines duplicates as images producing identical 63-coordinate landmark vectors. Per-class reporting is required for documentation. This demonstrates understanding of data quality issues in ML pipelines.

  - [ ]* 5.2 Write property test for duplicate detection invariant
    - **Property 4: Duplicate Detection Invariant**
    - **Validates: Requirements 4.2, 4.5**
    - **Why:** Guarantees the output contains no duplicates—the core invariant this component must maintain.

- [ ] 6. Checkpoint - Verify data preprocessing pipeline
  - Ensure all tests pass, ask the user if questions arise.
  - Verify DataLoader to DuplicateDetector chain works
  - Generate initial data quality report
  - **Why:** The coursework requires a complete preprocessing pipeline (load → extract → filter noise → remove duplicates). This checkpoint ensures all preprocessing components integrate correctly before moving to classification. Catching integration bugs early saves significant debugging time. The data quality report is a coursework deliverable.

- [ ] 7. Implement Preprocessor component
  - [ ] 7.1 Create `src/preprocessor.py` with Preprocessor class
    - Implement `create_dataframe()` for tabular format
    - Implement `normalize()` (optional based on classifier needs)
    - Implement `split()` for train/test split with justified ratios
    - Implement `encode_labels()` for A-J to 0-9 encoding
    - Implement `save_to_file()` to save cleaned data
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_
    - **Why:** The coursework requires data in tabular format with proper train/test splits. The split ratio must be justified in your report (common choices: 80/20 or 70/30). Label encoding converts 'A'-'J' to numerical values (0-9) that classifiers require. Saving cleaned data enables reproducibility and allows you to skip preprocessing when re-running experiments. Normalization may improve kNN performance (distance-based algorithms are sensitive to feature scales).

  - [ ]* 7.2 Write property test for split ratio preservation
    - **Property 8: Train/Test Split Ratio Preservation**
    - **Validates: Requirements 5.4**
    - **Why:** Ensures the split maintains the specified ratio and doesn't lose samples. Incorrect splits invalidate your evaluation methodology.

- [ ] 8. Implement kNN Classifier FROM SCRATCH
  - [ ] 8.1 Create `src/knn_scratch.py` with KNNClassifierFromScratch class
    - Use ONLY Python standard built-in libraries (math, collections)
    - DO NOT use NumPy or scikit-learn for core algorithm
    - Implement `__init__` with configurable k parameter
    - Implement `fit()` to store training data
    - Implement `_euclidean_distance()` using math.sqrt
    - Implement `_get_neighbors()` to find k nearest neighbors
    - Implement `_majority_vote()` using collections.Counter
    - Implement `predict()` for batch predictions
    - Implement `score()` for accuracy calculation
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_
    - **Why:** THIS IS A CRITICAL COURSEWORK REQUIREMENT. The coursework explicitly states kNN must be implemented from scratch using only Python standard libraries—no NumPy, no scikit-learn. This demonstrates your understanding of the algorithm, not just your ability to call library functions. Using external libraries here will likely result in significant mark deductions. The implementation must include distance calculation, neighbor finding, and majority voting—the core kNN components.

  - [ ]* 8.2 Write property tests for kNN implementation
    - **Property 5: kNN Distance Calculation Correctness**
    - **Property 6: kNN Majority Voting Correctness**
    - **Validates: Requirements 6.2, 6.4, 6.5**
    - **Why:** Validates the correctness of your from-scratch implementation. Since you can't rely on tested library code, these tests prove your algorithm works correctly.

- [ ] 9. Implement Decision Tree Classifier wrapper
  - [ ] 9.1 Create `src/decision_tree.py` with DecisionTreeWrapper class
    - Wrap scikit-learn DecisionTreeClassifier
    - Support hyperparameters: max_depth, min_samples_split
    - Implement fit(), predict(), score(), get_params(), set_params()
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_
    - **Why:** The coursework requires at least three classifiers. Decision Tree is one of the required/recommended classifiers. Unlike kNN, you CAN use scikit-learn here. The wrapper provides a consistent interface across all classifiers, making comparison easier. Hyperparameter support is required for the tuning experiments.

- [ ] 10. Implement Third Classifier (e.g., Random Forest or SVM)
  - [ ] 10.1 Create `src/third_classifier.py` with ThirdClassifierWrapper class
    - Choose classifier different from kNN and Decision Tree
    - Support at least two hyperparameters for tuning
    - Implement fit(), predict(), score(), get_params(), set_params()
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_
    - **Why:** The coursework requires three different classifiers for comparison. This demonstrates breadth of ML knowledge. Good choices include Random Forest (ensemble method), SVM (different decision boundary approach), or Naive Bayes (probabilistic approach). The choice should be justified in your report based on the problem characteristics.

- [ ] 11. Implement CrossValidator component
  - [ ] 11.1 Create `src/cross_validator.py` with CrossValidator class
    - Implement 5-fold cross validation
    - Implement `create_folds()` for stratified fold creation
    - Implement `cross_validate()` to test hyperparameter combinations
    - Return best params, best score, and all results
    - _Requirements: 6.6, 7.3, 8.4, 9.1, 9.2_
    - **Why:** The coursework explicitly requires 5-fold cross validation for hyperparameter tuning. This is the standard methodology for model selection—it prevents overfitting to a single train/test split and provides more reliable performance estimates. Stratified folds ensure each fold has representative class distribution. You must tune at least two hyperparameters per classifier.

  - [ ]* 11.2 Write property test for cross validation fold completeness
    - **Property 7: Cross Validation Fold Completeness**
    - **Validates: Requirements 9.1**
    - **Why:** Ensures all samples are used exactly once for testing across the 5 folds—the fundamental property of k-fold CV.

- [ ] 12. Checkpoint - Verify classifier implementations
  - Ensure all tests pass, ask the user if questions arise.
  - Verify kNN from scratch works correctly
  - Compare kNN scratch with scikit-learn kNN for validation
  - Verify all three classifiers can train and predict
  - **Why:** Critical validation before the expensive hyperparameter tuning phase. Comparing your from-scratch kNN with scikit-learn's implementation validates correctness—they should produce identical results for the same k value. This checkpoint catches classifier bugs before they corrupt your comparison results.

- [ ] 13. Implement Evaluator component
  - [ ] 13.1 Create `src/evaluator.py` with Evaluator class
    - Implement `evaluate()` for accuracy, sensitivity, confusion matrix
    - Implement `compare_classifiers()` for comparison table
    - Implement `generate_report()` for human-readable output
    - Implement `plot_confusion_matrix()` using matplotlib
    - Implement `plot_hyperparameter_performance()` for CV visualization
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8_
    - **Why:** The coursework requires comprehensive evaluation with specific metrics (accuracy, sensitivity) and visualizations (confusion matrices, hyperparameter performance plots). Confusion matrices show per-class performance—essential for understanding which ASL signs are confused with each other. The comparison table is a key coursework deliverable showing all classifiers side-by-side.

  - [ ]* 13.2 Write property test for confusion matrix consistency
    - **Property 10: Confusion Matrix Consistency**
    - **Validates: Requirements 9.7**
    - **Why:** Ensures confusion matrix row sums equal actual class counts—a mathematical invariant that validates your evaluation code.

- [ ] 14. Implement classifier optimization and comparison pipeline
  - [ ] 14.1 Create `src/classifier_comparison.py`
    - Fine-tune kNN with multiple k values using 5-fold CV
    - Fine-tune Decision Tree with max_depth, min_samples_split
    - Fine-tune third classifier with its hyperparameters
    - Include default hyperparameter values as baseline
    - Select best model with justification
    - Retrain best models on entire training set
    - Evaluate on held-out test set
    - Generate comparison visualizations
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8_
    - **Why:** This is the core experimental component of the coursework. You must demonstrate systematic hyperparameter tuning (not just picking arbitrary values), include baselines (default parameters), justify your best model selection, and properly evaluate on held-out test data. The visualizations (tables, plots) are required coursework deliverables that show performance across all hyperparameter settings.

- [ ] 15. Checkpoint - Verify classifier comparison pipeline
  - Ensure all tests pass, ask the user if questions arise.
  - Verify hyperparameter tuning produces valid results
  - Verify best model selection logic
  - **Why:** Ensures the comparison methodology is sound before generating final results. Incorrect tuning or selection logic invalidates your conclusions.

- [ ] 16. Implement ClusteringModule component
  - [ ] 16.1 Create `src/clustering.py` with ClusteringModule class
    - Implement `kmeans_cluster()` using scikit-learn KMeans with k=10
    - Implement `hierarchical_cluster()` using AgglomerativeClustering
    - Implement `analyze_clusters()` to assess cluster quality
    - Implement `compare_with_classifier()` for comparison
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7_
    - **Why:** The coursework requires unsupervised learning analysis alongside supervised classification. You must apply K-means and hierarchical clustering, then compare cluster assignments with actual labels and classifier predictions. This demonstrates understanding of both supervised and unsupervised paradigms. k=10 matches the 10 ASL classes (A-J). The comparison reveals whether natural data clusters align with ASL sign categories.

  - [ ]* 16.2 Write property test for clustering output validity
    - **Property 11: Clustering Output Validity**
    - **Validates: Requirements 10.2, 10.3**
    - **Why:** Ensures clustering produces valid assignments (all samples assigned, correct number of clusters).

- [ ] 17. Implement DataQualityReporter component
  - [ ] 17.1 Create `src/reporter.py` with DataQualityReporter class
    - Implement `generate_report()` for preprocessing statistics
    - Implement `visualize_distribution()` for class distribution plots
    - Implement `create_summary_table()` for tabular summary
    - Document train/test split ratios and counts
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6_
    - **Why:** The coursework requires comprehensive documentation of data preprocessing. You must report: original dataset size, noise removed per class, duplicates removed per class, final clean dataset size, and train/test split details. The visualizations (class distribution before/after preprocessing) are required deliverables. This demonstrates thorough data understanding and preprocessing transparency.

  - [ ]* 17.2 Write property test for reporting count consistency
    - **Property 12: Reporting Count Consistency**
    - **Validates: Requirements 11.1, 11.2, 11.3, 11.4**
    - **Why:** Ensures reported counts are mathematically consistent (original = clean + noise + duplicates).

- [ ] 18. Create main pipeline script
  - [ ] 18.1 Create `src/main.py` to orchestrate full pipeline
    - Load anonymised dataset using DataLoader
    - Extract landmarks using LandmarkExtractor
    - Filter noise using NoiseFilter
    - Remove duplicates using DuplicateDetector
    - Preprocess and split using Preprocessor
    - Train and tune all three classifiers with 5-fold CV
    - Compare classifiers using Evaluator
    - Perform clustering analysis using ClusteringModule
    - Generate quality report using DataQualityReporter
    - Save all results to reports/ directory
    - _Requirements: All_
    - **Why:** This is the entry point that ties everything together. A single script that runs the complete pipeline demonstrates integration and makes your submission easy to evaluate. Markers can run one command to reproduce all your results. Saving outputs to reports/ organizes deliverables for submission.

- [ ] 19. Final checkpoint - End-to-end verification
  - Ensure all tests pass, ask the user if questions arise.
  - Run full pipeline on anonymised dataset
  - Verify all outputs are generated correctly
  - Verify kNN uses only Python standard libraries
  - **Why:** Final validation before submission. Ensures the complete system works end-to-end on the actual coursework dataset. The kNN verification is critical—using external libraries for kNN is a major coursework violation.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- CRITICAL: kNN classifier MUST use only Python standard built-in libraries
- Property tests validate correctness properties using Hypothesis
- Unit tests validate specific examples and edge cases
- Python libraries:
  - For kNN core: ONLY Python standard libraries (math, collections)
  - For other components: mediapipe, numpy, pandas, scikit-learn, opencv-python, matplotlib, seaborn
  - For testing: hypothesis, pytest
