# Requirements Document

## Introduction

This document specifies the requirements for an ASL (American Sign Language) hand pose recognition system developed as part of CMP_6058A coursework. The system recognizes hand poses for ASL letters A-J using MediaPipe to extract 21 hand landmark coordinates from images, then trains and compares multiple machine learning classifiers. The dataset consists of anonymised 256x256 pixel images from student submissions. The system must handle data preprocessing (noise removal for MediaPipe failures, duplicate detection for identical landmark data), implement supervised learning with at least three classifiers (including a custom kNN implementation from scratch), and perform unsupervised clustering analysis.

## Glossary

- **MediaPipe**: Google's framework for building multimodal machine learning pipelines, used here for hand landmark detection via mediapipe.solutions.hands module
- **Landmark_Data**: The 21 3D hand landmark coordinates (x, y, z) returned by MediaPipe for a detected hand, where x and y are normalized coordinates based on image width/height, and z is estimated distance from camera
- **ASL_Sign**: A categorical label representing one of the first 10 letters (A-J) in the American Sign Language alphabet
- **Feature_Vector**: A 63-dimensional vector of floating-point values representing the (x, y, z) coordinates of all 21 hand landmarks, ordered according to MediaPipe's landmark indexing (wrist first)
- **Training_Data**: The preprocessed dataset of landmark data and corresponding ASL sign labels used to train classifiers
- **Classifier**: A machine learning model that predicts ASL sign classes from 63-dimensional landmark vectors
- **Noise_Image**: An image where MediaPipe fails to return valid landmark data; includes images with partial hand cropping, incorrect hand orientation, occlusions, or any condition preventing landmark detection
- **Duplicate_Instance**: Two or more images where MediaPipe returns identical landmark data (exact numerical equality of all 63 coordinates), indicating duplicate images in the dataset
- **kNN_Classifier**: K-Nearest Neighbors classifier that must be implemented from scratch using only Python standard built-in libraries
- **Cross_Validation**: 5-fold cross validation technique used for hyperparameter tuning

## Requirements

### Requirement 1: Data Loading and Exploration

**User Story:** As a developer, I want to load and explore the anonymised ASL hand gesture dataset, so that I can understand the data structure and class distribution before preprocessing.

#### Acceptance Criteria

1. THE Data_Loader SHALL load all 256x256 pixel images from the provided anonymised dataset directory
2. THE Data_Loader SHALL identify and report the number of images per ASL sign class (A-J)
3. THE Data_Loader SHALL report the total number of images in the dataset
4. WHEN loading images, THE Data_Loader SHALL preserve the association between images and their ASL sign labels derived from filenames
5. THE Data_Loader SHALL support the filename convention {ASL-sign}_sample_{number}.jpg (e.g., J_sample_500.jpg)

### Requirement 2: MediaPipe Landmark Extraction

**User Story:** As a developer, I want to extract hand landmarks from images using MediaPipe Hands, so that I can convert raw images into 63-dimensional feature vectors for classification.

#### Acceptance Criteria

1. THE Landmark_Extractor SHALL use mediapipe.solutions.hands module to process each image
2. WHEN MediaPipe successfully detects a hand, THE Landmark_Extractor SHALL extract all 21 landmark coordinates (x, y, z) where x and y are normalized by image dimensions
3. THE Landmark_Extractor SHALL flatten the 21 landmarks into a 63-dimensional feature vector ordered according to MediaPipe's landmark indexing (wrist landmark first)
4. WHEN MediaPipe fails to detect a hand in an image, THE Landmark_Extractor SHALL flag that image as noise for removal
5. THE Landmark_Extractor SHALL report the number of images where landmark extraction failed per class and total

### Requirement 3: Noise Removal

**User Story:** As a developer, I want to remove noisy images from the training data, so that the classifiers are trained only on valid hand pose samples.

#### Acceptance Criteria

1. WHEN MediaPipe fails to return landmark data for an image, THE Noise_Filter SHALL remove that image from the training data
2. THE Noise_Filter SHALL log the count of removed noise images per ASL sign class and total
3. THE Noise_Filter SHALL preserve all images where MediaPipe successfully extracted landmarks
4. IF an image contains partial hand cropping, incorrect hand orientation, or occlusions, THEN THE Noise_Filter SHALL treat it as noise when MediaPipe fails to detect landmarks
5. THE Noise_Filter SHALL report the data distribution before and after noise removal

### Requirement 4: Duplicate Detection and Removal

**User Story:** As a developer, I want to detect and remove duplicate instances from the training data, so that the classifiers are not biased by repeated samples.

#### Acceptance Criteria

1. THE Duplicate_Detector SHALL identify images that produce identical landmark data (exact numerical equality of all 63 coordinates)
2. WHEN two or more images have identical landmark vectors, THE Duplicate_Detector SHALL retain only one instance
3. THE Duplicate_Detector SHALL log the count of removed duplicate instances per class and total
4. THE Duplicate_Detector SHALL compare landmark vectors using exact numerical equality
5. FOR ALL retained instances, THE Duplicate_Detector SHALL ensure no two have identical landmark data

### Requirement 5: Data Preprocessing and Organization

**User Story:** As a developer, I want to preprocess and organize the landmark data in tabular format, so that the classifiers receive consistent input features.

#### Acceptance Criteria

1. THE Preprocessor SHALL organize data in tabular format with columns for instance-id, 63 feature columns, and ASL sign label
2. THE Preprocessor SHALL normalize landmark coordinates if appropriate for the chosen classifiers
3. THE Preprocessor SHALL handle any missing or invalid coordinate values
4. THE Preprocessor SHALL split the data into training and test sets with justified ratios
5. THE Preprocessor SHALL save the cleaned data to a file for further processing
6. THE Preprocessor SHALL encode ASL sign labels (A-J) for classifier training

### Requirement 6: kNN Classifier Implementation (From Scratch)

**User Story:** As a developer, I want to implement a kNN classifier from scratch using only Python standard libraries, so that I can demonstrate understanding of the algorithm.

#### Acceptance Criteria

1. THE kNN_Classifier SHALL be implemented using only Python standard built-in libraries (no NumPy, scikit-learn, or external ML libraries for the core algorithm)
2. THE kNN_Classifier SHALL compute distances between test instances and all training instances
3. THE kNN_Classifier SHALL support configurable k (number of neighbors) as a hyperparameter
4. THE kNN_Classifier SHALL support at least Euclidean distance metric
5. THE kNN_Classifier SHALL predict the class based on majority voting among k nearest neighbors
6. THE kNN_Classifier SHALL be optimized using 5-fold cross validation with multiple k values

### Requirement 7: Decision Tree Classifier

**User Story:** As a developer, I want to train and optimize a Decision Tree classifier, so that I can compare its performance with other classifiers.

#### Acceptance Criteria

1. THE Decision_Tree_Classifier SHALL be trained on the preprocessed landmark feature vectors
2. THE Decision_Tree_Classifier SHALL support hyperparameter tuning (e.g., max_depth, min_samples_split)
3. THE Decision_Tree_Classifier SHALL be optimized using 5-fold cross validation
4. WHEN training completes, THE Decision_Tree_Classifier SHALL report training metrics (accuracy)
5. THE Decision_Tree_Classifier MAY use scikit-learn implementation

### Requirement 8: Third Classifier (Choice)

**User Story:** As a developer, I want to train and optimize a third classifier of my choice, so that I can compare at least three different approaches.

#### Acceptance Criteria

1. THE Third_Classifier SHALL be a supervised learning classifier different from kNN and Decision Tree
2. THE Third_Classifier SHALL be trained on the preprocessed landmark feature vectors
3. THE Third_Classifier SHALL support hyperparameter tuning with at least two hyperparameters
4. THE Third_Classifier SHALL be optimized using 5-fold cross validation
5. WHEN training completes, THE Third_Classifier SHALL report training metrics

### Requirement 9: Classifier Comparison and Evaluation

**User Story:** As a developer, I want to compare and evaluate all classifiers, so that I can identify the best model for ASL hand pose recognition.

#### Acceptance Criteria

1. THE Evaluator SHALL fine-tune at least two hyperparameters per classifier using 5-fold cross validation
2. THE Evaluator SHALL include default hyperparameter values as baseline
3. THE Evaluator SHALL compute accuracy and sensitivity metrics for each model
4. THE Evaluator SHALL select the best model across all hyperparameter settings with justification
5. THE Evaluator SHALL retrain the best model on the entire training set using best hyperparameters
6. THE Evaluator SHALL assess the best model on the held-out test set
7. THE Evaluator SHALL generate confusion matrices for the best models
8. THE Evaluator SHALL create visualizations (tables, plots) showing performance at each hyperparameter setting

### Requirement 10: Clustering Analysis

**User Story:** As a developer, I want to apply clustering algorithms to the data, so that I can compare unsupervised learning results with classifier predictions.

#### Acceptance Criteria

1. THE Clustering_Module SHALL remove class labels before applying clustering algorithms
2. THE Clustering_Module SHALL apply K-means clustering algorithm
3. THE Clustering_Module SHALL apply hierarchical clustering algorithm
4. THE Clustering_Module SHALL analyze and report the effectiveness of each clustering method
5. THE Clustering_Module SHALL compare clustering output with actual class labels
6. THE Clustering_Module SHALL compare clustering output with best classifier predictions
7. THE Clustering_Module SHALL report whether clustering outputs match actual labels and classifier predictions

### Requirement 11: Data Quality Reporting

**User Story:** As a developer, I want comprehensive reports on data quality and preprocessing, so that I can document the dataset characteristics and preprocessing impact.

#### Acceptance Criteria

1. THE Reporter SHALL output statistics on the original dataset size per class and total
2. THE Reporter SHALL output the count of noise images removed (per class and total)
3. THE Reporter SHALL output the count of duplicate instances removed (per class and total)
4. THE Reporter SHALL output the final clean dataset size used for training
5. THE Reporter SHALL visualize the class distribution before and after preprocessing
6. THE Reporter SHALL document the train/test split ratios and instance counts
