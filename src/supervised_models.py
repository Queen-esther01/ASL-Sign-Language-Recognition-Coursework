from data_loader import DataLoader
from preprocessor import Preprocessor
import ast
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns

class SupervisedTasks:
	def __init__(self, k, dataset_path, labels, classes):
		self.k = k
		self.dataset_path = dataset_path
		self.labels = labels
		self.classes = classes
		self.training_data = None
	
	def preprocess_data(self):
		training_data, x, y = Preprocessor(self.dataset_path).reshape_data()
		x_train, x_test, y_train, y_test = train_test_split(training_data, self.labels, test_size=0.2, random_state=42)
		x_train = StandardScaler().fit_transform(x_train)
		x_test = StandardScaler().fit_transform(x_test)
		return training_data, x_train, x_test, y_train, y_test

	def train_svm(self, x_train, x_test, y_train, y_test):
		'''Trains the SVM model using the training data and labels'''
		model = SVC(kernel='linear')
		model.fit(x_train, y_train)

		predictions = model.predict(x_test)
		accuracy = accuracy_score(y_test, predictions)
		print(f'SVM Accuracy: {accuracy}')
		return predictions, y_test, accuracy

	def svm_grid_search(self, x_train, x_test, y_train, y_test):
		'''Performs a grid search for the SVM model'''
		parameters = {
			'C': [0.1, 0.5, 1, 5, 10],
			'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
		}
		grid_search = GridSearchCV(SVC(), parameters, cv=5, scoring='accuracy')
		grid_search.fit(x_train, y_train)
		best_model = grid_search.best_estimator_
		predictions = best_model.predict(x_test)
		accuracy = accuracy_score(y_test, predictions)

		# plot the cv results for the different kernels
		results = pd.DataFrame(grid_search.cv_results_)
		plt.figure(figsize=(8,6))
		for kernel in parameters['kernel']:
			subset = results[results['param_kernel'] == kernel]
			plt.plot(
				subset['param_C'],
				subset['mean_test_score'],
				marker='o',
				label=kernel
			)

		plt.xlabel('C (Regularization parameter)')
		plt.ylabel('Mean CV Accuracy')
		plt.title(f'SVM Hyperparameter Tuning (5-fold CV) - Accuracy: {accuracy * 100:.2f}%')
		plt.legend(title='Kernel')
		plt.grid(True)

		# plt.savefig('reports/svm_grid_search.png')
		plt.close()
		return predictions, y_test, accuracy

	def train_decision_tree(self, x_train, x_test, y_train, y_test):
		'''Trains the Decision Tree model using the training data and labels'''
		model = DecisionTreeClassifier(max_depth=10, random_state=42)
		model.fit(x_train, y_train)

		predictions = model.predict(x_test)
		accuracy = accuracy_score(y_test, predictions)
		print(f'Decision Tree Accuracy: {accuracy}')
		return predictions, y_test, accuracy

	def decision_tree_grid_search(self, x_train, x_test, y_train, y_test):
		'''Performs a grid search for the Decision Tree model'''
		parameters = {
			'max_depth': [10, 20, 30, 40, 50],
			'min_samples_split': [2, 5, 10, 15, 20]
		}
		grid_search = GridSearchCV(DecisionTreeClassifier(), parameters, cv=5, scoring='accuracy')
		grid_search.fit(x_train, y_train)
		best_model = grid_search.best_estimator_
		predictions = best_model.predict(x_test)
		accuracy = accuracy_score(y_test, predictions)
		print(f'Best Parameters: {grid_search.best_params_}')
		print(f'Best Score: {grid_search.best_score_}')
		print(f'Best Model: {best_model}')
		print(f'Decision Tree Accuracy: {accuracy * 100:.2f}%')

		# plot the cv results for the different max depths and min samples splits
		results = pd.DataFrame(grid_search.cv_results_)
		plt.figure(figsize=(8,6))
		for max_depth in parameters['max_depth']:
			subset = results[results['param_max_depth'] == max_depth]
			plt.plot(
				subset['param_min_samples_split'],
				subset['mean_test_score'],
				marker='o',
				label=max_depth
			)

		plt.xlabel('Min Samples Split')
		plt.ylabel('Mean CV Accuracy')
		plt.title(f'Decision Tree Hyperparameter Tuning (5-fold CV) - Accuracy: {accuracy * 100:.2f}%')
		plt.legend(title='Max Depth')
		plt.grid(True)

		# plt.savefig('reports/dt_grid_search.png')
		plt.close()

		return predictions, y_test, accuracy

	def get_euclidean_distance(self, point1, point2):
		'''Calculates the Euclidean distance between two points'''
		point1 = np.asarray(point1).ravel()
		point2 = np.asarray(point2).ravel()
		return np.sqrt(np.sum((point1 - point2) ** 2))

	def get_neighbors(self, test_instance, x_train, y_train):
		'''Finds the k nearest neighbors of a test instance'''
		distances = [self.get_euclidean_distance(test_instance, x) for x in x_train]
		neighbors = np.argsort(distances)[:self.k]
		neighbor_labels = [y_train[i] for i in neighbors]
		return neighbor_labels

	def predict_knn_manual(self, test_instance, x_train, y_train):
		'''Predicts the class of a test instance'''
		neighbor_labels = self.get_neighbors(test_instance, x_train, y_train)
		return Counter(neighbor_labels).most_common(1)[0][0]

	def run_knn_manual(self, x_train, x_test, y_train, y_test):
		'''Manual KNN implementation using the training data and labels'''
		# Convert y_train to list for indexing
		y_train_list = list(y_train)
		
		predictions = []
		for test_instance in x_test:
			pred = self.predict_knn_manual(test_instance, x_train, y_train_list)
			predictions.append(pred)

		predictions = np.array(predictions)
		accuracy = accuracy_score(y_test, predictions)
		print(f'KNN Accuracy: {accuracy}')

		return predictions, np.array(y_test), accuracy

	def knn_grid_search(self, x_train, x_test, y_train, y_test):
		'''Performs a grid search for the KNN model'''
		parameters = {
			'n_neighbors': [1, 3, 5, 7, 9],
			'weights': ['uniform', 'distance'],
			'metric': ['euclidean', 'manhattan', 'minkowski'],
			'p': [1, 2, 3, 4, 5]
		}
		grid_search = GridSearchCV(KNeighborsClassifier(), parameters, cv=5, scoring='accuracy')
		grid_search.fit(x_train, y_train)
		best_model = grid_search.best_estimator_
		predictions = best_model.predict(x_test)
		accuracy = accuracy_score(y_test, predictions)
		print(f'Best Parameters: {grid_search.best_params_}')
		print(f'Best Score: {grid_search.best_score_}')
		print(f'Best Model: {best_model}')
		print(f'KNN Accuracy: {accuracy * 100:.2f}%')

		# plot the cv results for the different n_neighbors values
		results = pd.DataFrame(grid_search.cv_results_)
		plt.figure(figsize=(8,6))
		for n in parameters['n_neighbors']:
			subset = results[results['param_n_neighbors'] == n]
			plt.plot(
				subset['param_weights'],
				subset['mean_test_score'],
				marker='o',
				label=f'k={n}'
			)

		plt.xlabel('Weights')
		plt.ylabel('Mean CV Accuracy')
		plt.title(f'KNN Hyperparameter Tuning (5-fold CV) - Accuracy: {accuracy * 100:.2f}%')
		plt.legend(title='n_neighbors')
		plt.grid(True)

		# plt.savefig('reports/knn_grid_search.png')
		plt.close()

		return predictions, y_test, accuracy

	def generate_confusion_matrix(self, svm_result, dt_result, knn_result):
		'''Generates the confusion matrix for the predictions
		Each result is a tuple of (predictions, y_true, accuracy)
		'''
		svm_preds, svm_y_true, svm_acc = svm_result
		dt_preds, dt_y_true, dt_acc = dt_result
		knn_preds, knn_y_true, knn_acc = knn_result

		plt.figure(figsize=(8,6))
		sns.heatmap(confusion_matrix(svm_y_true, svm_preds),annot=True,fmt="d",
				xticklabels=self.classes,yticklabels=self.classes,cmap="Blues")
		plt.title(f"Confusion Matrix: SVM, Accuracy: {svm_acc * 100:.2f}%")
		plt.savefig("reports/svm_cm.png")
		plt.close()

		plt.figure(figsize=(8,6))
		sns.heatmap(confusion_matrix(dt_y_true, dt_preds),annot=True,fmt="d",
				xticklabels=self.classes,yticklabels=self.classes,cmap="Blues")
		plt.title(f"Confusion Matrix: Decision Tree, Accuracy: {dt_acc * 100:.2f}%")
		plt.savefig("reports/dt_cm.png")
		plt.close()

		plt.figure(figsize=(8,6))
		sns.heatmap(confusion_matrix(knn_y_true, knn_preds),annot=True,fmt="d",
				xticklabels=self.classes,yticklabels=self.classes,cmap="Blues")
		plt.title(f"Confusion Matrix: KNN, Accuracy: {knn_acc * 100:.2f}%")
		plt.savefig("reports/knn_cm.png")
		plt.close()

	def plot_best_model_performance(self, svm_result, dt_result, knn_result):
		'''Plots the accuracy comparison of all models'''
		_, _, svm_acc = svm_result
		_, _, dt_acc = dt_result
		_, _, knn_acc = knn_result

		models = ['SVM', 'Decision Tree', 'KNN']
		accuracies = [svm_acc * 100, dt_acc * 100, knn_acc * 100]
		
		# Sort by accuracy to show best and worst
		sorted_data = sorted(zip(models, accuracies), key=lambda x: x[1], reverse=True)
		models_sorted = [x[0] for x in sorted_data]
		accuracies_sorted = [x[1] for x in sorted_data]
		
		# Color: best = green, worst = red, middle = blue
		colors = ['green', 'steelblue', 'red']

		plt.figure(figsize=(8,6))
		bars = plt.bar(models_sorted, accuracies_sorted, color=colors)
		
		# Add accuracy labels on bars
		for bar, acc in zip(bars, accuracies_sorted):
			plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
					f'{acc:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
		
		plt.xlabel('Model')
		plt.ylabel('Accuracy (%)')
		plt.title('Model Performance Comparison (Best to Worst)')
		plt.ylim(0, 105)
		plt.grid(axis='y', alpha=0.3)
		plt.savefig('reports/best_model_performance.png')
		plt.close()
		
		print(f'\nBest Model: {models_sorted[0]} ({accuracies_sorted[0]:.2f}%)')
		print(f'Worst Model: {models_sorted[-1]} ({accuracies_sorted[-1]:.2f}%)')

	def train_best_model(self):
		'''Trains the best model on the entire training set'''
		training_data, _, _ = Preprocessor(self.dataset_path).reshape_data()
		training_data = StandardScaler().fit_transform(training_data)
		best_model = SVC(kernel='rbf', C=10)
		best_model.fit(training_data, self.labels)
		predictions = best_model.predict(training_data)
		accuracy = accuracy_score(self.labels, predictions)
		print(f'Best Model Accuracy on entire training set: {accuracy * 100:.2f}%')

		# plot the confusion matrix
		cm = confusion_matrix(self.labels, predictions, labels=self.classes)
		ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.classes).plot()
		plt.title(f'Best Model Confusion Matrix - Accuracy: {accuracy * 100:.2f}%')
		plt.savefig('reports/best_model_cm.png')
		plt.close()
		return predictions, self.labels, accuracy

if __name__ == "__main__":
    dataset_path = 'data/clean_images/*'
    clean_dataset_path = 'data/clean_dataset/data.csv'
    classes = DataLoader(dataset_path).get_class_names()
    labels = DataLoader(dataset_path).load_dataset()
    supervised_tasks = SupervisedTasks(2, clean_dataset_path, labels, classes)
    
    # Get the shared train/test split
    _, x_train, x_test, y_train, y_test = supervised_tasks.preprocess_data()
    
    # Train all models using the same split
    # svm_result = supervised_tasks.train_svm(x_train, x_test, y_train, y_test)
    # dt_result = supervised_tasks.train_decision_tree(x_train, x_test, y_train, y_test)
    # knn_result = supervised_tasks.run_knn_manual(x_train, x_test, y_train, y_test)
    # supervised_tasks.generate_confusion_matrix(svm_result, dt_result, knn_result)

    # hyperparameter tuning
# svm_grid_search_result = supervised_tasks.svm_grid_search(x_train, x_test, y_train, y_test)
# dt_grid_search_result = supervised_tasks.decision_tree_grid_search(x_train, x_test, y_train, y_test)
# knn_grid_search_result = supervised_tasks.knn_grid_search(x_train, x_test, y_train, y_test)
# supervised_tasks.plot_best_model_performance(svm_grid_search_result, dt_grid_search_result, knn_grid_search_result)

# train the best model on the entire training set
supervised_tasks.train_best_model()