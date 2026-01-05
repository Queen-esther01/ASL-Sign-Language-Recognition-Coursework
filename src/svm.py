from data_loader import DataLoader
from preprocessor import Preprocessor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

class SVM:
    '''SVM class for training and evaluating the SVM model'''

    def __init__(self, data, labels, classes):
        self.data = data
        self.labels = labels
        self.classes = classes

    def train_svm(self):
        '''Trains the SVM model using the training data and labels'''
        training_data = Preprocessor(self.data).reshape_data()
        training_data = StandardScaler().fit_transform(training_data)

        # split the data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(training_data, self.labels, test_size=0.2, random_state=42)

        model = SVC(kernel='linear')

        # train the model
        model.fit(x_train, y_train)

        # make predictions
        predictions = model.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f'Accuracy: {accuracy}')

        # plot the confusion matrix
        cm = confusion_matrix(y_test, predictions, labels=self.classes)
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.classes).plot()
        plt.show()
        


if __name__ == "__main__":
    dataset_path = 'data/clean_images/*'
    clean_dataset_path = 'data/clean_dataset/data.csv'
    classes = DataLoader(dataset_path).get_class_names()
    labels = DataLoader(dataset_path).load_dataset()
    svm = SVM(clean_dataset_path, labels, classes)
    svm.train_svm()