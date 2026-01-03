from data_loader import DataLoader
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sklearn.cluster as cluster
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class Clustering:
    def __init__(self, k, data, classes):
        self.k = k
        self.data = data
        self.classes = classes

    def k_means_on_xy_coordinates(self):
        data = pd.read_csv(self.data)
        
        x = np.array(data['x'].apply(ast.literal_eval).tolist())
        y = np.array(data['y'].apply(ast.literal_eval).tolist())

        # concatenate x and y arrays -> [[x1, y1], [x2, y2], ...] to [[x1, y1, x2, y2, ...], [x1, y1, x2, y2, ...], ...]
        # shape should be (n_samples, 42) -> (n_samples, 21 * 2)
        training_data = np.concatenate((x, y), axis=1)
        training_data = StandardScaler().fit_transform(training_data)

        kmeans = cluster.KMeans(n_clusters=self.k, n_init=30, random_state=0)
        predicted_labels = kmeans.fit_predict(training_data)
        cluster_centers = kmeans.cluster_centers_

        pca = PCA(n_components = 2, random_state=0)
        pca_data = pca.fit_transform(training_data)
        pca_centroids = pca.transform(cluster_centers)

        clusters = np.unique(predicted_labels) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        cmap = plt.cm.get_cmap("tab10", len(clusters))

        plt.scatter(pca_data[:,0], pca_data[:,1], c=predicted_labels, cmap=cmap, s=10)
        handles = [
            patches.Patch(color=cmap(i), label=f"{self.classes[i]}")
            for i in range(self.k)
        ]
        plt.scatter(pca_centroids[:,0], pca_centroids[:,1], c="black", marker="x", s=100, label='Centroids')
        plt.title('K-Means Clustering on X and Y Coordinates with PCA')
        plt.legend(handles=handles, title="KMeans clusters", fontsize=9)
        # plt.show()

        return training_data, predicted_labels, cluster_centers


    
    def k_means_on_xyz_coordinates(self):
        data = pd.read_csv(self.data)
        landmarks = np.array(data['landmark'].apply(ast.literal_eval).tolist())
        features = landmarks.reshape(landmarks.shape[0], -1)
        kmeans = cluster.KMeans(n_clusters=self.k, n_init=30, random_state=0)
        predicted_labels = kmeans.fit_predict(features)
        cluster_centers = kmeans.cluster_centers_

        # squeeze the cluster centers to 2D
        pca = PCA(n_components = 2, random_state=0)
        pca_landmarks = pca.fit_transform(features)
        pca_centroids = pca.transform(cluster_centers)

        clusters = np.unique(predicted_labels) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        cmap = plt.cm.get_cmap("tab10", len(clusters))

        plt.scatter(pca_landmarks[:,0], pca_landmarks[:,1], c=predicted_labels, cmap=cmap, s=10)
        handles = [
            patches.Patch(color=cmap(i), label=f"{self.classes[i]}")
            for i in range(self.k)
        ]
        plt.scatter(pca_centroids[:,0], pca_centroids[:,1], c="black", marker="x", s=100, label='Centroids')
        plt.title('K-Means Clustering on X, Y and Z Coordinates with PCA')
        plt.legend(handles=handles, title="KMeans clusters", fontsize=9)
        # plt.show()

        return features, predicted_labels, cluster_centers

if __name__ == "__main__":
    dataset_path = 'data/clean_images/*'
    clean_dataset_path = 'data/clean_dataset/data.csv'
    classes = DataLoader(dataset_path).get_class_names()
    labels = DataLoader(dataset_path).load_dataset()
    clustering = Clustering(len(classes), clean_dataset_path, classes)
    # clustering.k_means_on_xy_coordinates()
    # clustering.kmeans_xy()
    # clustering.k_means_on_xyz_coordinates()

