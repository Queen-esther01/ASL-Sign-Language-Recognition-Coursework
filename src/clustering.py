from data_loader import DataLoader
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
from sklearn.decomposition import PCA

class Clustering:
    def __init__(self, k, data, labelsclear):
        self.k = k
        self.data = data
        self.labels = labels

    def k_means_on_xy_coordinates(self):
        data = pd.read_csv(self.data)
        
        x = data['x'].apply(ast.literal_eval).values
        print(f'x {x.shape} type {type(x)}, x example {x[0]}')
        # flatten x array from [[x1, y1], [x2, y2], ...] to [x1, y1, x2, y2, ...]
        x = np.array([np.array(sample).flatten() for sample in x])
        y = data['y'].apply(ast.literal_eval).values
        y = np.array([np.array(sample).flatten() for sample in y])
        print(f'x {x.shape} type {type(x)}, x example {x[0]}')
        print(f'y {y.shape} type {type(y)}, y example {y[0]}')
        targets = np.array(self.labels)

        # concatenate x and y arrays -> [[x1, y1], [x2, y2], ...] to [[x1, y1, x2, y2, ...], [x1, y1, x2, y2, ...], ...]
        # shape should be (n_samples, 42) -> (n_samples, 21 * 2)
        training_data = np.concatenate((x, y), axis=1)
        print(f'training_data {training_data.shape} type {type(training_data)}, training_data example {training_data[0]}')

        kmeans = cluster.KMeans(n_clusters=self.k, n_init=30)
        labels = kmeans.fit_predict(training_data)
        cluster_centers = kmeans.cluster_centers_
        # print(f'cluster_centers {cluster_centers}')

        # # squeeze the cluster centers to 2D
        pca = PCA(n_components = 2)
        pca_data = pca.fit_transform(training_data)
        pca_centroids = pca.transform(cluster_centers)

        true_labels = data['label'].values
        unique_labels = sorted(set(true_labels))

        # plt.scatter(pca_x[:, 0], pca_x[:, 1], c=labels, cmap='tab10', s=10)
        # plt.scatter(pca_centroids[:, 0], pca_centroids[:, 1], marker='x', c='black', s=100, alpha=0.75)
        # Plot each true sign label separately with a legend
        cmap = plt.cm.get_cmap('tab10', len(unique_labels))
        for i, sign_label in enumerate(unique_labels):
            mask = true_labels == sign_label
            plt.scatter(pca_data[mask, 0], pca_data[mask, 1], 
                       c=[cmap(i)], s=10, label=f'{sign_label}')
        plt.scatter(pca_centroids[:, 0], pca_centroids[:, 1], marker='x', c='black', s=100, alpha=0.75, label='Centroids')
        plt.title('K-Means Clustering on X and Y Coordinates with PCA')
        plt.colorbar(label='Cluster Label')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(loc='best', fontsize=8, markerscale=1.8, labelspacing=2)
        plt.show()

        return pca_data, pca_centroids
    
    def k_means_on_xyz_coordinates(self):
        data = pd.read_csv(self.data)
        # Parse string to list, then flatten each sample into a 1D array
        landmarks = data['landmark'].apply(ast.literal_eval).values
        print(f'landmarks {landmarks.shape} type {type(landmarks)}')
        # Flatten nested lists: [[[x,y,z], ...]] -> [x, y, z, x, y, z, ...]
        features = np.array([np.array(sample).flatten() for sample in landmarks])
        print(f'features {features.shape} type {type(features)}')
        targets = np.array(self.labels)
        print(f'targets {targets.shape}, targets {targets[0]}')
        kmeans = cluster.KMeans(n_clusters=self.k, n_init=30)
        labels = kmeans.fit_predict(features)
        cluster_centers = kmeans.cluster_centers_

        kmeans_predict = kmeans.predict(features)
        print(f'kmeans_predict {kmeans_predict}, {kmeans_predict.shape}')

        # squeeze the cluster centers to 2D
        pca = PCA(n_components = 2)
        pca_landmarks = pca.fit_transform(features)
        pca_centroids = pca.transform(cluster_centers)

        # plt.scatter(pca_landmarks[:, 0], pca_landmarks[:, 1], c=labels, cmap='tab10', s=10)
        # plt.scatter(pca_centroids[:, 0], pca_centroids[:, 1], marker='x', c='black', s=100, alpha=0.75)
        # Get the true labels from CSV
        true_labels = data['label'].values
        unique_labels = sorted(set(true_labels))
        
        # Plot each true sign label separately with a legend
        cmap = plt.cm.get_cmap('tab10', len(unique_labels))
        for i, sign_label in enumerate(unique_labels):
            mask = true_labels == sign_label
            plt.scatter(pca_landmarks[mask, 0], pca_landmarks[mask, 1], 
                       c=[cmap(i)], s=10, label=f'{sign_label}')
        
        plt.scatter(pca_centroids[:, 0], pca_centroids[:, 1], marker='x', c='black', s=100, alpha=0.75, label='Centroids')
        plt.title('K-Means Clustering on X, Y and Z Coordinates with PCA')
        plt.colorbar(label='Cluster Label')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(loc='best', fontsize=8, markerscale=1.8, labelspacing=2)
        plt.show()

        return pca_landmarks, pca_centroids

if __name__ == "__main__":
    dataset_path = 'data/clean_images/*'
    clean_dataset_path = 'data/clean_dataset/data.csv'
    classes = DataLoader(dataset_path).get_class_names()
    labels = DataLoader(dataset_path).load_dataset()
    clustering = Clustering(len(classes), clean_dataset_path, labels)
    clustering.k_means_on_xy_coordinates()
    # clustering.k_means_on_xyz_coordinates()

