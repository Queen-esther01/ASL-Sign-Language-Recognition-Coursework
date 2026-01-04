from data_loader import DataLoader
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sklearn.cluster as cluster
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from preprocessor import Preprocessor
from scipy.cluster.hierarchy import dendrogram

class Clustering:
    def __init__(self, k, data, classes):
        self.k = k
        self.data = data
        self.classes = classes

    def k_means_on_xy_coordinates(self):
        training_data = Preprocessor(self.data).reshape_data()
        training_data = StandardScaler().fit_transform(training_data)

        kmeans = cluster.KMeans(n_clusters=self.k, n_init=30, random_state=0)
        predicted_labels = kmeans.fit_predict(training_data)
        cluster_centers = kmeans.cluster_centers_

        pca = PCA(n_components = 2, random_state=0)
        pca_data = pca.fit_transform(training_data)
        pca_centroids = pca.transform(cluster_centers)

        return training_data, predicted_labels, cluster_centers, pca_data, pca_centroids

    
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

        return features, predicted_labels, cluster_centers, pca_landmarks, pca_centroids

    def agglomerative_clustering(self):
        training_data = Preprocessor(self.data).reshape_data()
        training_data = StandardScaler().fit_transform(training_data)

        clustering = cluster.AgglomerativeClustering(n_clusters=self.k)
        predicted_labels = clustering.fit_predict(training_data)

        agg = cluster.AgglomerativeClustering(distance_threshold=0, n_clusters=None, compute_distances=True)
        agg.fit(training_data)

        pca = PCA(n_components = 2, random_state=0)
        pca_data = pca.fit_transform(training_data)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        cmap = plt.cm.get_cmap("tab10", len(np.unique(predicted_labels)))
        ax1.scatter(pca_data[:, 0], pca_data[:, 1], c=predicted_labels, cmap=cmap, s=10)
        handles = [
            patches.Patch(color=cmap(i), label=f"Cluster {i}")
            for i in range(len(np.unique(predicted_labels)))
        ]
        ax1.legend(handles=handles, title="Agglomerative Clusters", fontsize=9)
        ax1.set_title("Agglomerative Clustering")
        ax1.set_xlabel("X Coordinate")
        ax1.set_ylabel("Y Coordinate")

        plt.sca(ax2)
        self.plot_dendrogram(agg, truncate_mode='level', p=5)
        plt.title("Hierarchical Clustering Dendrogram")
        plt.xlabel("Sample index")
        plt.ylabel("Distance")

        plt.tight_layout()
        plt.show()

    
    def plot_dendrogram(self, model, **kwargs):
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)

        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack(
            [model.children_, model.distances_, counts]).astype(float)
        dendrogram(linkage_matrix, **kwargs)


if __name__ == "__main__":
    dataset_path = 'data/clean_images/*'
    clean_dataset_path = 'data/clean_dataset/data.csv'
    classes = DataLoader(dataset_path).get_class_names()
    labels = DataLoader(dataset_path).load_dataset()
    clustering = Clustering(len(classes), clean_dataset_path, classes)
    clustering.k_means_on_xy_coordinates()
    clustering.k_means_on_xyz_coordinates()
    clustering.agglomerative_clustering()
    

