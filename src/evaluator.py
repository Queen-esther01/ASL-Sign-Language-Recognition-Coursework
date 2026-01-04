from sklearn.metrics import silhouette_score, silhouette_samples, adjusted_rand_score, normalized_mutual_info_score
from clustering import Clustering
from data_loader import DataLoader
from collections import Counter
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

class Evaluator:
    '''Evaluator class for evaluating both the supervised and unsupervised learning models'''

    def __init__(self, dataset_path, clean_dataset_path, labels, classes):
        self.dataset_path = dataset_path
        self.clean_dataset_path = clean_dataset_path
        self.labels = labels
        self.classes = classes

    def kmeans_evaluation(self, features, predicted_labels):
        '''Evaluates the K-Means clustering through silhouette score, adjusted rand index, and normalized mutual information'''

        silhouette_avg = silhouette_score(features, predicted_labels)
        cluster_to_class = self.map_cluster_to_classes(self.labels, predicted_labels)
        ari = adjusted_rand_score(self.labels, predicted_labels)
        nmi = normalized_mutual_info_score(self.labels, predicted_labels)
        df = pd.DataFrame({'Cluster': predicted_labels, 'Class': self.labels})
        table = pd.crosstab(df['Cluster'], df['Class'])
        return table, cluster_to_class, silhouette_avg, ari, nmi

    def evaluate_kmeans_on_xy_coordinates(self):
        '''Evaluates the K-Means clustering on the XY coordinates'''

        clustering = Clustering(len(self.classes), self.clean_dataset_path, self.labels)
        features, predicted_labels, cluster_centers, pca_data, pca_centroids = clustering.k_means_on_xy_coordinates()
        table, cluster_to_class, silhouette_avg, ari, nmi = self.kmeans_evaluation(features, predicted_labels)

        clusters = np.unique(predicted_labels) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        cmap = plt.cm.get_cmap("tab10", len(clusters))

        # Box 1: Visualize the clusters on the PCA plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        ax1.scatter(pca_data[:,0], pca_data[:,1], c=predicted_labels, cmap=cmap, s=10)
        handles = [
            patches.Patch(color=cmap(i), label=f"Cluster {i} - majority class: {cluster_to_class[i]}")
            for i in range(len(self.classes))
        ]
        ax1.scatter(pca_centroids[:,0], pca_centroids[:,1], c="black", marker="x", s=100, label='Centroids')
        ax1.set_title('K-Means Clustering on X and Y Coordinates with PCA')
        ax1.legend(handles=handles, title="KMeans clusters", fontsize=9)
        ax1.set_xlabel("X Coordinate")
        ax1.set_ylabel("Y Coordinate")


        # Box 2: Visualize the cluster-to-class distribution as a heatmap
        # Create heatmap
        im = ax2.imshow(table.values, cmap='Blues', aspect='auto')
        
        # Add colorbar
        cbar = ax2.figure.colorbar(im, ax=ax2)
        cbar.ax.set_ylabel("Count", rotation=-90, va="bottom")
        
        # Set ticks and labels
        ax2.set_xticks(np.arange(len(table.columns)))
        ax2.set_yticks(np.arange(len(table.index)))
        ax2.set_xticklabels(table.columns)
        ax2.set_yticklabels([f"{idx} → {cluster_to_class.get(idx, '?')}" for idx in table.index])
        
        # Rotate x labels for readability
        plt.setp(ax2.get_xticklabels(), rotation=0, ha="center")
        
        # Add text annotations in each cell
        for i in range(len(table.index)):
            for j in range(len(table.columns)):
                value = table.values[i, j]
                # Use white text on dark cells, black on light cells
                text_color = "white" if value > table.values.max() / 2 else "black"
                ax2.text(j, i, value, ha="center", va="center", color=text_color, fontsize=9)
        
        ax2.set_xlabel("True Class")
        ax2.set_ylabel("Cluster (→ Majority Class)")
        ax2.set_title("K-Means on XY: Cluster vs Class Distribution")
        
        plt.tight_layout()
        plt.show()
        
        return table, cluster_to_class

    def evaluate_kmeans_on_xyz_coordinates(self):
        '''Evaluates the K-Means clustering on the XYZ coordinates'''

        clustering = Clustering(len(self.classes), self.clean_dataset_path, self.labels)
        features, predicted_labels, cluster_centers, pca_data, pca_centroids = clustering.k_means_on_xyz_coordinates()
        table, cluster_to_class, silhouette_avg, ari, nmi = self.kmeans_evaluation(features, predicted_labels)

        clusters = np.unique(predicted_labels) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        cmap = plt.cm.get_cmap("tab10", len(clusters))

        # Box 1: Visualize the clusters on the PCA plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        ax1.scatter(pca_data[:,0], pca_data[:,1], c=predicted_labels, cmap=cmap, s=10)
        handles = [
            patches.Patch(color=cmap(i), label=f"Cluster {i} - majority class: {cluster_to_class[i]}")
            for i in range(len(self.classes))
        ]
        ax1.scatter(pca_centroids[:,0], pca_centroids[:,1], c="black", marker="x", s=100, label='Centroids')
        ax1.set_title('K-Means Clustering on X, Y and Z Coordinates with PCA')
        ax1.legend(handles=handles, title="KMeans clusters", fontsize=9)
        ax1.set_xlabel("X Coordinate")
        ax1.set_ylabel("Y Coordinate")

        # Box 2: Visualize the cluster-to-class distribution as a heatmap
        im = ax2.imshow(table.values, cmap='Blues', aspect='auto')
        
        # Add colorbar
        cbar = ax2.figure.colorbar(im, ax=ax2)
        cbar.ax.set_ylabel("Count", rotation=-90, va="bottom")
        
        # Set ticks and labels
        ax2.set_xticks(np.arange(len(table.columns)))
        ax2.set_yticks(np.arange(len(table.index)))
        ax2.set_xticklabels(table.columns)
        ax2.set_yticklabels([f"{idx} → {cluster_to_class.get(idx, '?')}" for idx in table.index])
        plt.setp(ax2.get_xticklabels(), rotation=0, ha="center")
        for i in range(len(table.index)):
            for j in range(len(table.columns)):
                value = table.values[i, j]
                text_color = "white" if value > table.values.max() / 2 else "black"
                ax2.text(j, i, value, ha="center", va="center", color=text_color, fontsize=9)
        ax2.set_xlabel("True Class")
        ax2.set_ylabel("Cluster (→ Majority Class)")
        ax2.set_title("K-Means on XYZ: Cluster vs Class Distribution")
        plt.tight_layout()
        plt.show()

    def map_cluster_to_classes(self, labels, predicted_labels):
        '''Maps the predicted clusters to the true classes'''

        cluster_to_class = {}
        for cluster in np.unique(predicted_labels):
            true_in_cluster = [
                true_label for true_label, predicted_label in zip(self.labels, predicted_labels) if predicted_label == cluster
            ]
            cluster_to_class[cluster] = Counter(true_in_cluster).most_common(1)[0][0]
        return cluster_to_class

    def plot_silhouette_score(self, silhouette_avg, features, predicted_labels, cluster_centers):
        '''Plot the silhouette score for the clusters'''

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(features) + (len(self.classes) + 1) * 10])
        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(features, predicted_labels)

        y_lower = 10
        for i in range(len(self.classes)):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[predicted_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / len(self.classes))
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(predicted_labels.astype(float) / len(self.classes))
        ax2.scatter(
            features[:, 0], features[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
        )

        # Labeling the clusters
        centers = cluster_centers
        # Draw white circles at cluster centers
        ax2.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="o",
            c="white",
            alpha=1,
            s=200,
            edgecolor="k",
        )

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(
            "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
            % len(self.classes),
            fontsize=14,
            fontweight="bold",
        )

        plt.show()

    def plot_cluster_class_heatmap(self, table, cluster_to_class, title="Cluster vs Class Distribution"):
        """
        Visualize cluster-to-class distribution as a heatmap.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred):
        pass

    def plot_hyperparameter_performance(self, y_true, y_pred):
        pass

if __name__ == "__main__":
    dataset_path = 'data/clean_images/*'
    clean_dataset_path = 'data/clean_dataset/data.csv'
    classes = DataLoader(dataset_path).get_class_names() # ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    labels = DataLoader(dataset_path).load_dataset() 
    evaluator = Evaluator(dataset_path, clean_dataset_path, labels, classes)
    evaluator.evaluate_kmeans_on_xy_coordinates()
    evaluator.evaluate_kmeans_on_xyz_coordinates()