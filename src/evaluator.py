from sklearn.metrics import silhouette_score, silhouette_samples, adjusted_rand_score, normalized_mutual_info_score
from clustering import Clustering
from data_loader import DataLoader
from collections import Counter
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

class Evaluator:
    def __init__(self, dataset_path, clean_dataset_path, labels, classes):
        self.dataset_path = dataset_path
        self.clean_dataset_path = clean_dataset_path
        self.labels = labels
        self.classes = classes

    def evaluate_kmeans_on_xy_coordinates(self):
        clustering = Clustering(len(self.classes), self.clean_dataset_path, self.labels)
        features, predicted_labels, cluster_centers = clustering.k_means_on_xy_coordinates()
        silhouette_avg = silhouette_score(features, predicted_labels)
        cluster_to_class = self.map_cluster_to_classes(self.labels, predicted_labels)

        print(f'Silhouette score: {silhouette_avg}')
        print(f'Cluster to class mapping: {cluster_to_class}')

    def evaluate_kmeans_on_xyz_coordinates(self):
        clustering = Clustering(len(self.classes), self.clean_dataset_path, self.labels)
        features, predicted_labels, cluster_centers = clustering.k_means_on_xyz_coordinates()
        silhouette_avg = silhouette_score(features, predicted_labels)
        # self.plot_silhouette_score(silhouette_avg, features, predicted_labels, cluster_centers)
        print(f'Silhouette score: {silhouette_avg}')

    def map_cluster_to_classes(self, labels, predicted_labels):
        cluster_to_class = {}
        for cluster in np.unique(predicted_labels):
            true_in_cluster = [
                true_label for true_label, predicted_label in zip(self.labels, predicted_labels) if predicted_label == cluster
            ]
            cluster_to_class[cluster] = Counter(true_in_cluster).most_common(1)[0][0]
        return cluster_to_class

    def plot_silhouette_score(self, silhouette_avg, features, predicted_labels, cluster_centers):
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