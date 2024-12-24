from sklearn.decomposition import KernelPCA
from preprocessing.data_loader import DataLoader
from preprocessing.data_processor import DataProcessor
from sklearn.cluster import KMeans, OPTICS
import numpy as np
import matplotlib.pyplot as plt

def load_and_preprocess_data(data_loader, data_processor, dataset_name):
    df, labels = data_loader.load_arff_data(dataset_name)
    df = data_processor.preprocess_dataset(df)
    return df, labels

# Function to plot clustering results
def plot_clustering_results(projected_data, cluster_labels, title):
    plt.figure(figsize=(10, 7))
    unique_clusters = np.unique(cluster_labels)
    for cluster in unique_clusters:
        cluster_points = projected_data[cluster_labels == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}')

    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True)
    plt.show()

# Main workflow
def main():
    data_loader = DataLoader()
    data_processor = DataProcessor()
    pca = KernelPCA()

    # Load and preprocess dataset
    df_satimage, labels_satimage = load_and_preprocess_data(data_loader, data_processor, 'satimage')

    # Apply PCA
    projected_data = pca.fit_transform(df_satimage)

    plot_clustering_results(projected_data, labels_satimage, 'Labels on PCA Projected Data')

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=6)
    kmeans.fit(projected_data)

    # Get cluster labels
    kmeans_labels = kmeans.labels_

    # Plot clustering results
    plot_clustering_results(projected_data, kmeans_labels, 'KMeans Clusters on PCA Projected Data')

    optics = OPTICS()
    optics.fit(projected_data)
    optics_labels = optics.labels_
    plot_clustering_results(projected_data, optics_labels, 'OPTICS Clusters on PCA Projected Data')

if __name__ == "__main__":
    main()