from sklearn.decomposition import KernelPCA
from preprocessing.data_loader import DataLoader
from preprocessing.data_processor import DataProcessor
from sklearn.cluster import OPTICS
from sklearn.metrics import adjusted_rand_score
from clustering.kmeans.kmeans import KMeans
import matplotlib.pyplot as plt
import numpy as np

def load_and_preprocess_data(data_loader, data_processor, dataset_name):
    df, labels = data_loader.load_arff_data(dataset_name)
    df = data_processor.preprocess_dataset(df)
    return df, labels

def plot_clustering_results(projected_data, cluster_labels, title, legend=True):

    plt.figure(figsize=(10, 8))
    unique_clusters = np.unique(cluster_labels)
    for i, cluster in enumerate(unique_clusters):
        cluster_points = projected_data[cluster_labels == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i}')

    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    if legend:
        plt.legend()
    plt.grid(True)
    plt.show()

def load_and_preprocess_data(data_loader, data_processor, dataset_name):
    df, labels = data_loader.load_arff_data(dataset_name)
    df = data_processor.preprocess_dataset(df)
    return df, labels

def experiment(dataset_name, kmeans, optics, n_components):
    data_loader = DataLoader()
    data_processor = DataProcessor()

    # Create KernelPCA instance
    pca = KernelPCA(n_components=n_components)

    # Load and preprocess dataset
    df, labels = load_and_preprocess_data(data_loader, data_processor, dataset_name)

    # Apply KernelPCA
    projected_data = pca.fit_transform(df)

    # Visualize PCA subspace (custom plotting function should be adapted if needed)
    # Assuming a placeholder function for plotting
    plot_clustering_results(projected_data, labels, dataset_name)

    max_score = 0
    best_seed = -1
    seeds = [0, 1, 2, 3, 4]

    # Perform KMeans clustering
    for seed in seeds:
        kmeans.seed = seed
        kmeans_labels = kmeans.fit_predict(projected_data)
        score = adjusted_rand_score(labels, kmeans_labels)
        if score > max_score:
            max_score = score
            best_seed = seed

    kmeans.seed = best_seed
    kmeans_labels = kmeans.fit_predict(projected_data)
    plot_clustering_results(projected_data, kmeans_labels, dataset_name)

    print("ARI: " + str(adjusted_rand_score(labels, kmeans_labels)))

    # Perform OPTICS clustering
    optics_labels = optics.fit_predict(projected_data)
    plot_clustering_results(projected_data, optics_labels, dataset_name, legend=False)
    print("ARI: " + str(adjusted_rand_score(labels, optics_labels)))


# Main workflow
def main_sklearn_kernelpca_clustering():

    # SATIMAGE

    # Best models Work3
    kmeans = KMeans(k=6, distance='cosine')

    optics = OPTICS(metric='euclidean',
                    algorithm='brute',
                    min_samples=2,
                    n_jobs=-1)
    
    experiment('satimage', kmeans, optics, 2)

    # SPLICE

    # Best models Work3
    kmeans = KMeans(k=4, distance='euclidean')

    optics = OPTICS(metric='l1',
                    algorithm='brute',
                    min_samples=2,
                    n_jobs=-1)
    
    experiment('splice', kmeans, optics, 133)

    # VOWEL

    kmeans = KMeans(k=6, distance='euclidean')

    optics = OPTICS(metric='euclidean',
                    algorithm='brute',
                    min_samples=4,
                    n_jobs=-1)
    
    experiment('vowel', kmeans, optics, 8)

if __name__ == "__main__":
    main_sklearn_pca_clustering()
