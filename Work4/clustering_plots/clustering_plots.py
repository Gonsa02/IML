import time
import numpy as np
import pandas as pd
from umap import UMAP
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import adjusted_rand_score

from PCA import PCA
from preprocessing import DataLoader, DataProcessor
from clustering.kmeans import kmeans, global_kmeans
from clustering.optics import optics

def load_data():
    data_loader = DataLoader()
    data_processor = DataProcessor()

    datasets_info = {
            'satimage': data_loader.load_arff_data('satimage'),
            'splice': data_loader.load_arff_data('splice'),
            'vowel': data_loader.load_arff_data('vowel')
        }

    preprocessed_datasets = {}
    for dataset_name, (df, labels) in datasets_info.items():
        preprocessed_df = data_processor.preprocess_dataset(df)
        
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)

        preprocessed_datasets[dataset_name] = {
            'df': preprocessed_df,
            'labels': encoded_labels
        }
    return preprocessed_datasets


def compute_kmeans(data, k, distance, seed):
    km = kmeans.KMeans(k=k, distance=distance, seed=seed)
    return km.fit_predict(data) 


def compute_optics(data, metric, algorithm, min_samples, n_jobs=1):
    # returns labels from optics
    return optics.opticsAlgorithm(data, metric, algorithm, min_samples, n_jobs)


def plot_umap_2d(df, labels, title, show_legend=True):
    sns.set_style('white')

    fig = plt.figure(figsize=(8, 6))

    plot_df = pd.DataFrame(df, columns=['UMAP Dim 1', 'UMAP Dim 2'])
    plot_df['Cluster'] = labels.astype(int)
    plot_df['Cluster'] = pd.Categorical(plot_df['Cluster'], categories=sorted(plot_df['Cluster'].unique()), ordered=True)

    ax = fig.add_subplot(111)
    scatter = ax.scatter(plot_df['UMAP Dim 1'], plot_df['UMAP Dim 2'],
                        c=plot_df['Cluster'], cmap='tab10', s=10, alpha=0.8)
    
    if show_legend:
        legend = ax.legend(*scatter.legend_elements(),
                        title="Clusters",
                        loc="best")
        ax.add_artist(legend)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel('UMAP Dim 1')
    ax.set_ylabel('UMAP Dim 2')

    plt.tight_layout()
    plt.show()


def plot_UMAP(df, labels, title, show_legend=True):
    n_neighbors = 15
    um = UMAP(n_neighbors=n_neighbors, min_dist=0.1, n_components=2, random_state=0)
    embedding = um.fit_transform(df)

    plot_umap_2d(embedding, labels, title, show_legend)


def plot_PCA(data, labels, dataset_name, show_legend=True):
    reducer = PCA.imlPCA()
    reduced_data = reducer.fit_transform(data)
    reducer.plot_pca_subspace(reduced_data, labels, dataset_name[0].upper()+dataset_name[1:], legend=show_legend)


def run_experiments():
    preprocessed_datasets = load_data()
    seeds = [0, 1, 2, 3, 4]

    # ------- Original Data -------
    for dataset_name, dataset_content in preprocessed_datasets.items():
        data = dataset_content['df']
        labels = np.array(dataset_content['labels']).astype(int)

        plot_PCA(data, labels, dataset_name)
        plot_UMAP(data, labels, f"{dataset_name} original dataset")


    # ------- Clustering Original Features -------
    for dataset_name, dataset_content in preprocessed_datasets.items():
        data = dataset_content['df']
        labels = np.array(dataset_content['labels']).astype(int)

        # KMeans
        max_ari = -1
        best_pred = None

        for seed in seeds:
            if dataset_name == 'satimage':
                y_pred = compute_kmeans(data, 6, 'cosine', seed)
            elif dataset_name == 'splice':
                y_pred = compute_kmeans(data, 4, 'euclidean', seed)
            else:
                y_pred = compute_kmeans(data, 6, 'euclidean', seed)
            
            ari_score = adjusted_rand_score(labels, y_pred)
            if ari_score > max_ari:
                max_ari = ari_score
                best_pred = y_pred

        print(f"{dataset_name} got ARI score = {max_ari}")

        plot_PCA(data, best_pred, dataset_name)
        plot_UMAP(data, best_pred, f"{dataset_name} KMeans clustering")

        # OPTICS
        if dataset_name == 'satimage':
            y_pred_optics = compute_optics(data, "euclidean", "brute", 2)
        elif dataset_name == 'splice':
            y_pred_optics = compute_optics(data, "l1", "brute", 2)
        else:
            y_pred_optics = compute_optics(data, "euclidean", "brute", 4)
        
        ari_score_optics = adjusted_rand_score(labels, y_pred_optics)
        print(f"{dataset_name} got ARI score = {ari_score_optics}")

        plot_PCA(data, y_pred_optics, dataset_name, show_legend=False)
        plot_UMAP(data, y_pred_optics, f"{dataset_name} OPTICS clustering", show_legend=False)


def umap_reduce_and_cluster():
    preprocessed_datasets = load_data()
    seeds = [0, 1, 2, 3, 4]
    results = []

    # ------- Clustering Reduced Features -------
    for dataset_name, dataset_content in preprocessed_datasets.items():
        data = dataset_content['df']
        labels = np.array(dataset_content['labels']).astype(int)

        best_kmeans_ari_score = -1
        best_kmeans_reduced_data = None
        best_kmeans_y_pred = None

        best_optics_ari_score = -1
        best_optics_reduced_data = None
        best_optics_y_pred = None

        total_time = 0

        for seed in seeds:
            start = time.time()
            reducer = UMAP(n_components=2, random_state=seed)
            reduced_data = reducer.fit_transform(data)
            end = time.time()
            total_time += (end - start)

            # KMeans Clustering
            for cluster_seed in seeds:
                if dataset_name == 'satimage':
                    kmeans_y_pred = compute_kmeans(reduced_data, 6, 'cosine', cluster_seed)
                elif dataset_name == 'splice':
                    kmeans_y_pred = compute_kmeans(reduced_data, 4, 'euclidean', cluster_seed)
                else:
                    kmeans_y_pred = compute_kmeans(reduced_data, 6, 'euclidean', cluster_seed)

                kmeans_ari_score = adjusted_rand_score(labels, kmeans_y_pred)
                if kmeans_ari_score > best_kmeans_ari_score:
                    best_kmeans_ari_score = kmeans_ari_score
                    best_kmeans_reduced_data = reduced_data
                    best_kmeans_y_pred = kmeans_y_pred

            # OPTICS Clustering
            if dataset_name == 'satimage':
                optics_y_pred = compute_optics(reduced_data, "euclidean", "brute", 2)
            elif dataset_name == 'splice':
                optics_y_pred = compute_optics(reduced_data, "l1", "brute", 2)
            else:
                optics_y_pred = compute_optics(reduced_data, "euclidean", "brute", 4)

            optics_ari_score = adjusted_rand_score(labels, optics_y_pred)
            if optics_ari_score > best_optics_ari_score:
                best_optics_ari_score = optics_ari_score
                best_optics_reduced_data = reduced_data
                best_optics_y_pred = optics_y_pred

        print(f"{dataset_name} got ARI score for KMEANS = {best_kmeans_ari_score}")
        print(f"{dataset_name} got ARI score for OPTICS = {best_optics_ari_score}")
        mean_time = total_time / len(seeds)

        results.append({
            'dataset': dataset_name,
            'clustering': 'kmeans',
            'best_ari_score': best_kmeans_ari_score,
            'mean_time': mean_time
        })

        results.append({
            'dataset': dataset_name,
            'clustering': 'optics',
            'best_ari_score': best_optics_ari_score,
            'mean_time': mean_time
        })

        plot_UMAP(data, best_kmeans_y_pred, f"{dataset_name} KMeans clustering")
        plot_UMAP(data, best_optics_y_pred, f"{dataset_name} OPTICS clustering", show_legend=False)

    results_df = pd.DataFrame(results)
    results_df.to_csv('umap_clustering.csv', index=False)

def pca_reduce_and_cluster():
    preprocessed_datasets = load_data()
    seeds = [0, 1, 2, 3, 4]
    results = []

    # ------- Clustering Original Features -------
    for dataset_name, dataset_content in preprocessed_datasets.items():
        data = dataset_content['df']
        labels = np.array(dataset_content['labels']).astype(int)

        best_kmeans_ari_score = -1
        best_kmeans_reduced_data = None
        best_kmeans_y_pred = None

        total_time = 0.0

        start = time.time()
        reducer = PCA.imlPCA()
        reduced_data = reducer.fit_transform(data)
        end = time.time()
        total_time += (end - start)

        # KMeans Clustering
        for cluster_seed in seeds:
            if dataset_name == 'satimage':
                kmeans_y_pred = compute_kmeans(reduced_data, 6, 'cosine', cluster_seed)
            elif dataset_name == 'splice':
                kmeans_y_pred = compute_kmeans(reduced_data, 4, 'euclidean', cluster_seed)
            else:
                kmeans_y_pred = compute_kmeans(reduced_data, 6, 'euclidean', cluster_seed)

            kmeans_ari_score = adjusted_rand_score(labels, kmeans_y_pred)
            if kmeans_ari_score > best_kmeans_ari_score:
                best_kmeans_ari_score = kmeans_ari_score
                best_kmeans_reduced_data = reduced_data
                best_kmeans_y_pred = kmeans_y_pred

        # OPTICS Clustering
        if dataset_name == 'satimage':
            optics_y_pred = compute_optics(reduced_data, "euclidean", "brute", 2)
        elif dataset_name == 'splice':
            optics_y_pred = compute_optics(reduced_data, "l1", "brute", 2)
        else:
            optics_y_pred = compute_optics(reduced_data, "euclidean", "brute", 4)

        optics_ari_score = adjusted_rand_score(labels, optics_y_pred)

        print(f"{dataset_name} got ARI score for KMEANS = {best_kmeans_ari_score}")
        print(f"{dataset_name} got ARI score for OPTICS = {optics_ari_score}")
        mean_time = total_time / len(seeds)

        results.append({
            'dataset': dataset_name,
            'clustering': 'kmeans',
            'best_ari_score': best_kmeans_ari_score,
            'mean_time': mean_time
        })

        results.append({
            'dataset': dataset_name,
            'clustering': 'optics',
            'best_ari_score': optics_ari_score,
            'mean_time': mean_time
        })

        plot_PCA(data, best_kmeans_y_pred, dataset_name)
        plot_PCA(data, optics_y_pred, dataset_name, show_legend=False)

    results_df = pd.DataFrame(results)
    results_df.to_csv('pca_clustering.csv', index=False)


        
if __name__ == '__main__':
    run_experiments()
    umap_reduce_and_cluster()
    pca_reduce_and_cluster()