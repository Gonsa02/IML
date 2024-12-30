import time
import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from functools import partial
from itertools import product
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score, cluster
from concurrent.futures import ProcessPoolExecutor, as_completed

from utils import save_xmeans_results
from preprocessing import DataLoader, DataProcessor
from kmeans.xmeans import XMeans

def purity_score(y_pred, y_true):
    """
    Calculate the purity score for the given clustering.

    Args:
        y_true (array-like): Ground truth class labels.
        y_pred (array-like): Cluster labels assigned by the clustering algorithm.
    Returns:
        purity (float): Purity score ranging from 0 to 1.
    """
    # Compute confusion matrix (contingency matrix)
    contingency_matrix = cluster.contingency_matrix(y_true, y_pred)
    
    # Sum the maximum counts for each cluster
    max_counts = np.amax(contingency_matrix, axis=0)
    return np.sum(max_counts) / np.sum(contingency_matrix)


def process_combination(params, datasets):
    dataset_name, k, seed = params

    X = datasets[dataset_name]['df']
    y = datasets[dataset_name]['labels']

    # Run xmeans algorithm
    total_time = None
    try:
        xmeans = XMeans(k, max_iters=100, seed=seed)
        start = time.time()
        predictions = xmeans.fit_predict(X)
        total_time = time.time() - start

        ari = adjusted_rand_score(y, predictions)
        scoef = silhouette_score(X, predictions)
        dbi = davies_bouldin_score(X, predictions)
        purity = purity_score(y, predictions)

    except Exception as e:
        print(f"""Error running xmeans on dataset {dataset_name} with k {k}: {e}""")
        return None

    result_entry = {
        'Dataset': dataset_name,
        'k_max': k,
        'best_k': xmeans.get_best_k(),
        'Seed': seed,
        'ARI': ari,
        'Silhouette': scoef,
        'Purity': purity,
        'DBI': dbi,
        'Time (s)': total_time
    }

    return result_entry


def run_xmeans():
    # Initialize DataLoader and DataProcessor
    data_loader = DataLoader()
    data_processor = DataProcessor()

    # Load Datasets
    df_satimage, labels_satimage = data_loader.load_arff_data('satimage')
    df_splice, labels_splice = data_loader.load_arff_data('splice')
    df_vowel, labels_vowel = data_loader.load_arff_data('vowel')

    # Preprocess Datasets
    df_satimage = data_processor.preprocess_dataset(df_satimage)
    df_splice = data_processor.preprocess_dataset(df_splice)
    df_vowel = data_processor.preprocess_dataset(df_vowel)


    # Organize Datasets Into a Dictionary
    datasets = {
        'satimage': {
            'df': df_satimage,
            'labels': labels_satimage
        },
        'splice': {
            'df': df_splice,
            'labels': labels_splice
        },
        'vowel': {
            'df': df_vowel,
            'labels': labels_vowel
        }
    }

    # Parameter Lists
    
    k_values = {
        'satimage': [4, 8, 16, 32, 64, 128, 256, 512, 1024],
        'splice': [4, 8, 16, 32, 64, 128, 256, 512, 1024],
        'vowel': [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    }

    seeds = [0, 1, 2, 3, 4]

    parameter_combinations = []
    for dataset_name, k_values in k_values.items():
        combinations = product(
            [dataset_name], k_values, seeds
        )
        parameter_combinations.extend(combinations)

    # Load Existing Results
    xmeans_csv_file = 'results/xmeans_results.csv'

    existing_combinations = set()

    parameter_combinations = [
        params for params in parameter_combinations if params not in existing_combinations]

    total_combinations = len(parameter_combinations)
    print(f"Total combinations to run: {total_combinations}")

    process_func = partial(process_combination, datasets=datasets)

    for params in tqdm(parameter_combinations, total=total_combinations, desc='Experiments'):
        try:
            result = process_func(params)
            save_xmeans_results(result, xmeans_csv_file)
        except Exception as e:
            print(f'Combination {params} generated an exception: {e}')

    # Sort CSV
    xmeans_sort_csv()


def xmeans_sort_csv():
    xmeans_csv_file = 'results/xmeans_results.csv'
    df = pd.read_csv(xmeans_csv_file)
    sort_columns = ['Dataset', 'k_max', 'Seed']
    df_sorted = df.sort_values(
        by=sort_columns, ascending=True, ignore_index=True)
    df_sorted.to_csv(xmeans_csv_file, index=False)


if __name__ == "__main__":
    run_xmeans()
