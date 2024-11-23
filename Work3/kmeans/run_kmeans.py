import time
import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from functools import partial
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed

from utils import save_kmeans_results
from preprocessing import DataLoader, DataProcessor
from .kmeans import KMeans


def process_combination(params, datasets):
    dataset_name, k, distance, seed = params

    X = datasets[dataset_name]['df']
    y = datasets[dataset_name]['labels']

    # Run KMeans algorithm
    total_time = None
    try:
        kmeans = KMeans(k=k, distance=distance, seed=seed)
        start = time.time()
        predictions = kmeans.fit_predict(X)
        total_time = time.time() - start
        accuracy = KMeans.compute_accuracy(predictions, y)

    except Exception as e:
        print(f"""Error running KMeans on dataset {dataset_name} with k {k} and distance {
              distance}: {e}""")
        return None

    result_entry = {
        'Dataset': dataset_name,
        'k': k,
        'Distance': distance,
        'Seed': seed,
        'Accuracy': accuracy,
        'Time (s)': total_time
    }

    return result_entry


def run_kmeans():
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
    k_values = np.arange(2, 16)
    distance_metrics = ['euclidean', 'manhattan', 'cosine']
    seeds = [0, 42, 36]

    parameter_combinations = list(product(
        datasets.keys(), k_values, distance_metrics, seeds
    ))

    # Load Existing Results
    try:
        kmeans_csv_file = 'results/kmeans_results.csv'
        kmeans_df = pd.read_csv(kmeans_csv_file)
    except FileNotFoundError:
        kmeans_df = pd.DataFrame()

    # Build Set of Existing Combinations
    if not kmeans_df.empty:
        existing_combinations = set(zip(
            kmeans_df['Dataset'], kmeans_df['K'], kmeans_df['Distance'], kmeans_df['Seed']
        ))
    else:
        existing_combinations = set()

    parameter_combinations = [
        params for params in parameter_combinations if params not in existing_combinations]

    total_combinations = len(parameter_combinations)
    print(f"Total combinations to run: {total_combinations}")

    process_func = partial(process_combination, datasets=datasets)

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_func, params)
                                   : params for params in parameter_combinations}

        for future in tqdm(as_completed(futures), total=total_combinations, desc='Experiments'):
            params = futures[future]
            try:
                result = future.result()
                save_kmeans_results(result, kmeans_csv_file)
            except Exception as e:
                print(f'Combination {params} generated an exception: {e}')

    # Sort CSV
    kmeans_sort_csv()


def kmeans_sort_csv():
    optics_csv_file = 'results/kmeans_results.csv'
    df = pd.read_csv(optics_csv_file)
    sort_columns = ['Dataset', 'k', 'Distance', 'Seed']
    df_sorted = df.sort_values(
        by=sort_columns, ascending=True, ignore_index=True)
    df_sorted.to_csv(optics_csv_file, index=False)


if __name__ == "__main__":
    run_kmeans()
