import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import time
from tqdm import tqdm
from itertools import product
from sklearn.metrics import silhouette_score, adjusted_rand_score, davies_bouldin_score
from functools import partial

from optics.optics import opticsAlgorithm
from preprocessing import DataLoader, DataProcessor
from utils import save_optics_results, purity_score

# Function to Process Each Combination
def process_combination(params, datasets):
    dataset_name, metric, algorithm, min_samples = params

    X = datasets[dataset_name]['df']
    y = datasets[dataset_name]['labels']

    # Run OPTICS Algorithm
    total_time = None
    try:  
        start = time.time()
        labels = opticsAlgorithm(X, metric, algorithm, min_samples, n_jobs=1)
        total_time = time.time() - start

        # Compute Number of Clusters (excluding noise)
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        
    except Exception as e:
        print(f"Error running OPTICS on dataset '{dataset_name}' with metric '{metric}', algorithm '{algorithm}', and min_samples '{min_samples}': {e}")
        return None

    # Compute Metrics
    try:
        # Exclude noise points (-1) for silhouette and DBI
        mask = labels != -1
        if mask.sum() > 1 and len(set(labels[mask])) > 1:
            silhouette = silhouette_score(X[mask], labels[mask])
            dbi = davies_bouldin_score(X[mask], labels[mask])
        else:
            silhouette = 'NA'
            dbi = 'NA'

        ari = adjusted_rand_score(y, labels)
        purity = purity_score(y, labels)

    except Exception as e:
        print(f"Error computing metrics for dataset {dataset_name}: {e}")
        silhouette = ari = dbi = purity = n_clusters = 'NA'

    result_entry = {
        'Dataset': dataset_name,
        'Metric': metric,
        'Algorithm': algorithm,
        'Min Samples': min_samples,
        'Silhouette': silhouette,
        'ARI': ari,
        'DBI': dbi,
        'Purity': purity,
        'Num Clusters': n_clusters,
        'Time (s)': total_time
    }

    return result_entry

def run_optics():
    # Initialize DataLoader and DataProcessor
    data_loader = DataLoader()
    data_processor = DataProcessor()

    # Load Datasets
    df_satimage, labels_satimage = data_loader.load_arff_data('satimage')
    df_splice, labels_splice = data_loader.load_arff_data('splice')
    df_vowel, labels_vowel = data_loader.load_arff_data('vowel')

    # Preprocess Datasets
    df_satimage = data_processor.preprocess_dataset(df_satimage)
    df_splice   = data_processor.preprocess_dataset(df_splice)
    df_vowel    = data_processor.preprocess_dataset(df_vowel)

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
    metrics = ['euclidean', 'manhattan', 'chebyshev', 'l1']
    algorithms = ['ball_tree', 'kd_tree', 'brute']
    
    # Additional Parameters
    min_samples_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    # Prepare All Parameter Combinations
    parameter_combinations = list(product(
        datasets.keys(), metrics, algorithms, min_samples_list
    ))

    # Load Existing Results
    try:
        optics_csv_file = 'results/optics_results.csv'
        optics_df = pd.read_csv(optics_csv_file)
    except FileNotFoundError:
        optics_df = pd.DataFrame()

    # Build Set of Existing Combinations
    if not optics_df.empty:
        existing_combinations = set(zip(
        optics_df['Dataset'], optics_df['Metric'], optics_df['Algorithm'], optics_df['Min Samples']
        ))
    else:
        existing_combinations = set()

    # Filter Out Already Processed Combinations
    parameter_combinations = [params for params in parameter_combinations if params not in existing_combinations]

    total_combinations = len(parameter_combinations)
    print(f"Total combinations to run: {total_combinations}")

    # Prepare the partial function with datasets
    process_func = partial(process_combination, datasets=datasets)
    
    # Run Parameter Combinations in Parallel
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_func, params): params for params in parameter_combinations}

        for future in tqdm(as_completed(futures), total=total_combinations, desc='Experiments'):
            params = futures[future]
            try:
                result = future.result()
                # Save Result to CSV file
                save_optics_results(result, optics_csv_file)
            except Exception as e:
                print(f'Combination {params} generated an exception: {e}')

    # Sort CSV
    optics_sort_csv()

def optics_sort_csv():
    optics_csv_file = 'results/optics_results.csv'
    df = pd.read_csv(optics_csv_file)
    sort_columns = ['Dataset', 'Metric', 'Algorithm', 'Min Samples']
    df_sorted = df.sort_values(by=sort_columns, ascending=True, ignore_index=True)
    df_sorted.to_csv(optics_csv_file, index=False)

def check_sparse():
    # Initialize DataLoader and DataProcessor
    data_loader = DataLoader()
    data_processor = DataProcessor()

    # Load Datasets
    df_satimage, labels_satimage = data_loader.load_arff_data('satimage')
    df_splice, labels_splice = data_loader.load_arff_data('splice')
    df_vowel, labels_vowel = data_loader.load_arff_data('vowel')

    # Preprocess Datasets
    df_satimage = data_processor.preprocess_dataset(df_satimage)
    df_splice   = data_processor.preprocess_dataset(df_splice)
    df_vowel    = data_processor.preprocess_dataset(df_vowel)

    X = df_satimage.values
    Y = df_splice.values
    Z = df_vowel.values
    from scipy.sparse import csr_matrix
    X_sparse = csr_matrix(X)
    Y_sparse = csr_matrix(Y)
    Z_sparse = csr_matrix(Z)
    print('satimage is ', type(X_sparse))
    print('splice is ', type(Y_sparse))
    print('vowel is ', type(Z_sparse))
