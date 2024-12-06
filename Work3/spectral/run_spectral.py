import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import time
from tqdm import tqdm
from itertools import product
from sklearn.metrics import silhouette_score, adjusted_rand_score, davies_bouldin_score
from functools import partial

from spectral.spectral import spectralAlgorithm
from preprocessing import DataLoader, DataProcessor
from utils import save_spectral_results, purity_score

# Function to Process Each Combination
def process_combination(params, datasets):
    dataset_name, n_neighbors, affinity, eigen_solver, assign_labels, n_clusters, seed = params
    #print(f"Running combination {dataset_name},{n_neighbors},{affinity},{eigen_solver},{assign_labels},{n_clusters}")

    X = datasets[dataset_name]['df']
    y = datasets[dataset_name]['labels']

    # Run Spectral Algorithm
    total_time = None
    try:  
        start = time.time()
        labels = spectralAlgorithm(X, eigen_solver, affinity, n_neighbors, assign_labels, n_clusters, seed, n_jobs=1)
        total_time = time.time() - start
        
    except Exception as e:
        print(f"Error running SpectralAlgorithm on dataset '{dataset_name}' with eigen solver '{eigen_solver}', affinity '{affinity}', "
                f"number of neighbors '{n_neighbors}', assigned label '{assign_labels}', and number of clusters '{n_clusters}': {e}\n")
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
        print(f"Error computing metrics for dataset {dataset_name}: {e}\n")
        silhouette = ari = dbi = purity = 'NA'

    result_entry = {
        'Dataset': dataset_name,
        'N Neighbors': n_neighbors,
        'Affinity': affinity,
        'Eigen Solver': eigen_solver,
        'Assign Labels': assign_labels,
        'N Clusters': n_clusters,
        'Seed': seed,
        'ARI': ari,
        'DBI': dbi,
        'Silhouette': silhouette,
        'Purity': purity,
        'Time (s)': total_time
    }
    #print(f"Finished combination {dataset_name},{n_neighbors},{affinity},{eigen_solver},{assign_labels},{n_clusters}")
    return result_entry

def run_spectral():
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
    eigen_solvers = ['arpack', 'lobpcg', 'amg'] # Only these three exist
    affinities = ['nearest_neighbors', 'rbf'] # Only can use these two
    assign_labels_list = ['kmeans', 'cluster_qr'] # Only these two are asked
    n_neighbors_list = [3, 5, 10, 15, 20, 30, 50] # Ignored for 'rbf'. ToDo: How many values?
    
    # Additional Parameters
    n_clusters_list = list(range(2, 16))
    seeds = [0, 1, 2, 3, 4]

    # Prepare All Parameter Combinations
    parameter_combinations = []
    for dataset, affinity, eigen_solver, assign_labels, n_clusters, seed in product(datasets.keys(), affinities, eigen_solvers, assign_labels_list, n_clusters_list, seeds):
        if affinity == 'rbf':
            parameter_combinations.append((dataset, 1, affinity, eigen_solver, assign_labels, n_clusters, seed))
        else:
            for n_neighbors in n_neighbors_list:
                parameter_combinations.append((dataset, n_neighbors, affinity, eigen_solver, assign_labels, n_clusters, seed))

    # Load Existing Results
    try:
        spectral_csv_file = 'results/spectral_results.csv'
        spectral_df = pd.read_csv(spectral_csv_file)
    except FileNotFoundError:
        spectral_df = pd.DataFrame()
        print(f"{spectral_csv_file} was not found.")

    # Build Set of Existing Combinations
    if not spectral_df.empty:
        existing_combinations = set(zip(
        spectral_df['Dataset'], spectral_df['N Neighbors'], spectral_df['Affinity'], spectral_df['Eigen Solver'],
        spectral_df['Assign Labels'], spectral_df['N Clusters'], spectral_df['Seed']
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
                save_spectral_results(result, spectral_csv_file)
            except Exception as e:
                print(f'Combination {params} generated an exception: {e}')

    # Sort CSV# Sort CSV
    spectral_sort_csv()

def spectral_sort_csv():
    spectral_csv_file = 'results/spectral_results.csv'
    df = pd.read_csv(spectral_csv_file)
    sort_columns = ['Dataset', 'N Neighbors', 'Affinity', 'Eigen Solver', 'Assign Labels', 'N Clusters', 'Seed']
    df_sorted = df.sort_values(by=sort_columns, ascending=True, ignore_index=True)
    df_sorted.to_csv(spectral_csv_file, index=False)

def spectral_sort_and_remove_duplicates_csv():
    """
    Reads the spectral_results.csv file, sorts it based on specified columns,
    removes duplicate rows (keeping the first occurrence), and writes the
    cleaned DataFrame back to the CSV.
    """
    # Path to the CSV file
    spectral_csv_file = 'results/spectral_results.csv'
    
    # Read the CSV into a DataFrame
    df = pd.read_csv(spectral_csv_file)
    
    # Define the columns to sort by
    sort_columns = ['Dataset', 'N Neighbors', 'Affinity', 'Eigen Solver', 'Assign Labels', 'N Clusters', 'Seed']
    
    # Sort the DataFrame by the specified columns
    df_sorted = df.sort_values(by=sort_columns, ascending=True, ignore_index=True)
    
    # Remove duplicate rows based on all sort_columns
    # This ensures that rows with identical values in sort_columns are considered duplicates
    df_unique = df_sorted.drop_duplicates(subset=sort_columns, keep='first')
    
    # Optional: If you want to consider all columns for identifying duplicates,
    # simply use drop_duplicates without the subset parameter
    # df_unique = df_sorted.drop_duplicates(keep='first')
    
    # Write the cleaned DataFrame back to the CSV, overwriting the original file
    df_unique.to_csv(spectral_csv_file, index=False)