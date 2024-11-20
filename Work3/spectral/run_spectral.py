import os
from joblib import Parallel, delayed
import pandas as pd
import time
from itertools import product
from sklearn.metrics import silhouette_score, adjusted_rand_score, davies_bouldin_score, normalized_mutual_info_score

from spectral.spectral import spectralAlgorithm
from preprocessing import DataLoader, DataProcessor   


def run_spectral():
    # Initialize DataLoader
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
    eigen_solvers = ['arpack', 'lobpcg', 'amg'] # There are only these three
    affinities = ['nearest_neighbors', 'rbf', 'precomputed', 'precomputed_nearest_neighbors'] # There are more with pairwise_kernels
    n_neighbors_list = [3, 5, 10, 15] # Ignored for 'rbf'. ToDo: How many values?
    assign_labels_list = ['kmeans', 'cluster_qr'] # Only these two are asked
    
    # Additional Parameters

    # Prepare All Parameter Combinations
    parameter_combinations = list(product(
        datasets.keys(), eigen_solvers, affinities, n_neighbors_list, assign_labels_list
    ))

    # Function to Process Each Parameter Combination
    def process_combination(params):
        dataset_name, eigen_solver, affinity, n_neighbor, assign_labels = params

        X = datasets[dataset_name]['df']
        y = datasets[dataset_name]['labels']

        # Run Spectral Algorithm
        total_time = None
        try:  
            start = time.time()
            labels = spectralAlgorithm(X, eigen_solver, affinity, n_neighbor, assign_label, n_jobs=1)
            total_time = time.time() - start
            
        except Exception as e:
            print(f"Error running SpectralAlgorithm on dataset {dataset_name} with eigen solver {eigen_solver}, affinity {affinity}, "
                  f"number of neighbors {n_neighbor}, and assigned label {assign_label}: {e}")
            return None
        
        # Compute Metrics
        try:
            # Exclude noise points (-1) for silhouette and DBI
            mask = labels != -1
            if mask.sum() > 1 and len(set(labels[mask])) > 1:
                silhouette = silhouette_score(X[mask], labels[mask])
                dbi = davies_bouldin_score(X[mask], labels[mask])
            else:
                silhouette = float('nan')
                dbi = float('nan')

            ari = adjusted_rand_score(y, labels)
            #purity = compute_purity(y, labels)
            nmi = normalized_mutual_info_score(y, labels)
        except Exception as e:
            print(f"Error computing metrics for dataset {dataset_name}: {e}")
            silhouette = ari = dbi = nmi = float('nan')

        result_entry = {
            'Dataset': dataset_name,
            'Eigen Solver': eigen_solver,
            'Affinity': affinity,
            'N Neighbor': n_neighbor,
            'Assigned Label': assign_label,
            'Silhouette': silhouette,
            'ARI': ari,
            #'Purity': purity,
            'DBI': dbi,
            'NMI': nmi,
            'Time (s)': total_time
        }
    
        return result_entry
    
    # Run Parameter Combinations in Parallel
    results = Parallel(n_jobs=-1, backend='threading')(
        delayed(process_combination)(params) for params in parameter_combinations
    )

    # Filter out None results (failed combinations)
    print(f"Number of failed combinations (None results): {sum(res is None for res in results)}")
    results = [res for res in results if res is not None]

    # Save results
    results_df = pd.DataFrame(results)
    os.makedirs('results', exist_ok=True)
    results_filename = 'results/spectral_results.csv'
    results_df.to_csv(results_filename, index=False)
    print(f"Results have been saved to '{results_filename}'")