import os
import pandas as pd
from sklearn.metrics import silhouette_score, adjusted_rand_score, f1_score, davies_bouldin_score
import time
from tqdm import tqdm

from spectral.spectral import spectralAlgorithm


def run_spectral():
    # Load and preprocess data
    # ...

    # Parameter Lists
    eigen_solvers = ['arpack', 'lobpcg', 'amg'] # There are only these three
    affinities = ['nearest_neighbors', 'rbf', 'precomputed', 'precomputed_nearest_neighbors'] # There are more with pairwise_kernels
    ns_neighbors = [3, 5, 10, 15] # Ignored for 'rbf'. ToDo: How many values?
    assigns_labels = ['kmeans', 'cluster_qr'] # Only these two are asked
    # Other params: Set defaults or make a study?

    # Prepare combinations
    parameter_combinations = []
    for dataset_name in ['SatImage', 'Splice', 'Vowel']:
        for eigen_solver in eigen_solvers:
            for affinity in affinities:
                for n_neighbor in ns_neighbors:
                    for assign_label in assigns_labels:
                        parameter_combinations.append((dataset_name, eigen_solver, affinity, n_neighbor, assign_label))

    # Save results
    results = []

    for dataset_name, eigen_solver, affinity, n_neighbor, assign_label in tqdm(parameter_combinations):
        # ...

        # Obtain labels and track time
        start = time.time()
        labels = spectralAlgorithm(X, eigen_solver, affinity, n_neighbor, assign_label)
        end = time.time()
        total_time = end - start
        
        # Calculate Metrics # ToDo: Choose which metrics to use
        silhouette = silhouette_score
        ari = adjusted_rand_score
        purity = 1
        dbi = davies_bouldin_score
        f_measure = 1

        result_entry = {
            'Dataset': dataset_name,
            'Eigen Solver': eigen_solver,
            'Affinity': affinity,
            'N Neighbor': n_neighbor,
            'Assign Label': assign_label,
            'ARI': ari,
            'Purity': purity,
            'DBI': dbi,
            'F-measure': f_measure,
            'Time (s)': total_time
        }
    
        results.append(result_entry)

    # Save results
    results_df = pd.DataFrame(results)
    os.makedirs('results', exist_ok=True)
    results_filename = 'results/spectral_results.csv'
    results_df.to_csv(results_filename, index=False)
    print(f"Results have been saved to '{results_filename}'")