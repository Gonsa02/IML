import os
import pandas as pd
from sklearn.metrics import silhouette_score, adjusted_rand_score, f1_score, davies_bouldin_score
import time
from tqdm import tqdm

from optics.optics import opticsAlgorithm


def run_optics():
    # Load and preprocess data
    # ...

    # Parameter Lists
    metrics = ['euclidean', 'cosine', 'l1'] # 'euclidean', 'cosine', 'l1', 'l2', 'manhattan', 'cityblock'. There are more with scipy.spatial.distance
    algorithms = ['ball_tree', 'kd_tree']   # There exists 'auto', 'ball_tree', 'kd_tree', and 'brute'.
    # Other params: Set defaults or make a study?

    # Prepare combinations
    parameter_combinations = []
    for dataset_name in ['SatImage', 'Splice', 'Vowel']:
        for metric in metrics:
            for algorithm in algorithms:
                parameter_combinations.append((dataset_name, metric, algorithm))

    # Save results
    results = []

    for dataset_name, metric, algorithm in tqdm(parameter_combinations):
        # ...

        # Obtain labels and track time
        start = time.time()
        labels = opticsAlgorithm(X, metric, algorithm)
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
            'Metric': metric,
            'Algorithm': algorithm,
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
    results_filename = 'results/optics_results.csv'
    results_df.to_csv(results_filename, index=False)
    print(f"Results have been saved to '{results_filename}'")