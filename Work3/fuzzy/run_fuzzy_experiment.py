import os  # For directory handling
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from tqdm import tqdm
from preprocessing import DataLoader, DataProcessor  # Assuming your preprocessing scripts
from fuzzy import GSFuzzyCMeans  # Assuming the class implementation from earlier
from itertools import product
import time  # For timing
from concurrent.futures import ProcessPoolExecutor, as_completed


def run_fuzzy_experiment():

    fuzzy_exponents = [1.5, 2.5, 3.5]   # Testing more values for m
    xi_values       = [0.3, 0.5, 0.7]   # Testing more values for xi
    k_values        = range(2, 16)      # Testing k from 2 to 15 clusters

    parameter_combinations = list(product(k_values, fuzzy_exponents, xi_values))
    print(f"Total parameter combinations: {len(parameter_combinations)}")

    # in case it doesnt exist!
    os.makedirs("results", exist_ok=True)

    def run_experiment(args):
        dataset_name, parameters = args
        (k, m, xi) = parameters

        try:
            # Load DataLoader and DataProcessor
            data_loader = DataLoader()
            data_processor = DataProcessor()

            # Load and preprocess the dataset
            data, labels = data_loader.load_arff_data(dataset_name)
            X = data_processor.preprocess_dataset(data)
            y_true = labels

            # Initialize accumulators for metrics
            total_scores = {
                "Adjusted Rand Index": 0.0,
                "Davies-Bouldin Index": 0.0,
                "Silhouette Score": 0.0,
                "Time (s)": 0.0,
                "Iterations": 0
            }

            num_runs = 3  # Number of times to run each experiment

            for run in range(num_runs):
                start_time = time.time()
                # Fit GSFuzzyCMeans
                gs_fcm = GSFuzzyCMeans(n_clusters=k, m=m, xi=xi, max_iter=100, random_state=None)  # No state for random!
                gs_fcm.fit(X)
                y_pred = gs_fcm.predict(X)
                # Record time and iterations
                elapsed_time = time.time() - start_time
                num_iterations = gs_fcm.n_iter_

                # Accumulate metrics
                total_scores["Adjusted Rand Index"] += adjusted_rand_score(y_true, y_pred)
                total_scores["Davies-Bouldin Index"] += davies_bouldin_score(X, y_pred)
                total_scores["Silhouette Score"] += silhouette_score(X, y_pred)
                total_scores["Time (s)"] += elapsed_time
                total_scores["Iterations"] += num_iterations

            # Average the metrics
            averaged_scores = {
                "k": k,
                "m": m,
                "xi": xi,
                "Adjusted Rand Index": total_scores["Adjusted Rand Index"] / num_runs,
                "Davies-Bouldin Index": total_scores["Davies-Bouldin Index"] / num_runs,
                "Silhouette Score": total_scores["Silhouette Score"] / num_runs,
                "Time (s)": total_scores["Time (s)"] / num_runs,
                "Iterations": total_scores["Iterations"] / num_runs
            }

            return dataset_name, averaged_scores

        except Exception as e:
            # Return the exception to be handled in the main loop
            return dataset_name, e

    datasets = ["vowel", "splice", "satimage"] # vowel first cos it is smaller; if the code fails, it fails faster!

    for dataset_name in datasets:
        print(f"\nRunning experiments for {dataset_name} dataset...")
        results_file = f"results/FUZZY_{dataset_name}_results_AVG.csv"

        # Load existing results if they exist
        try:
            existing_df = pd.read_csv(results_file)
        except FileNotFoundError:
            existing_df = pd.DataFrame()

        # Build the list of parameter combinations to test
        experiments = [(dataset_name, (k, m, xi)) for k, m, xi in parameter_combinations]

        # Run experiments in parallel
        total_experiments = len(experiments)
        results_list = []
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(run_experiment, args): args for args in experiments}

            for future in tqdm(as_completed(futures), total=total_experiments, desc=f"Processing combinations for {dataset_name}"):
                args = futures[future]
                try:
                    dataset_name, result = future.result()
                    if isinstance(result, Exception):
                        print(f"Experiment {args} generated an exception: {result}")
                    else:
                        results_list.append(result)
                except Exception as e:
                    print(f"Experiment {args} generated an exception: {e}")

        # If there are new results, append them to existing_df
        if results_list:
            # Extract only valid results (skip exceptions)
            valid_results = [result[1] for result in results_list if isinstance(result, tuple)]
            if valid_results:  # Check if there are valid results to avoid empty DataFrame
                new_results_df = pd.DataFrame(valid_results)
                existing_df = pd.concat([existing_df, new_results_df], ignore_index=True)

        # Save the updated DataFrame
        existing_df.to_csv(results_file, index=False)
        print(f"Results saved to {results_file}")
