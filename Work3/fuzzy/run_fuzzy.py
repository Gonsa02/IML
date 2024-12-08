import os
import csv
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.metrics.cluster import contingency_matrix

from preprocessing import DataLoader, DataProcessor
from fuzzy import GSFuzzyCMeans


# Define evaluation metrics modes
metric_modes = {
    "Adjusted Rand Index": "max",
    "Davies-Bouldin Index": "min",
    "Silhouette Score": "max",
    "Purity Score": "max",
    "Time (s)": "min",
    "Iterations": "min"
}


def purity_score_func(y_true, y_pred):
    matrix = contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(matrix, axis=0)) / np.sum(matrix)


def run_experiment(dataset_name, params):
    k, m, xi = params
    seeds = [0, 1, 2, 3, 4]  # Run 5 times with these seeds

    try:
        # Initialize loaders and preprocessors
        data_loader = DataLoader()
        data_processor = DataProcessor()

        # Load dataset
        data, labels = data_loader.load_arff_data(dataset_name)
        X = data_processor.preprocess_dataset(data)
        y_true = labels

        # Initialize best scores
        best_scores = {}
        for metric_name, mode in metric_modes.items():
            best_scores[metric_name] = (-np.inf, None) if mode == "max" else (np.inf, None)

        for seed in seeds:
            start_time = time.time()
            gs_fcm = GSFuzzyCMeans(n_clusters=k, m=m, xi=xi, max_iter=100, random_state=seed)
            gs_fcm.fit(X)
            y_pred = gs_fcm.predict(X)
            elapsed_time = time.time() - start_time

            run_scores = {
                "Adjusted Rand Index": adjusted_rand_score(y_true, y_pred),
                "Davies-Bouldin Index": davies_bouldin_score(X, y_pred),
                "Silhouette Score": silhouette_score(X, y_pred),
                "Purity Score": purity_score_func(y_true, y_pred),
                "Time (s)": elapsed_time,
                "Iterations": gs_fcm.n_iter_
            }

            # Update best scores based on metric modes
            for metric_name, (best_val, best_seed) in best_scores.items():
                current_val = run_scores[metric_name]
                mode = metric_modes[metric_name]
                if (mode == "max" and current_val > best_val) or (mode == "min" and current_val < best_val):
                    best_scores[metric_name] = (current_val, seed)

        # Prepare the result entry with best scores and corresponding seeds
        result_entry = {
            "k": k,
            "m": m,
            "xi": xi
        }

        for metric_name, (val, seed) in best_scores.items():
            result_entry[metric_name] = val
            result_entry[f"{metric_name} Seed"] = seed

        return result_entry

    except Exception as e:
        # Return the exception with parameters for easier debugging
        return {"k": k, "m": m, "xi": xi, "Error": str(e)}


def run_fuzzy():
    # Initialize DataLoader and DataProcessor
    data_loader = DataLoader()
    data_processor = DataProcessor()

    # Load and preprocess datasets
    datasets_info = {
        'satimage': data_loader.load_arff_data('satimage'),
        'splice': data_loader.load_arff_data('splice'),
        'vowel': data_loader.load_arff_data('vowel')
    }

    preprocessed_datasets = {}
    for dataset_name, (df, labels) in datasets_info.items():
        preprocessed_df = data_processor.preprocess_dataset(df)
        preprocessed_datasets[dataset_name] = {
            'df': preprocessed_df,
            'labels': labels
        }

    # Define parameter ranges
    fuzzy_exponents = [1.5, 2.5, 3.5]
    xi_values = [0.3, 0.5, 0.7]
    k_values = range(2, 16)

    datasets = ["vowel", "splice", "satimage"]

    for dataset_name in datasets:
        print(f"\nRunning experiments for {dataset_name} dataset...")
        results_file = f"results/FUZZY_{dataset_name}_results_MAX.csv"

        # Create results directory if it doesn't exist
        os.makedirs("results", exist_ok=True)

        # Prepare parameter combinations
        parameter_combinations = list(product(k_values, fuzzy_exponents, xi_values))
        print(f"Total parameter combinations: {len(parameter_combinations)} for dataset {dataset_name}")

        # Load Existing Results to avoid re-running
        if os.path.isfile(results_file):
            fuzzy_df = pd.read_csv(results_file)
            existing_combinations = set(zip(
                fuzzy_df['k'], fuzzy_df['m'], fuzzy_df['xi']
            ))
        else:
            fuzzy_df = pd.DataFrame()
            existing_combinations = set()

        # Filter out already processed combinations
        parameter_combinations = [
            params for params in parameter_combinations if params not in existing_combinations
        ]

        total_combinations = len(parameter_combinations)
        print(f"Total new combinations to run: {total_combinations}")

        if total_combinations == 0:
            print(f"All combinations for {dataset_name} have been processed.")
            continue

        with ProcessPoolExecutor() as executor:
            # Prepare experiments as tuples (dataset_name, params)
            experiments = [(dataset_name, params) for params in parameter_combinations]

            # Submit all experiments to the executor
            futures = {executor.submit(run_experiment, dataset, params): params for dataset, params in experiments}

            # Open the CSV file in append mode
            with open(results_file, 'a', newline='') as csvfile:
                fieldnames = [
                    "k", "m", "xi",
                    "Adjusted Rand Index", "Adjusted Rand Index Seed",
                    "Davies-Bouldin Index", "Davies-Bouldin Index Seed",
                    "Silhouette Score", "Silhouette Score Seed",
                    "Purity Score", "Purity Score Seed",
                    "Time (s)", "Time (s) Seed",
                    "Iterations", "Iterations Seed",
                    "Error"
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                # Write header only if the file doesn't exist or is empty
                if not os.path.isfile(results_file) or os.stat(results_file).st_size == 0:
                    writer.writeheader()

                # Iterate over completed futures with progress bar
                for future in tqdm(as_completed(futures), total=total_combinations, desc=f"Processing {dataset_name}"):
                    try:
                        result = future.result()
                        if result:
                            writer.writerow(result)
                    except Exception as e:
                        params = futures[future]
                        print(f"Combination {params} generated an exception: {e}")

        # Optional: Sort the CSV after all experiments are done
        fuzzy_sort_csv(results_file)

    # **Add this section to create the combined CSV after processing all datasets**
    create_combined_csv(datasets, "results/fuzzy_results.csv")


def fuzzy_sort_csv(results_file):
    df = pd.read_csv(results_file)
    sort_columns = ['k', 'm', 'xi']
    df_sorted = df.sort_values(by=sort_columns, ascending=True, ignore_index=True)
    df_sorted.to_csv(results_file, index=False)
    print(f"Sorted the results and saved to {results_file}")


def create_combined_csv(datasets, combined_file):
    try:
        # Initialize an empty list to hold DataFrames
        df_list = []

        for dataset_name in datasets:
            individual_file = f"results/FUZZY_{dataset_name}_results_MAX.csv"
            if os.path.isfile(individual_file):
                df = pd.read_csv(individual_file)
                df['dataset'] = dataset_name  # Add the dataset column
                df_list.append(df)
                print(f"Added {individual_file} to combined CSV.")
            else:
                print(f"Warning: {individual_file} does not exist and will be skipped.")

        if df_list:
            # Concatenate all DataFrames
            combined_df = pd.concat(df_list, ignore_index=True)

            # Sort the combined DataFrame
            sort_columns = ['k', 'm', 'xi']
            combined_df_sorted = combined_df.sort_values(by=sort_columns, ascending=True, ignore_index=True)

            # Save to the combined CSV
            combined_df_sorted.to_csv(combined_file, index=False)
            print(f"Combined CSV saved as {combined_file}")

            # **Delete Original CSVs After Combining**
            for dataset_name in datasets:
                individual_file = f"results/FUZZY_{dataset_name}_results_MAX.csv"
                if os.path.isfile(individual_file):
                    os.remove(individual_file)
                    print(f"Deleted original file: {individual_file}")
        else:
            print("No individual CSV files found to combine.")

    except Exception as e:
        print(f"Failed to create combined CSV: {e}")


if __name__ == "__main__":
    run_fuzzy()
