import os
import numpy as np
import pandas as pd
import csv
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.metrics.cluster import contingency_matrix
from tqdm import tqdm
from itertools import product
import time
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from preprocessing import DataLoader, DataProcessor
from .global_kmeans import fast_global_k_means


# Define evaluation metrics
metrics = {
    "Adjusted Rand Index": adjusted_rand_score,
    "Davies-Bouldin Index": davies_bouldin_score,
    "Silhouette Score": silhouette_score
}


def purity_score(y_true, y_pred):
    matrix = contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(matrix, axis=0)) / np.sum(matrix)


def run_experiment(args):
    dataset_name, parameters = args
    k, dist = parameters

    try:
        # Initialize loaders and preprocessors
        data_loader = DataLoader()
        data_processor = DataProcessor()

        # Load dataset
        data, labels = data_loader.load_arff_data(dataset_name)
        X = data_processor.preprocess_dataset(data)
        y_true = labels

        start_time = time.time()
        km = fast_global_k_means(k=k, distance=dist)
        y_pred = km.fit_predict(X)
        elapsed_time = time.time() - start_time

        scores = {
            "k": k, "distance": dist,
            "Adjusted Rand Index": adjusted_rand_score(y_true, y_pred),
            "Davies-Bouldin Index": davies_bouldin_score(X, y_pred),
            "Silhouette Score": silhouette_score(X, y_pred),
            "Purity Score": purity_score(y_true, y_pred),
            "Time (s)": elapsed_time,
            "Iterations": km.n_iter_
        }

        return scores

    except Exception as e:
        # Return the exception with parameters for easier debugging
        return {"k": k, "distance": dist, "Error": str(e)}


def run_kmeans():
    datasets = ["vowel", "splice", "satimage"]
    k_values = range(2, 16)
    distance_metrics = ['euclidean', 'manhattan', 'cosine']

    # Prepare parameter combinations
    parameter_combinations = list(product(k_values, distance_metrics))
    print(f"Total parameter combinations: {len(parameter_combinations)}")

    # Create results directory
    os.makedirs("results", exist_ok=True)

    for dataset_name in datasets:
        print(f"\nRunning experiments for {dataset_name} dataset...")
        results_file = f"results/GLOBAL_KMEANS_{dataset_name}_results.csv"

        # Check if the results file exists to determine if we need to write headers
        file_exists = os.path.isfile(results_file)

        # Prepare experiments
        experiments = [(dataset_name, params)
                       for params in parameter_combinations]

        # Run experiments in parallel
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(
                run_experiment, args): args for args in experiments}

            # Open the CSV file in append mode
            with open(results_file, 'a', newline='') as csvfile:
                fieldnames = ["k", "distance", "Adjusted Rand Index", "Davies-Bouldin Index",
                              "Silhouette Score", "Purity Score", "Time (s)", "Iterations", "Error"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                # Write header only if the file doesn't exist or is empty
                if not file_exists or os.stat(results_file).st_size == 0:
                    writer.writeheader()

                for future in tqdm(as_completed(futures), total=len(experiments), desc=f"Processing {dataset_name}"):
                    try:
                        result = future.result()
                        writer.writerow(result)
                    except Exception as e:
                        print(f"Experiment failed: {e}")

    create_combined_csv(datasets, "results/globalkmeans_results.csv")


def create_combined_csv(datasets, combined_file):
    try:
        df_list = []

        for dataset_name in datasets:
            individual_file = f"""results/GLOBAL_KMEANS_{
                dataset_name}_results.csv"""
            if os.path.isfile(individual_file):
                df = pd.read_csv(individual_file)
                df['dataset'] = dataset_name
                df_list.append(df)
                print(f"Added {individual_file} to combined CSV.")
            else:
                print(f"""Warning: {
                      individual_file} does not exist and will be skipped.""")

        if df_list:
            # Concatenate all DataFrames
            combined_df = pd.concat(df_list, ignore_index=True)

            # Sort the combined DataFrame
            sort_columns = ['k', 'distance']
            combined_df_sorted = combined_df.sort_values(
                by=sort_columns, ascending=True, ignore_index=True)

            # Save to the combined CSV
            combined_df_sorted.to_csv(combined_file, index=False)
            print(f"Combined CSV saved as {combined_file}")

            # **Delete Original CSVs After Combining**
            for dataset_name in datasets:
                individual_file = f"""results/GLOBAL_KMEANS_{
                    dataset_name}_results.csv"""
                if os.path.isfile(individual_file):
                    # os.remove(individual_file)
                    print(f"Deleted original file: {individual_file}")
        else:
            print("No individual CSV files found to combine.")

    except Exception as e:
        print(f"Failed to create combined CSV: {e}")


if __name__ == "__main__":
    run_kmeans()
