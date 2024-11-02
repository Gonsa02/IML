import time
import os
import sys

import pandas as pd
from tqdm import tqdm

# Add parent folder to path
parent_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_folder_path)
print("Parent folder path:", parent_folder_path)

from data_preparation import CrossValidationDataLoader, DataProcessor
from knn.knn_algorithm import KnnAlgorithm

def run_knn_experiment():
    # Load and preprocess the data
    loader_balance = CrossValidationDataLoader("bal")
    loader_sick = CrossValidationDataLoader("sick")

    data_preprocessor = DataProcessor()

    all_folds_balance = loader_balance.load_all_folds()
    all_folds_balance = data_preprocessor.preprocess_all_bal_folds(all_folds_balance)

    all_folds_sick = loader_sick.load_all_folds()
    all_folds_sick = data_preprocessor.preprocess_all_sick_folds(all_folds_sick)

    # Map datasets
    dataset_folds = {
        "balance": all_folds_balance,
        "sick": all_folds_sick
    }

    # Parameter list
    ks = [1, 3, 5, 7]
    weighting_methods = ["eq_weight", "information_gain", "chi2"]
    voting_policies = ["majority", "idw", "sheppard"]
    distance_metrics = ["minkowski_r1", "minkowski_r2", "hamming"]

    # Prepare combinations
    parameter_combinations = []
    for dataset_name, folds in dataset_folds.items():
        num_folds = len(folds)
        for i in range(num_folds):
            for k in ks:
                for wm in weighting_methods:
                    for policy in voting_policies:
                        for metric in distance_metrics:
                            parameter_combinations.append((dataset_name, i, k, wm, policy, metric))

    # For results
    results = []

    for dataset_name, fold_index, k, wm, policy, metric in tqdm(parameter_combinations, desc="Processing combinations"):
        folds = dataset_folds[dataset_name]
        Train, Test = folds[fold_index]

        Y_train = Train["class"]
        X_train = Train.drop("class", axis=1)

        Y_test = Test["class"]
        X_test = Test.drop("class", axis=1)

        start = time.time()

        knn = KnnAlgorithm()
        knn.train(X_train, Y_train, weight_method=wm)
        predictions = knn.predict(X_test, k, metric, policy)

        end = time.time()

        total_time = end - start

        accuracy = (Y_test == predictions).mean()

        # Prettier distance metric
        if "minkowski_r" in metric:
            dist_metric = "minkowski"
            r = metric.split("_r")[1]
        else:
            dist_metric = metric
            r = ""

        result_entry = {
            "Dataset": f"{dataset_name}_{fold_index}",
            "k": k,
            "Feature Weighting Method": wm,
            "Selection Method": policy,
            "Distance Metric": dist_metric,
            "r (if Minkowski)": r,
            "Accuracy": accuracy,
            "Time (seconds)": total_time
        }

        results.append(result_entry)

    # Save results
    results_df = pd.DataFrame(results)
    os.makedirs('results', exist_ok=True)
    results_filename = 'results/knn_results.csv'
    results_df.to_csv(results_filename, index=False)

    print(f"Results have been saved to '{results_filename}'")