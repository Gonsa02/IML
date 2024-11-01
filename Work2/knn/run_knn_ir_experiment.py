from instance_reduction import reductionAlgorithm
from knn.knn_algorithm import KnnAlgorithm
from data_preparation import CrossValidationDataLoader, DataProcessor
import time
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent folder to path
parent_folder_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_folder_path)
print("Parent folder path:", parent_folder_path)


def run_knn_ir_experiment(ir_method):
    """
    Runs the KNN experiment with the specified instance reduction method using the best hyperparameters.

    Parameters:
    - ir_method (str): The instance reduction method to apply ('drop3', 'ennth', 'gcnn').
    """
    # Define the best KNN hyperparameters
    best_params = {
        "balance": {
            "k": 7,
            "weight_method": "eq_weight",
            "voting_policy": "majority",
            "distance_metric": "hamming"
        },
        "sick": {
            "k": 7,
            "weight_method": "eq_weight",
            "voting_policy": "sheppard",
            "distance_metric": "hamming"
        }
    }

    # Load and preprocess the data
    loader_balance = CrossValidationDataLoader("bal")
    loader_sick = CrossValidationDataLoader("sick")

    data_preprocessor = DataProcessor()

    all_folds_balance = loader_balance.load_all_folds()
    all_folds_balance = data_preprocessor.preprocess_all_bal_folds(
        all_folds_balance)

    all_folds_sick = loader_sick.load_all_folds()
    all_folds_sick = data_preprocessor.preprocess_all_sick_folds(
        all_folds_sick)

    # Map datasets
    dataset_folds = {
        "balance": all_folds_balance,
        "sick": all_folds_sick
    }

    # For results
    results = []

    for dataset_name, folds in dataset_folds.items():
        params = best_params[dataset_name]
        num_folds = len(folds)
        for i in range(num_folds):
            fold = folds[i]
            Train, Test = fold

            Y_train = Train["class"]
            X_train = Train.drop("class", axis=1)

            Y_test = Test["class"]
            X_test = Test.drop("class", axis=1)

            # Prepare additional parameters for instance reduction
            ir_kwargs = {}
            # if ir_method == "drop3":
            #     ir_kwargs["metric"] = params["distance_metric"]
            #     ir_kwargs["voting"] = params["voting_policy"]

            # Apply instance reduction
            X_train_reduced, Y_train_reduced = reductionAlgorithm(
                X_train, Y_train, ir_method, **ir_kwargs)

            # Storage
            storage = (len(Y_train)/len(Y_train_reduced))*100

            start = time.time()

            knn = KnnAlgorithm()
            knn.train(X_train_reduced, Y_train_reduced,
                      weight_method=params["weight_method"])
            predictions = knn.predict(
                X_test, params["k"], params["distance_metric"], params["voting_policy"])

            end = time.time()

            total_time = end - start

            accuracy = (Y_test == predictions).mean()

            # Prettier distance metric
            if "minkowski_r" in params["distance_metric"]:
                dist_metric = "minkowski"
                r = params["distance_metric"].split("_r")[1]
            else:
                dist_metric = params["distance_metric"]
                r = ""

            result_entry = {
                "Dataset": f"{dataset_name}_{i}",
                "Instance Reduction Method": ir_method,
                "k": params["k"],
                "Feature Weighting Method": params["weight_method"],
                "Selection Method": params["voting_policy"],
                "Distance Metric": dist_metric,
                "r (if Minkowski)": r,
                "Accuracy": accuracy,
                "Time (seconds)": total_time,
                "Storage percentage": storage
            }

            results.append(result_entry)

    # Save results
    results_df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    results_filename = f"results/knn_results_ir_{ir_method}.csv"
    results_df.to_csv(results_filename, index=False)

    print(f"Results have been saved to '{results_filename}'")
