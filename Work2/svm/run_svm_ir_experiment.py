import time
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent folder to path
parent_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_folder_path)
print("Parent folder path:", parent_folder_path)

from data_preparation import CrossValidationDataLoader, DataProcessor
from svm.svm_algorithm import svmAlgorithm
from instance_reduction import reductionAlgorithm

def run_svm_ir_experiment(ir_method):
    """
    Runs the SVM experiment with the specified instance reduction method using the best hyperparameters.

    Parameters:
    - ir_method (str): The instance reduction method to apply ('drop3', 'ennth', 'gcnn').
    """
    # Define the best SVM hyperparameters
    best_params = {
        "balance": {
            "kernel": "linear",
            "C": 10.0,
            "class_weight": "balanced",
            "shrinking": True
        },
        "sick": {
            "kernel": "rbf",
            "C": 100.0,
            "class_weight": "balanced", 
            "shrinking": True,
            "gamma": 1
        }
    }

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

            # Apply instance reduction
            X_train_reduced, Y_train_reduced = reductionAlgorithm(X_train, Y_train, ir_method)

            # Storage
            storage = (len(Y_train_reduced)/len(Y_train))*100

            start = time.time()
            try:
                predictions = svmAlgorithm(X_train_reduced, Y_train_reduced, X_test, **params)
                end = time.time()
                total_time = end - start

                accuracy = (Y_test == predictions).mean()

                result_entry = {
                    "Dataset": f"{dataset_name}_{i}",
                    "Instance Reduction Method": ir_method,
                    **params,
                    "Accuracy": accuracy,
                    "Time (seconds)": total_time,
                    "Storage percentage": storage
                }

                results.append(result_entry)
            except Exception as e:
                # Handle exceptions (e.g., invalid parameter combinations)
                print(f"Error with parameters {params}: {e}")

    # Save results
    results_df = pd.DataFrame(results)
    os.makedirs('results', exist_ok=True)
    results_filename = f'results/svm_results_ir_{ir_method}.csv'
    results_df.to_csv(results_filename, index=False)

    print(f"Results have been saved to '{results_filename}'")