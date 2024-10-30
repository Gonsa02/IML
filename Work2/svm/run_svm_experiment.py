import time
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product

# Add parent folder to path
parent_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_folder_path)
print("Parent folder path:", parent_folder_path)

from data_preparation import CrossValidationDataLoader, DataProcessor
from svm.svm_algorithm import svmAlgorithm

def run_svm_experiment():
    # Load and preprocess data
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

    # Define the parameter grid with extensive parameters for 'linear' and 'rbf' kernels
    parameter_grid = [
        {
            'kernel': ['linear'],
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            # 'tol': [1e-3, 1e-4, 1e-5],
            'class_weight': [None, 'balanced'],
            'shrinking': [True, False],
        },
        {
            'kernel': ['rbf'],
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            # 'tol': [1e-3, 1e-4, 1e-5],
            'class_weight': [None, 'balanced'],
            'shrinking': [True, False],
        }
    ]

    def get_param_combinations(parameter_grid):
        all_param_combinations = []
        for param_set in parameter_grid:
            keys = param_set.keys()
            values = (param_set[key] if isinstance(param_set[key], list) else [param_set[key]] for key in keys)
            for combination in product(*values):
                param_dict = dict(zip(keys, combination))
                all_param_combinations.append(param_dict)
        return all_param_combinations

    all_param_combinations = get_param_combinations(parameter_grid)

    # For results
    results = []

    # Prepare combinations
    parameter_combinations = []
    for dataset_name, folds in dataset_folds.items():
        num_folds = len(folds)
        for i in range(num_folds):
            for param_dict in all_param_combinations:
                parameter_combinations.append((dataset_name, i, param_dict))

    for dataset_name, fold_index, param_dict in tqdm(parameter_combinations, desc="Processing combinations"):
        folds = dataset_folds[dataset_name]
        Train, Test = folds[fold_index]

        Y_train = Train["class"]
        X_train = Train.drop("class", axis=1)

        Y_test = Test["class"]
        X_test = Test.drop("class", axis=1)

        start = time.time()

        # Call svmAlgorithm with hyperparameters
        try:
            predictions = svmAlgorithm(X_train, Y_train, X_test, **param_dict)
            end = time.time()
            total_time = end - start

            accuracy = (Y_test == predictions).mean()

            result_entry = {
                'Dataset': f"{dataset_name}_{fold_index}",
                'Accuracy': accuracy,
                'Time (seconds)': total_time,
                **param_dict
            }

            results.append(result_entry)
        except Exception as e:
            # Handle exceptions (e.g., invalid parameter combinations)
            print(f"Error with parameters {param_dict}: {e}")

    # Save results
    results_df = pd.DataFrame(results)
    os.makedirs('results', exist_ok=True)
    results_filename = 'results/svm_results.csv'
    results_df.to_csv(results_filename, index=False)

    print(f"Results have been saved to '{results_filename}'")
