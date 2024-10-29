import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product

from data_loader import CrossValidationDataLoader
from data_processor import DataProcessor
from knn_algorithm import KnnAlgorithm

# Load and preprocess the data
loader_balance = CrossValidationDataLoader("bal")
loader_sick = CrossValidationDataLoader("sick")

data_preprocessor = DataProcessor()

all_folds_balance = loader_balance.load_all_folds()
all_folds_balance = data_preprocessor.preprocess_all_bal_folds(all_folds_balance)

all_folds_sick = loader_sick.load_all_folds()
all_folds_sick = data_preprocessor.preprocess_all_sick_folds(all_folds_sick)

# map (readibility)
dataset_folds = {
    "balance": all_folds_balance,
    "sick": all_folds_sick
}

# parameter list
ks                  = [1, 3, 5, 7]
weighting_methods   = ["eq_weight", "information_gain", "chi2"]
voting_policies     = ["majority", "idw", "sheppard"]
distance_metrics    = ["minkowski_r1", "minkowski_r2", "hamming"]

# prepare combinations (so we can use tqdm)
parameter_combinations = []
for dataset_name, folds in dataset_folds.items():
    num_folds = len(folds)
    for i in range(num_folds):
        for k in ks:
            for wm in weighting_methods:
                for policy in voting_policies:
                    for metric in distance_metrics:
                        parameter_combinations.append((dataset_name, i, k, wm, policy, metric))

# for results
results = []


for dataset_name, fold_index, k, wm, policy, metric in tqdm(parameter_combinations, desc="Processing combinations"):
    folds       =  dataset_folds[dataset_name]
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

    # prettier ds
    if "minkowski_r" in metric:
        r = metric.split("_r")[1]
    else:
        r = ""

    result_entry = {
        "Dataset": f"{dataset_name}_{fold_index}",
        "k": k,
        "Feature Weighting Method": wm,
        "Selection Method": policy,
        "Distance Metric": metric.split("_")[0],
        "r (if Minkowski)": r,
        "Accuracy": accuracy,
        "Time (seconds)": total_time
    }

    results.append(result_entry)

# save results
results_df = pd.DataFrame(results)
results_df.to_csv('knn_results.csv', index=False)

print("Results have been saved to '.knn_results.csv'")
