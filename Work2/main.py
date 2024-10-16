import time
import numpy as np

from data_loader import CrossValidationDataLoader
from data_processor import DataProcessor
from knn_algorithm import KnnAlgorithm

loader_balance = CrossValidationDataLoader("bal")
loader_sick = CrossValidationDataLoader("sick")

data_preprocessor = DataProcessor()

all_folds_balance = loader_balance.load_all_folds()
all_folds_balance = data_preprocessor.preprocess_all_bal_folds(all_folds_balance)

all_folds_sick = loader_sick.load_all_folds()
all_folds_sick = data_preprocessor.preprocess_all_sick_folds(all_folds_sick)

folds_datasets = [("balance", all_folds_balance), ("sick", all_folds_sick)]
weighting_methods = ["eq_weight", "information_gain", "chi2"]
voting_policies = ["majority", "idw", "sheppard"]
distance_metrics = ["minkowski_r1", "minkowski_r2", "hamming"]

for dataset_name, folds in folds_datasets:
    for k in [1, 3, 5, 7]:
        for wm in weighting_methods:
            for policy in voting_policies:
                for metric in distance_metrics:
                    accuracies = np.array([])
                    times = np.array([])
                    for i, (Train, Test) in enumerate(folds):
                        Y_train = Train["class"]
                        X_train = Train.drop("class", axis=1)

                        Y_test = Test["class"]
                        X_test = Test.drop("class", axis=1)

                        start = time.time()

                        knn = KnnAlgorithm()
                        knn.train(X_train, Y_train, weight_method=wm)
                        predictions = knn.predict(X_test,  k, metric, policy)

                        end = time.time()

                        total_time = end - start

                        accuracy = (Y_test == predictions).mean()

                        accuracies = np.append(accuracies, accuracy)
                        times = np.append(times, total_time)

                    with open(str(f'./results/{dataset_name}_k_{k}_{wm}_{policy}_{metric}.txt'), "w") as f:
                        f.write(f"{accuracies.mean()}\n{times.mean()}")
                    print(f'Accuracy with k = {k} wm = {wm} policy = {policy} metric = {metric}:   {accuracies.mean():.2f}')
                    print(f'Total time: {times.mean()}')