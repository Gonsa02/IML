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

for k in [1, 3, 5, 7]:
    accuracies = np.array([])
    for i, (Train, Test) in enumerate(all_folds_balance):
        Y_train = Train["class"]
        X_train = Train.drop("class", axis=1)

        Y_test = Test["class"]
        X_test = Test.drop("class", axis=1)

        knn = KnnAlgorithm(X_train, Y_train)
        predictions = knn.predict(X_test,   k)

        accuracy = (Y_test == predictions).mean()

        accuracies = np.append(accuracies, accuracy)

    print(f'Accuracy with k = {k}: {accuracies.mean():.2f}')