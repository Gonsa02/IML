import pandas as pd
from scipy.io import arff
import os

class CrossValidationDataLoader:
    def __init__(self, dataset_name, num_folds=10):
        """
        Initialize the data loader with the dataset name and number of folds.
        :param dataset_name: Name of the dataset (the folder and filename base).
        :param num_folds: Number of cross-validation folds (default: 10).
        """
        self.base_filename = os.path.join("data", dataset_name, dataset_name)
        self.num_folds = num_folds

    def load_arff_data(self, file_path):
        """
        Load an ARFF file and convert it to a pandas DataFrame.
        :param file_path: Path to the ARFF file.
        :return: A pandas DataFrame with the loaded data.
        """
        data, meta = arff.loadarff(file_path)
        df = pd.DataFrame(data)

        # Convert bytes to strings for categorical data
        for col in df.select_dtypes([object]).columns:
            df[col] = df[col].str.decode('utf-8')

        return df

    def load_fold(self, fold_num):
        """
        Load a specific fold for training and testing.
        :param fold_num: Fold number (0-indexed).
        :return: Tuple (TrainMatrix, TestMatrix) containing the training and test data for the given fold.
        """
        train_file = f"{self.base_filename}.fold.{fold_num:06d}.train.arff"
        test_file = f"{self.base_filename}.fold.{fold_num:06d}.test.arff"

        # Load train and test data
        train_data = self.load_arff_data(train_file)
        test_data = self.load_arff_data(test_file)

        return train_data, test_data

    def load_all_folds(self):
        """
        Load all folds and return a list of (TrainMatrix, TestMatrix) for each fold.
        :return: List of tuples [(TrainMatrix, TestMatrix), ...] for each fold.
        """
        folds = []
        for fold_num in range(self.num_folds):
            print(f"Loading fold {fold_num + 1}/{self.num_folds}")
            train_data, test_data = self.load_fold(fold_num)
            folds.append((train_data, test_data))
        return folds
