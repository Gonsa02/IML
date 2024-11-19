import pandas as pd
from scipy.io import arff
import os

class DataLoader:
    def __init__(self):
        """
        Initialize the data loader without specifying a dataset.
        """
        pass
    
    def load_arff_data(self, dataset_name):
        """
        Load an ARFF file and return features and class labels separately.
        Missing values are handled as NaN.
        Categorical data is decoded from bytes to strings.
        :param dataset_name: Name of the dataset (the filename without extension).
        :return: A tuple (features_df, labels_series).
        """

        if dataset_name in ["vowel", "splice"]:
            class_column_name = "Class"
        elif dataset_name in ["satimage"]:
            class_column_name = "clase"
        else:
            raise("error! unimplemented dataset")

        file_path = os.path.join("data", f"{dataset_name}.arff")
        try:
            data, meta = arff.loadarff(file_path)
        except FileNotFoundError:
            print(f"File {file_path} not found.")
            return None, None
        
        df = pd.DataFrame(data)
        
        for col in df.select_dtypes([object]).columns:
            if col != class_column_name:
                df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
        
        if class_column_name not in df.columns:
            print(f"Class column '{class_column_name}' not found in dataset '{dataset_name}'.")
            return df, None
        
        labels = df[class_column_name]
        features = df.drop(columns=[class_column_name])
        
        if labels.dtype == object:
            labels = labels.apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
        
        return features, labels
