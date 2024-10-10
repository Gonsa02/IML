import data_loader
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class DataProcessor:

    def __init__(self, num_folds=10):
        self.num_folds = num_folds

    def preprocess_bal(self, df):
        df[['a1', 'a2', 'a3', 'a4']] = df[[
            'a1', 'a2', 'a3', 'a4']].astype(int)

        encoder = OneHotEncoder(
            categories=[[1, 2, 3, 4, 5]], sparse_output=False)

        for col in ['a1', 'a2', 'a3', 'a4']:
            encoded_column = encoder.fit_transform(df[[col]])
            df = pd.concat([df.drop(columns=[col]), pd.DataFrame(
                encoded_column, columns=encoder.get_feature_names_out([col]))], axis=1)
        return df

    def preprocess_all_bal_folds(self, df):
        folds = []
        for fold_num in range(self.num_folds):
            print(f"Preprocessing fold {fold_num + 1}/{self.num_folds}")
            folds.append((self.preprocess_bal(df[fold_num][0]),
                          self.preprocess_bal(df[fold_num][1])))
        return folds
