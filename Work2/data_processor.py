import data_loader
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)


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

    def preprocess_all_bal_folds(self, fold_list):
        folds = []
        for fold_num in range(self.num_folds):
            print(f"Preprocessing fold {fold_num + 1}/{self.num_folds}")
            folds.append((self.preprocess_bal(fold_list[fold_num][0]),
                          self.preprocess_bal(fold_list[fold_num][1])))
        return folds

    def preprocess_sick(self, df):
        columns_to_remove = ['TSH_measured', 'T3_measured',
                             'TT4_measured', 'T4U_measured', 'FTI_measured', 'TBG_measured', 'TBG']
        df = df.replace({"t": 1, "f": 0, "F": 1, "M": 0, "?": np.nan})
        df = df.drop(columns_to_remove, axis=1)
        df = df.dropna().reset_index(drop=True)

        return df

    def preprocess_sick_fold(self, df_train, df_test):
        # Cleaning data
        df_train = self.preprocess_sick(df_train)
        df_test = self.preprocess_sick(df_test)

        # Encoding categorical variables
        encoder = OneHotEncoder(
            categories=[['other', 'SVI', 'SVHC', 'STMW', 'SVHD']], sparse_output=False)

        encoded_column = encoder.fit_transform(df_train[['referral_source']])
        df_train = pd.concat([df_train.drop(columns=['referral_source']), pd.DataFrame(
            encoded_column, columns=encoder.get_feature_names_out(['referral_source']), index=df_train.index)], axis=1)

        encoded_column = encoder.transform(df_test[['referral_source']])
        df_test = pd.concat([df_test.drop(columns=['referral_source']), pd.DataFrame(
            encoded_column, columns=encoder.get_feature_names_out(['referral_source']), index=df_test.index)], axis=1)

        # Applying Normalization
        columns_to_normalize = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']
        scaler = MinMaxScaler()
        df_train[columns_to_normalize] = scaler.fit_transform(
            df_train[columns_to_normalize])
        df_test[columns_to_normalize] = scaler.transform(
            df_test[columns_to_normalize])

        # Moving y at the end
        df_train['class'] = df_train.pop('Class')
        df_test['class'] = df_test.pop('Class')

        return df_train, df_test

    def preprocess_all_sick_folds(self, fold_list):
        folds = []
        for fold_num in range(self.num_folds):
            print(f"Preprocessing fold {fold_num + 1}/{self.num_folds}")
            folds.append((self.preprocess_sick_fold(
                fold_list[fold_num][0], fold_list[fold_num][1])))
        return folds
