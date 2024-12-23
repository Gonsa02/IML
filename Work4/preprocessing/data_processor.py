import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import warnings
import sklearn

warnings.filterwarnings('ignore', category=FutureWarning)

class DataProcessor:

    def __init__(self):
        pass

    def preprocess_dataset(self, features_df):
        """
        Preprocess the dataset by deleting duplicates, one-hot encoding categorical features, and standardizing numerical features.
        Handles datasets with only numerical features, only categorical features, or both.
        :param features_df: DataFrame containing the features.
        :return: Processed DataFrame with duplicates removed, numerical features standardized, and one-hot encoded categorical features.
        """

        numerical_cols = features_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = features_df.select_dtypes(include=['object']).columns.tolist()

        df_processed_list = []

        if numerical_cols:
            scaler_num = StandardScaler()
            df_numerical = pd.DataFrame(
                scaler_num.fit_transform(features_df[numerical_cols]),
                columns=numerical_cols,
                index=features_df.index
            )
            df_processed_list.append(df_numerical)

        if categorical_cols:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            categorical_encoded = ohe.fit_transform(features_df[categorical_cols])

            ohe_columns = ohe.get_feature_names_out(categorical_cols)
            df_categorical_encoded = pd.DataFrame(
                categorical_encoded,
                columns=ohe_columns,
                index=features_df.index
            )

            # Do NOT standardize one-hot encoded features
            df_processed_list.append(df_categorical_encoded)

        if df_processed_list:
            df_processed = pd.concat(df_processed_list, axis=1)
        else:
            df_processed = pd.DataFrame(index=features_df.index)

        return df_processed
