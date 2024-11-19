import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# DataProcessor class as modified
class DataProcessor:

    def __init__(self):
        pass

    def preprocess_dataset(self, features_df):
        """
        Preprocess the dataset by one-hot encoding categorical features and standardizing all features.
        :param features_df: DataFrame containing the features.
        :return: Processed DataFrame with all numeric features standardized to mean 0 and std 1.
        """
        # Identify numerical and categorical columns
        numerical_cols = features_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = features_df.select_dtypes(include=['object']).columns.tolist()
        
        df_processed = features_df.copy()
        
        # One-hot encode categorical columns
        if categorical_cols:
            # Use OneHotEncoder from sklearn
            ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
            ohe.fit(df_processed[categorical_cols])
            categorical_encoded = ohe.transform(df_processed[categorical_cols])
            # Get new column names
            ohe_columns = ohe.get_feature_names_out(categorical_cols)
            df_categorical_encoded = pd.DataFrame(categorical_encoded, columns=ohe_columns, index=features_df.index)
            
            # Drop original categorical columns and concatenate one-hot encoded columns
            df_processed = df_processed.drop(columns=categorical_cols)
            df_processed = pd.concat([df_processed, df_categorical_encoded], axis=1)
        
        # Now, standardize all features to mean 0, std 1
        scaler = StandardScaler()
        df_processed[df_processed.columns] = scaler.fit_transform(df_processed[df_processed.columns])
        
        return df_processed