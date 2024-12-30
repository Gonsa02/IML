import os
import pandas as pd
from sklearn.decomposition import IncrementalPCA
from preprocessing import DataLoader, DataProcessor

def run_incremental_pca():

    data_loader = DataLoader()
    data_processor = DataProcessor()

    os.makedirs('results', exist_ok=True)

    datasets_info = {
        'satimage': data_loader.load_arff_data('satimage'),
        'splice': data_loader.load_arff_data('splice'),
    }

    preprocessed_datasets = {}
    for dataset_name, (df, labels) in datasets_info.items():
        preprocessed_df = data_processor.preprocess_dataset(df)
        preprocessed_datasets[dataset_name] = {
            'df': preprocessed_df,
            'labels': labels
        }

    for dataset_name, dataset_content in preprocessed_datasets.items():
        df = dataset_content['df']
        labels = dataset_content['labels']

        ipca = IncrementalPCA(n_components=min(df.shape[1], 10))  # max 10 components o menys
        ipca_result = ipca.fit_transform(df)

        ipca_df = pd.DataFrame(ipca_result, columns=[f'PC{i+1}' for i in range(ipca_result.shape[1])])
        ipca_df['label'] = labels

        output_file = os.path.join('results', f'{dataset_name}_incremental_pca_results.csv')
        ipca_df.to_csv(output_file, index=False)
        print(f"Incremental PCA results saved for {dataset_name} to {output_file}")