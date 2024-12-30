import os
import pandas as pd
from sklearn.decomposition import PCA
from preprocessing import DataLoader, DataProcessor


def run_sklearn_pca():

    data_loader     = DataLoader()
    data_processor  = DataProcessor()

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

        # PCA
        pca = PCA()
        pca_result = pca.fit_transform(df)

        # Save results
        pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(pca_result.shape[1])])
        pca_df['label'] = labels

        output_file = os.path.join('results', f'{dataset_name}_pca_results.csv')
        pca_df.to_csv(output_file, index=False)
        print(f"PCA results saved for {dataset_name} to {output_file}")

