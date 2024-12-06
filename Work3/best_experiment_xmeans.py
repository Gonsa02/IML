import pandas as pd
import numpy as np
import json

np.set_printoptions(suppress=True)
np.set_printoptions(threshold=np.inf)

#pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

df = pd.read_csv("./results/xmeans_results.csv")

grouped = df.groupby(['Dataset', 'k_max'])

results = []

for (dataset, k_max), group in grouped:
    ari = []
    silhouette = []
    purity = []
    dbi = []

    ari_seeds = []
    silhouette_seeds = []
    purity_seeds = []
    dbi_seeds = []

    for _, row in group.iterrows():
        ari.append(row['ARI'])
        silhouette.append(row['Silhouette'])
        purity.append(row['Purity'])
        dbi.append(row['DBI'])

        ari_seeds.append(row['Seed'])
        silhouette_seeds.append(row['Seed'])
        purity_seeds.append(row['Seed'])
        dbi_seeds.append(row['Seed'])

    ari = np.array(ari)
    silhouette = np.array(silhouette)
    purity = np.array(purity)
    dbi = np.array(dbi) # More lower better value

    results.append({
        "Dataset": dataset,
        "k_max": k_max,
        "Best ARI": np.max(ari),
        "Best K ARI": group.iloc[np.argmax(ari)]["best_k"],
        "Best ARI Seed": group.iloc[np.argmax(ari)]["Seed"],
        "Best Silhouette": np.max(silhouette),
        "Best K Silhouette": group.iloc[np.argmax(silhouette)]["best_k"],
        "Best Silhouette Seed": group.iloc[np.argmax(silhouette)]["Seed"],
        "Best Purity": np.max(purity),
        "Best K Purity": group.iloc[np.argmax(purity)]["best_k"],
        "Best Purity Seed": group.iloc[np.argmax(purity)]["Seed"],
        "Best DBI": np.max(dbi),
        "Best K DBI": group.iloc[np.argmax(dbi)]["best_k"],
        "Best DBI Seed": group.iloc[np.argmax(dbi)]["Seed"],
    })

results_df = pd.DataFrame(results)

results_df.to_csv("./results/xmeans_processed_results"  + '.csv', index=False)

grouped = results_df.groupby("Dataset")

best_configs = {}

for dataset, group in grouped:
    print(f"Dataset: {dataset}")

    best_configs[dataset] = {}
    
    best_ari_row = group.loc[group["Best ARI"].idxmax()]
    best_silhouette_row = group.loc[group["Best Silhouette"].idxmax()]
    best_purity_row = group.loc[group["Best Purity"].idxmax()]
    best_dbi_row = group.loc[group["Best DBI"].idxmin()]

    best_configs[dataset]["Adjusted Rand Index"] = {
        'best_k_max': best_ari_row['k_max'],
        'best_k': best_ari_row['Best K ARI'],
        'best_seed': best_ari_row['Best ARI Seed'],
        'ARI Score': best_ari_row['Best ARI']
    }

    best_configs[dataset]["Davies-Bouldin Index"] = {
        'best_k_max': best_dbi_row['k_max'],
        'best_k': best_dbi_row["Best K DBI"],
        'best_seed': best_dbi_row['Best DBI Seed'],
        'DBI Score': best_dbi_row["Best DBI"]
    }

    best_configs[dataset]["Silhouette Score"] = {
        'best_k_max': best_silhouette_row['k_max'],
        'best_k': best_silhouette_row["Best K Silhouette"],
        'best_seed': best_silhouette_row['Best Silhouette Seed'],
        'Silhouette Score': best_silhouette_row["Best Silhouette"]
    }

    best_configs[dataset]["Purity Score"] = {
        'best_k_max': best_purity_row['k_max'],
        'best_k': best_purity_row["Best K Purity"],
        'best_seed': best_purity_row['Best Purity Seed'],
        'Purity Score': best_purity_row["Best Purity"]
    }



with open("./results/xmeans_best_results.txt", "w") as file:
    for key, value in best_configs.items():
        file.write(f"{key}:\n")
        for subkey, subvalue in value.items():
            file.write(f"  {subkey}: {subvalue}\n")
        file.write("\n")

#<-------------------------------------------------------------------->

import numpy as np
from preprocessing import DataLoader, DataProcessor
from kmeans.xmeans import XMeans

datasets = ['satimage', 'splice', 'vowel']

metrics = {
    "Adjusted Rand Index": "max",      # Maximized
    "Davies-Bouldin Index": "min",    # Minimized
    "Silhouette Score": "max",        # Maximized
    "Purity Score": "max"             # Maximized
}

dataset_labels = {}

for dataset in datasets:
    # Initialize loaders and preprocessors
    data_loader = DataLoader()
    data_processor = DataProcessor()
    
    # Load dataset using your custom loader
    data, labels = data_loader.load_arff_data(dataset)  # Assuming datasets are in ARFF format
    X = data_processor.preprocess_dataset(data)
    y_true = labels  # Ground truth labels (if needed)
    
    dataset_labels[dataset] = {}
    
    for metric in metrics.keys():  # Iterate over all metrics
        # Get best parameters for the specific metric
        best_k_max = int(best_configs[dataset][metric]['best_k_max'])
        best_seed = int(best_configs[dataset][metric]['best_seed'])
        
        # Apply clustering with best parameters
        xmeans = XMeans(best_k_max, max_iters=100, seed=best_seed)
        y_pred = xmeans.fit_predict(X)
        
        # Store the data and labels for this metric
        dataset_labels[dataset][metric] = {'data': X, 'labels': y_pred}
        # dataset_labels[dataset][metric] = {'data': X, 'labels': y_true}


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA

# Set seaborn style for better aesthetics
sns.set_style('whitegrid')

# Total number of subplots: 2 (rows) x len(datasets)*2 (columns) for 2x2 per dataset in a single row
fig, axes = plt.subplots(2, len(datasets) * 2, figsize=(len(datasets) * 10, 10), sharex=False, sharey=False)

# Set the overall title higher to avoid overlap
fig.suptitle('PCA Clustering Results for All Datasets (2x2 Per Dataset)', fontsize=18, y=1.0)

# Iterate over datasets and metrics
for dataset_idx, dataset in enumerate(datasets):
    # Subplots indices for the dataset
    dataset_axes = axes[:, dataset_idx * 2:(dataset_idx + 1) * 2].flat
    
    # Add a title centered above the 2x2 grid for the current dataset
    x_center = (dataset_idx * 2 + 1) / (len(datasets) * 2)
    fig.text(x_center, 0.96, f'{dataset.capitalize()} Dataset', ha='center', va='center', fontsize=16, weight='bold')
    
    for ax, (metric, _) in zip(dataset_axes, metrics.items()):
        # Retrieve data and labels for the specific metric
        data = dataset_labels[dataset][metric]['data']
        labels = dataset_labels[dataset][metric]['labels']
        
        # Apply PCA
        pca = PCA(n_components=2)
        data_reduced = pca.fit_transform(data)
        variance_ratio = pca.explained_variance_ratio_ * 100  # Variance percentages
        
        # Create a DataFrame for plotting
        plot_df = pd.DataFrame(data_reduced, columns=['Principal Component 1', 'Principal Component 2'])
        plot_df['Cluster'] = labels
        # .astype(int)  # Convert to integers for proper numerical sorting
        
        # Ensure the legend is ordered numerically
        plot_df['Cluster'] = pd.Categorical(plot_df['Cluster'], categories=sorted(plot_df['Cluster'].unique()), ordered=True)
        
        # Plot
        sns.scatterplot(data=plot_df, x='Principal Component 1', y='Principal Component 2',
                        hue='Cluster', palette='tab10', ax=ax, s=50, alpha=0.8, legend='full')

        ax.set_title(f'{metric} Metric')
        ax.set_xlabel(f'Principal Component 1 ({variance_ratio[0]:.2f}% Variance)')
        ax.set_ylabel(f'Principal Component 2 ({variance_ratio[1]:.2f}% Variance)')
        ax.legend(title='Cluster', loc='best', bbox_to_anchor=(1, 1))

        if len(plot_df['Cluster'].unique()) > 16:
            ax.legend_.set_visible(False)
    
# Adjust the layout to leave room for titles
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust top margin to leave space for the suptitle
#plt.show()

plt.savefig('./images/custer_pca.jpg')


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
import numpy as np

# Set seaborn style for better aesthetics
sns.set_style('whitegrid')

# Total number of subplots: 2 (rows) x len(datasets)*2 (columns) for 2x2 per dataset in a single row
fig, axes = plt.subplots(2, len(datasets) * 2, figsize=(len(datasets) * 10, 10), sharex=False, sharey=False)

# Set the overall title higher to avoid overlap
fig.suptitle('Diagonalized Confusion Matrices for All Datasets (2x2 Per Dataset)', fontsize=18, y=1.02)

# Iterate over datasets and metrics
for dataset_idx, dataset in enumerate(datasets):
    # Subplots indices for the dataset
    dataset_axes = axes[:, dataset_idx * 2:(dataset_idx + 1) * 2].flat
    
    # Add a title centered above the 2x2 grid for the current dataset
    x_center = (dataset_idx * 2 + 1) / (len(datasets) * 2)
    fig.text(x_center, 0.96, f'{dataset.capitalize()} Dataset', ha='center', va='center', fontsize=16, weight='bold')
    
    for ax, (metric, _) in zip(dataset_axes, metrics.items()):
        # Retrieve true labels and predicted labels
        data_loader = DataLoader()
        _, y_true = data_loader.load_arff_data(dataset)
        y_pred = dataset_labels[dataset][metric]['labels']
        
        # Map true labels and predicted labels to consecutive integers
        y_true_unique = np.unique(y_true)
        y_pred_unique = np.unique(y_pred)
        
        y_true_mapping = {label: idx for idx, label in enumerate(y_true_unique)}
        y_pred_mapping = {label: idx for idx, label in enumerate(y_pred_unique)}
        
        y_true_mapped = np.array([y_true_mapping[label] for label in y_true])
        y_pred_mapped = np.array([y_pred_mapping[label] for label in y_pred])
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true_mapped, y_pred_mapped)
        
        # Apply the Hungarian algorithm to find the best assignment
        cost_matrix = cm.max() - cm  # Convert to cost matrix for maximization
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Reorder the confusion matrix
        cm_reordered = cm[:, col_ind]
        
        # Remove rows and columns that sum to 0
        non_zero_rows = np.any(cm_reordered > 0, axis=1)
        non_zero_cols = np.any(cm_reordered > 0, axis=0)
        cm_reordered = cm_reordered[non_zero_rows, :][:, non_zero_cols]
        
        # Normalize confusion matrix to [0,1]
        cm_row_sums = cm_reordered.sum(axis=1, keepdims=True)
        cm_row_sums[cm_row_sums == 0] = 1  # Avoid division by zero
        cm_normalized = cm_reordered.astype('float') / cm_row_sums
        
        # Generate labels for heatmap axes
        true_labels_names = [str(y_true_unique[i]) for i in range(len(y_true_unique)) if non_zero_rows[i]]
        pred_labels_names = [str(y_pred_unique[col_ind[j]]) for j in range(len(col_ind)) if non_zero_cols[j]]
        
        # Plot heatmap
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=ax,
                    xticklabels=pred_labels_names, yticklabels=true_labels_names, cbar=False)
        ax.set_title(f'{metric} Metric')
        ax.set_xlabel('Predicted Cluster')
        ax.set_ylabel('True Label')

# Adjust the layout to leave room for titles
plt.tight_layout(rect=[0, 0, 1, 0.93])  # Adjust top margin to leave space for the suptitle
#plt.show()

plt.savefig('./images/confussion_matrix.jpg')