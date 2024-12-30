import numpy as np
import pandas as pd

results = pd.read_csv("results/kmeans_results.csv")
datasets = ['satimage', 'splice', 'vowel']
best_values = []

for dataset_names in datasets:
    filtered_df = results[(results['Dataset'] == dataset_names)]

    best_ari = filtered_df.loc[filtered_df['ARI'].idxmax()]
    best_silhouette = filtered_df.loc[filtered_df['Silhouette'].idxmax()]
    best_DBI = filtered_df.loc[filtered_df['DBI'].idxmax()]

    best_values.append(best_ari)
    best_values.append(best_silhouette)
    best_values.append(best_DBI)


pd.DataFrame(best_values).to_csv(
    'results/kmeans_best_parameteres.csv', index=False)
