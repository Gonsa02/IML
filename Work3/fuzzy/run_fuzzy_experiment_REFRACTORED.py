def run_fuzzy_experiment():
    import os
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import OrderedDict
    from itertools import product
    from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score, contingency_matrix
    from scipy.optimize import linear_sum_assignment
    from tqdm import tqdm
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from preprocessing import DataLoader, DataProcessor  # Assuming your preprocessing scripts
    from fuzzy import GSFuzzyCMeans  # Assuming the class implementation from earlier

    def define_parameters():
        """Define the parameters and metrics."""
        fuzzy_exponents = [1.5, 2.5, 3.5]  # Testing more values for m
        xi_values = [0.3, 0.5, 0.7]        # Testing more values for xi
        k_values = range(2, 16)            # Testing k from 2 to 15 clusters
        parameter_combinations = list(product(k_values, fuzzy_exponents, xi_values))
        print(f"Total parameter combinations: {len(parameter_combinations)}")

        metrics = {
            "Adjusted Rand Index": adjusted_rand_score,
            "Davies-Bouldin Index": davies_bouldin_score,
            "Silhouette Score": silhouette_score,
            "Time (s)": "min",
            "Iterations": "min",
        }

        metric_direction = {
            "Adjusted Rand Index": "max",
            "Silhouette Score": "max",
            "Davies-Bouldin Index": "min",
            "Time (s)": "min",
            "Iterations": "min",
        }

        return parameter_combinations, metrics, metric_direction

    def run_experiment(args, data_loader, data_processor):
        """Run an individual experiment."""
        dataset_name, (k, m, xi) = args

        try:
            # Load and preprocess the dataset
            data, labels = data_loader.load_arff_data(dataset_name)
            X = data_processor.preprocess_dataset(data)
            y_true = labels

            # Initialize accumulators for metrics
            total_scores = {key: 0.0 for key in ["Adjusted Rand Index", "Davies-Bouldin Index", "Silhouette Score", "Time (s)", "Iterations"]}
            num_runs = 3  # Number of times to run each experiment

            for _ in range(num_runs):
                start_time = time.time()
                gs_fcm = GSFuzzyCMeans(n_clusters=k, m=m, xi=xi, max_iter=100, random_state=None)
                gs_fcm.fit(X)
                y_pred = gs_fcm.predict(X)
                elapsed_time = time.time() - start_time

                total_scores["Adjusted Rand Index"] += adjusted_rand_score(y_true, y_pred)
                total_scores["Davies-Bouldin Index"] += davies_bouldin_score(X, y_pred)
                total_scores["Silhouette Score"] += silhouette_score(X, y_pred)
                total_scores["Time (s)"] += elapsed_time
                total_scores["Iterations"] += gs_fcm.n_iter_

            # Average the metrics
            averaged_scores = {key: total / num_runs for key, total in total_scores.items()}
            averaged_scores.update({"k": k, "m": m, "xi": xi})
            return dataset_name, averaged_scores

        except Exception as e:
            return dataset_name, e

    def process_datasets(datasets, parameter_combinations, metrics, metric_direction):
        """Process all datasets."""
        data_loader = DataLoader()
        data_processor = DataProcessor()

        for dataset_name in datasets:
            print(f"\nRunning experiments for {dataset_name} dataset...")
            results_file = f"results/FUZZY_{dataset_name}_results_AVG.csv"

            try:
                existing_df = pd.read_csv(results_file)
            except FileNotFoundError:
                existing_df = pd.DataFrame()

            # Run experiments in parallel
            experiments = [(dataset_name, params) for params in parameter_combinations]
            results_list = []

            with ProcessPoolExecutor() as executor:
                futures = {executor.submit(run_experiment, args, data_loader, data_processor): args for args in experiments}
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {dataset_name}"):
                    _, result = future.result()
                    if not isinstance(result, Exception):
                        results_list.append(result)

            if results_list:
                new_results_df = pd.DataFrame([result[1] for result in results_list])
                existing_df = pd.concat([existing_df, new_results_df], ignore_index=True)

            existing_df.to_csv(results_file, index=False)
            print(f"Results saved to {results_file}")

    def plot_metrics(results, number_of_classes, metrics, metric_direction):
        """Plot the lineplots for metrics."""
        for metric_key in metrics.keys():
            fig, axes = plt.subplots(1, 3, figsize=(21, 6))
            for ax, (dataset, df) in zip(axes, results.items()):
                sns.lineplot(data=df, x="k", y=metric_key, hue="m", style="xi", markers=True, dashes=False, ax=ax, linewidth=1)
                ax.set_title(f'{dataset.capitalize()} Dataset')
                ax.set_xlabel('Number of Clusters (k)')
                ax.set_ylabel(metrics[metric_key])
                ax.axvline(x=number_of_classes[dataset], color="red", linestyle="--", linewidth=1, label="True number of classes")
            plt.tight_layout()
            plt.show()

    # Main execution
    parameter_combinations, metrics, metric_direction = define_parameters()
    datasets = ["vowel", "splice", "satimage"]
    process_datasets(datasets, parameter_combinations, metrics, metric_direction)
