import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA, IncrementalPCA
import time
import os

from preprocessing.data_loader import DataLoader
from preprocessing.data_processor import DataProcessor

class imlPCA:
    def __init__(self):
        self.explained_variance_ratio_ = None

    def _select_most_informative_features(self, X, labels, num_features):
        """
        Select the top `num_features` most informative features based on mutual information.
        
        Parameters:
            - X (pandas.DataFrame): The input samples.
            - labels (array-like): The target labels.
            - num_features (int): The number of top features to select (2 or 3).
        
        Returns:
            - selected_indices (list): Indices of the selected features.
        """
        if num_features not in [2, 3]:
            raise ValueError("num_features must be either 2 or 3.")
        
        # Compute mutual information between each feature and the labels
        mi = mutual_info_classif(X, labels, discrete_features='auto')
        
        # Get indices of features with highest mutual information
        selected_indices = np.argsort(mi)[-num_features:][::-1]
        
        return selected_indices.tolist()
    

    def plot_original_dataset(self, X, labels, num_features=2): # Step 2
        """
        Plot the original dataset using the top `num_features` most informative features.
        
        Parameters:
        - X (pandas.DataFrame): The input samples.
        - labels (array-like): The target labels.
        - num_features (int): Number of top features to plot (2 or 3).
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame with feature names.")
        
        # Select the most informative features
        feature_indices = self._select_most_informative_features(X, labels, num_features)

        # Retrieve feature names based on selected indices
        feature_names = X.columns[feature_indices].tolist()

        plt.figure(figsize=(8, 6))

        if num_features == 2:
            for label in set(labels):
                mask = (labels == label)
                plt.scatter(
                    X.values[mask, feature_indices[0]],
                    X.values[mask, feature_indices[1]],
                    label=str(label),
                    alpha=0.7
                )
            plt.xlabel(feature_names[0])
            plt.ylabel(feature_names[1])

        elif num_features == 3:
            from mpl_toolkits.mplot3d import Axes3D
            ax = plt.axes(projection='3d')
            for label in set(labels):
                mask = (labels == label)
                ax.scatter(
                    X.values[mask, feature_indices[0]],
                    X.values[mask, feature_indices[1]],
                    X.values[mask, feature_indices[2]],
                    label=str(label),
                    alpha=0.7
                )
            ax.set_xlabel(feature_names[0])
            ax.set_ylabel(feature_names[1])
            ax.set_zlabel(feature_names[2])

        else:
            raise ValueError("`num_features` must be 2 or 3.")
        
        plt.legend()
        plt.title(f"Original Data with {num_features} Most Informative Features")
        plt.grid(True)
        plt.show()


    def _compute_mean_vector(self, X): # Step 3
        return np.mean(X, axis=0)


    def _compute_covariance_matrix(self, X, mean_vec): # Step 4
        """
        Computes the covariance matrix of the dataset X.
        """
        n, _ = X.shape
        X_centered = X - mean_vec
        cov_matrix = (X_centered.T @ X_centered) / (n - 1)
        return cov_matrix


    def _eigen_decomposition(self, cov_matrix): # Step 5
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        return eigenvalues, eigenvectors
    

    def _sort_eigens(self, eigenvalues, eigenvectors, k=None, cumulative_threshold=0.85): # Step 6
        """
        Sorts eigenvalues in descending order along with their corresponding eigenvectors.
        If k is specified, selects the top k eigenvectors.
        If k is not specified, selects the minimum number of eigenvectors required to reach a cumulative explained variance of 85%.

        Parameters:
            - eigenvalues (numpy.ndarray): Array of eigenvalues.
            - eigenvectors (numpy.ndarray): Matrix of eigenvectors.
            - k (int, optional): Number of top eigenvectors to select. Defaults to None.
        
        Returns:
            - sorted_eigenvalues (numpy.ndarray): Sorted eigenvalues.
            - selected_eigenvectors (numpy.ndarray): Selected eigenvectors corresponding to the sorted eigenvalues.
        """
        # 1. Make pairs of (eigenvalue, eigenvector)
        eig_pairs = [(eigenvalues[i], eigenvectors[:, i]) for i in range(len(eigenvalues))]

        # 2. Sort by absolute value of eigenvalue (descending)
        eig_pairs.sort(key=lambda x: np.abs(x[0]), reverse=True)

        # 3. Separate them back
        sorted_eigenvalues =  np.array([pair[0] for pair in eig_pairs])
        sorted_eigenvectors = np.array([pair[1] for pair in eig_pairs]).T

        if k is not None:
            if k > len(sorted_eigenvalues):
                raise ValueError(f"k={k} is greater than the number of available eigenvalues ({len(sorted_eigenvalues)}).")
            # 4a. Select the top k eigenvalues and eigenvectors
            selected_eigenvalues = sorted_eigenvalues[:k]
            selected_eigenvectors = sorted_eigenvectors[:, :k]
        
        else:
            # 4b. Calculate cumulative explained variance
            explained_variances = sorted_eigenvalues / np.sum(sorted_eigenvalues)
            cumulative_variances = np.cumsum(explained_variances)
            
            # 4c. Determine the number of components to reach cumulative_threshold% variance
            k = np.argmax(cumulative_variances >= cumulative_threshold) + 1
            selected_eigenvalues = sorted_eigenvalues[:k]
            selected_eigenvectors = sorted_eigenvectors[:, :k]
            print(f"Selected top {k} eigenvectors to reach at least {cumulative_threshold*100}% (reached {cumulative_variances[k-1]:.2%}) cumulative explained variance.\n")

        # 5. (Extra) Store explained variance ratio
        self.explained_variance_ratio_ = selected_eigenvalues / np.sum(eigenvalues)

        return selected_eigenvalues, selected_eigenvectors


    def _project_data(self, X, mean_vec, W): # Step 7
        """
        Projects the data X onto the subspace formed by W.
        X shape: (n, d), W shape: (d, k)
        """
        if X.shape[1] != mean_vec.shape[0]:
            raise ValueError("Number of features in X and mean vector must match.")
    
        if W.shape[0] != X.shape[1]:
            raise ValueError("Number of rows in W must match number of features in X.")
    
        X_centered = X - mean_vec
        X_projected = X_centered @ W

        return X_projected
    
    
    def plot_pca_subspace(self, X_projected, labels, dataset_name, num_components=2, legend=True): # Step 8
        """
        Plots the 2D data from the projected subspace.
        """
        # Create a mirrored version of Custom PCA results by flipping the first principal component
        mirrored_X_projected = X_projected.copy()
        if dataset_name == "satimage":
            mirrored_X_projected.loc[:, 0] *= -1  # Flip the x-axis for mirroring
        elif dataset_name == "splice":
            mirrored_X_projected.loc[:, 1] *= -1  # Flip the y-axis for mirroring
        elif dataset_name != "vowel":
            raise ValueError(f"Dataset name must be 'satimage', 'splice', or 'vowel', right now it's '{dataset_name}'.")

        if num_components == 2:
                
            plt.figure(figsize=(10, 8))
            sns.scatterplot(
                data=mirrored_X_projected,
                x=mirrored_X_projected.columns[0],
                y=mirrored_X_projected.columns[1],
                hue=labels,
                palette='tab10',
                s=60,
                alpha=0.8,
                edgecolor='k',
                legend=legend
            )
            
            plt.title(f"{dataset_name} Dataset", fontsize=16)
            plt.xlabel(f'Principal Component 1 ({self.explained_variance_ratio_[0] * 100:.2f}% Variance)', fontsize=14)
            plt.ylabel(f'Principal Component 2 ({self.explained_variance_ratio_[1] * 100:.2f}% Variance)', fontsize=14)
            if legend:
                plt.legend(title='Cluster', fontsize=12, title_fontsize=12)
            plt.tight_layout()
            plt.show()

        else:
            raise ValueError("Can only plot 2D PCA subspaces.")

    def _reconstruct_data(self, X_projected, mean_vec, W): # Step 9
        """
        Reconstructs the data from the projected subspace back to original dimension.
        """
        return X_projected @ W.T + mean_vec
    
    def plot_original_and_reconstructed_dataset(self, X_original, X_reconstructed, labels, num_features=2, cumulative_threshold=0.85):
        """
        Plots the original and reconstructed datasets side by side using the top `num_features` most informative features.
        
        Parameters:
        - X_original (pandas.DataFrame): The original input samples with feature names.
        - X_reconstructed (pandas.DataFrame or numpy.ndarray): The reconstructed input samples.
        - labels (array-like): The target labels.
        - num_features (int): Number of top features to plot (default is 2).
        """

        # Validate original dataset
        if not isinstance(X_original, pd.DataFrame):
            raise TypeError("X_original must be a pandas DataFrame with feature names.")
        
        # Handle reconstructed dataset
        if isinstance(X_reconstructed, np.ndarray):
            # Assume columns are in the same order and assign original column names
            X_reconstructed = pd.DataFrame(X_reconstructed, columns=X_original.columns)
        elif isinstance(X_reconstructed, pd.DataFrame):
            # If columns are numbered (0, 1, 2, ...), assign original column names
            if list(X_reconstructed.columns) == list(range(X_reconstructed.shape[1])):
                X_reconstructed.columns = X_original.columns
        else:
            raise TypeError("X_reconstructed must be a pandas DataFrame or numpy.ndarray.")
        
        # Select the most informative features
        feature_indices = self._select_most_informative_features(X_original, labels, num_features)
        feature_names = X_original.columns[feature_indices].tolist()

        # Create a color map based on unique labels
        unique_labels = np.unique(labels)
        colors = plt.cm.get_cmap('tab10', len(unique_labels))
        label_color_dict = {label: colors(i) for i, label in enumerate(unique_labels)}
        
        # Calculate shared axis limits
        x_min = min(X_original[feature_names[0]].min(), X_reconstructed[feature_names[0]].min()) - 0.5
        x_max = max(X_original[feature_names[0]].max(), X_reconstructed[feature_names[0]].max()) + 0.5
        y_min = min(X_original[feature_names[1]].min(), X_reconstructed[feature_names[1]].min()) - 0.5
        y_max = max(X_original[feature_names[1]].max(), X_reconstructed[feature_names[1]].max()) + 0.5
        
        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot Original Dataset
        ax = axes[0]
        for label in np.unique(labels):
            mask = (labels == label)
            ax.scatter(
                X_original.loc[mask, feature_names[0]],
                X_original.loc[mask, feature_names[1]],
                label=str(label),
                alpha=0.7,
                color=label_color_dict[label]
            )
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
        ax.set_title("Original Dataset")
        ax.legend(title="Labels")
        
        # Plot Reconstructed Dataset
        ax = axes[1]
        for label in np.unique(labels):
            mask = (labels == label)
            ax.scatter(
                X_reconstructed.loc[mask, feature_names[0]],
                X_reconstructed.loc[mask, feature_names[1]],
                label=str(label),
                alpha=0.7,
                color=label_color_dict[label]
            )
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
        ax.set_title(f"Reconstructed Dataset ({cumulative_threshold*100}% Cumulative Variance)")
        ax.legend(title="Labels")
        
        plt.tight_layout()
        plt.show()



    def fit_transform(self, X, cumulative_threshold=0.85):
        """
        Fit the model with X and apply the dimensionality reduction on X.

        Parameters:
            - X (pd.DataFrame): The input data to fit and transform.
            - cumulative_threshold (float): The threshold for cumulative explained variance to determine the number of components.

        Returns:
            - X_projected (np.ndarray): The transformed data in the PCA subspace.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected input X to be a pandas DataFrame, but got {type(X).__name__} instead.")

        # Step 3: Compute mean vector
        mean_vec = self._compute_mean_vector(X.values)

        # Step 4: Compute covariance matrix
        cov_matrix = self._compute_covariance_matrix(X, mean_vec)

        # Step 5: Eigen decomposition
        eigenvalues, eigenvectors = self._eigen_decomposition(cov_matrix)

        # Step 6: Sort eigenvectors
        sorted_eigenvalues, sorted_eigenvectors = self._sort_eigens(eigenvalues, eigenvectors, cumulative_threshold=cumulative_threshold)

        X_projected = self._project_data(X, mean_vec, sorted_eigenvectors)

        return X_projected


def main_PCA():
    # Initialize PCA
    pca = imlPCA()

    ## --- Step 1 --- ##
    # Initialize DataLoader and DataProcessor
    data_loader    = DataLoader()
    data_processor = DataProcessor()

    # Load Datasets
    df_satimage, labels_satimage = data_loader.load_arff_data('satimage')
    df_splice,   labels_splice   = data_loader.load_arff_data('splice')

    # Preprocess Datasets
    df_satimage = data_processor.preprocess_dataset(df_satimage)
    df_splice   = data_processor.preprocess_dataset(df_splice)

    ## --- Step 2: Plot Original Datasets --- ##
    print("\n--- Step 2: Plotting Original Satimage Dataset ---")
    pca.plot_original_dataset(df_satimage, labels_satimage)

    print("\n--- Step 2: Plotting Original Splice Dataset ---")
    pca.plot_original_dataset(df_splice, labels_splice)

    ## --- Step 3: Compute Mean Vectors --- ##
    print("\n--- Step 3: Computing Mean Vectors ---")
    mean_vec_satimage = pca._compute_mean_vector(df_satimage.values)
    mean_vec_splice   = pca._compute_mean_vector(df_splice.values)

    ## --- Step 4: Compute Covariance Matrices --- ##
    print("\n--- Step 4: Computing Covariance Matrices ---")
    cov_matrix_satimage = pca._compute_covariance_matrix(df_satimage, mean_vec_satimage)
    print("Satimage Covariance Matrix:\n", cov_matrix_satimage)

    cov_matrix_splice = pca._compute_covariance_matrix(df_splice, mean_vec_splice)
    print("\nSplice Covariance Matrix:\n", cov_matrix_splice)

    ## --- Step 5: Calculate Eigenvectors --- ##
    print("\n--- Step 5: Calculating Eigenvectors and Eigenvalues ---")
    eigenvalues_satimage, eigenvectors_satimage = pca._eigen_decomposition(cov_matrix_satimage)
    print("Satimage Eigenvalues:\n", eigenvalues_satimage)
    print("\nSatimage Eigenvectors:\n", eigenvectors_satimage)
    
    eigenvalues_splice, eigenvectors_splice = pca._eigen_decomposition(cov_matrix_splice)
    print("\n\nSplice Eigenvalues:\n", eigenvalues_splice)
    print("\nSplice Eigenvectors:\n", eigenvectors_splice)
    
    ## --- Step 6: Sort Eigenvectors --- ##
    print("\n--- Step 6: Sorting Eigenvectors ---")

    sorted_eigenvalues_satimage, sorted_eigenvectors_satimage = pca._sort_eigens(eigenvalues_satimage, eigenvectors_satimage)
    print(f"Satimage Sorted Eigenvalues:\n", sorted_eigenvalues_satimage)
    print(f"\nSatimage Sorted Eigenvectors:\n", sorted_eigenvectors_satimage)
    
    sorted_eigenvalues_splice, sorted_eigenvectors_splice = pca._sort_eigens(eigenvalues_splice, eigenvectors_splice)
    print(f"Splice Sorted Eigenvalues:\n", sorted_eigenvalues_splice)
    print(f"\nSplice Sorted Eigenvectors:\n", sorted_eigenvectors_splice)

    ## --- Step 7: Derive New Datasets --- ##
    print("\n--- Step 7: Projecting Data onto New Subspace ---")
    projected_satimage = pca._project_data(df_satimage, mean_vec_satimage, sorted_eigenvectors_satimage)
    projected_splice   = pca._project_data(df_splice,   mean_vec_splice, sorted_eigenvectors_splice)

    ## --- Step 8: Plot New Subspaces --- ##
    print("\n--- Step 8: Plotting PCA Subspace for Satimage ---")
    pca.plot_pca_subspace(projected_satimage, labels_satimage, 'Satimage')

    print("\n--- Step 8: Plotting PCA Subspace for Splice ---")
    pca.plot_pca_subspace(projected_splice, labels_splice, 'Splice')

    ## --- Step 9: Reconstruct and Plot Datasets --- ##
    print("\n--- Step 9: Reconstructing Data from PCA Subspace ---")
    reconstructed_satimage = pca._reconstruct_data(projected_satimage, mean_vec_satimage, sorted_eigenvectors_satimage)
    reconstructed_splice   = pca._reconstruct_data(projected_splice,   mean_vec_splice,   sorted_eigenvectors_splice)

    mse_satimage = mean_squared_error(df_satimage, reconstructed_satimage)
    print(f"Mean Squared Error between original and reconstructed Satimage data: {mse_satimage:.4f}")

    mse_splice = mean_squared_error(df_satimage, reconstructed_satimage)
    print(f"Mean Squared Error between original and reconstructed Splice data: {mse_splice:.4f}")

    # Plot Reconstructed Data vs Original Data for comparison
    print("\n--- Plotting Reconstructed vs Original Satimage Data ---")
    pca.plot_original_and_reconstructed_dataset(df_satimage, reconstructed_satimage, labels_satimage, 2)

    print("\n--- Plotting Reconstructed vs Original Splice Data ---")
    pca.plot_original_and_reconstructed_dataset(df_splice, reconstructed_splice, labels_splice, 2)

def main_incremental_PCA():

    data_loader = DataLoader()
    data_processor = DataProcessor()

    # Create results directory
    os.makedirs('results', exist_ok=True)

    # Initialize your custom PCA
    custom_pca = imlPCA()

    # Load Datasets
    datasets_info = {
        'satimage': data_loader.load_arff_data('satimage'),
        'splice': data_loader.load_arff_data('splice'),
    }

    # Dictionary to store fit times
    fit_times = {}

    # Preprocess and perform PCA & Incremental PCA & Custom PCA
    preprocessed_datasets = {}
    for dataset_name, (df, labels) in datasets_info.items():
        print(f"\nProcessing dataset: {dataset_name}")
        preprocessed_df = data_processor.preprocess_dataset(df)

        # Initialize a dictionary to store times for this dataset
        fit_times[dataset_name] = {}

        # --- Custom PCA ---
        start_time = time.time()
        custom_pca_result = custom_pca.fit_transform(preprocessed_df)
        end_time = time.time()
        fit_times[dataset_name]['Custom PCA'] = end_time - start_time
        
        # If custom_pca_result is a DataFrame, convert it to a NumPy array
        if isinstance(custom_pca_result, pd.DataFrame):
            custom_pca_result = custom_pca_result.values

        # Obtain the explained variance ratio (in %) from your custom PCA
        custom_pca_variance_ratio = custom_pca.explained_variance_ratio_ * 100

        # Determine number of components for 85% explained variance
        cumulative_variance = custom_pca.explained_variance_ratio_.cumsum()
        n_components_85 = np.argmax(cumulative_variance >= 0.85) + 1
        print(f"Number of components to reach 85% variance for {dataset_name}: {n_components_85}")
        
        # Save Custom PCA results to CSV
        custom_pca_df = pd.DataFrame(custom_pca_result, columns=[f'Custom_PC{i+1}' for i in range(custom_pca_result.shape[1])])
        custom_pca_df['label'] = labels
        
        output_file_custom_pca = os.path.join('results', f'{dataset_name}_custom_pca_results.csv')
        custom_pca_df.to_csv(output_file_custom_pca, index=False)
        print(f"Custom PCA results saved to {output_file_custom_pca}")
        
        # --- Sklearn PCA ---
        start_time = time.time()
        standard_pca = PCA(n_components=n_components_85)
        pca_result = standard_pca.fit_transform(preprocessed_df)
        end_time = time.time()
        fit_times[dataset_name]['Sklearn PCA'] = end_time - start_time

        # Calculate explained variance ratio (in %)
        pca_variance_ratio = standard_pca.explained_variance_ratio_ * 100
        
        # Save Sklearn PCA results to CSV
        pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(pca_result.shape[1])])
        pca_df['label'] = labels
        
        output_file_pca = os.path.join('results', f'{dataset_name}_sklearn_pca_results.csv')
        pca_df.to_csv(output_file_pca, index=False)
        print(f"Sklearn PCA results saved to {output_file_pca}")
        
        # --- Incremental PCA ---
        start_time = time.time()
        ipca = IncrementalPCA(n_components=n_components_85)
        ipca_result = ipca.fit_transform(preprocessed_df)
        end_time = time.time()
        fit_times[dataset_name]['Incremental PCA'] = end_time - start_time

        # Calculate explained variance ratio (in %)
        ipca_variance_ratio = ipca.explained_variance_ratio_ * 100
        
        # Save Incremental PCA results to CSV
        ipca_df = pd.DataFrame(ipca_result, columns=[f'PC{i+1}' for i in range(ipca_result.shape[1])])
        ipca_df['label'] = labels
        
        output_file_ipca = os.path.join('results', f'{dataset_name}_sklearn_incremental_pca_results.csv')
        ipca_df.to_csv(output_file_ipca, index=False)
        print(f"Incremental PCA results saved to {output_file_ipca}")
        
        # Store all PCA results and labels in the dictionary for plotting
        preprocessed_datasets[dataset_name] = {
            'custom_pca_result': custom_pca_result,
            'custom_pca_variance_ratio': custom_pca_variance_ratio,
            'sklearn_pca_result': pca_result,
            'sklearn_pca_variance_ratio': pca_variance_ratio,
            'sklearn_ipca_result': ipca_result,
            'sklearn_ipca_variance_ratio': ipca_variance_ratio,
            'labels': labels
        }

    # Print the fit times
    print("\nFit times (in seconds) for each dataset and implementation:")
    for dataset_name, times in fit_times.items():
        print(f"\nDataset: {dataset_name}")
        for method, duration in times.items():
            print(f"{method}: {duration:.4f} seconds")

if __name__ == '__main__':
    main_PCA()