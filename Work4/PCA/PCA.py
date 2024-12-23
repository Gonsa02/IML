import numpy as np
import matplotlib.pyplot as plt

from preprocessing.data_loader import DataLoader
from preprocessing.data_processor import DataProcessor

class imlPCA:
    def plot_original_dataset(self, X, labels, feature_indices=[0, 1]): # Step 2
        plt.figure(figsize=(6, 5))

        if len(feature_indices) == 2:
            for label in set(labels):
                mask = (labels == label)
                plt.scatter(X[mask, feature_indices[0]], X[mask, feature_indices[1]], label=str(label), alpha=0.7)
            plt.xlabel(f"Feature {feature_indices[0]+1}")
            plt.ylabel(f"Feature {feature_indices[1]+1}")

        elif len(feature_indices) == 3:
            from mpl_toolkits.mplot3d import Axes3D
            ax = plt.axes(projection='3d')
            for label in set(labels):
                mask = (labels == label)
                ax.scatter(X[mask, feature_indices[0]], X[mask, feature_indices[1]], X[mask, feature_indices[2]], label=str(label), alpha=0.7)
            ax.set_xlabel(f"Feature {feature_indices[0]+1}")
            ax.set_ylabel(f"Feature {feature_indices[1]+1}")
            ax.set_zlabel(f"Feature {feature_indices[2]+1}")

        else:
            raise ValueError("feature_indices must be a list of 2 or 3 integers.")
        
        plt.legend()
        plt.title("Original Data")
        plt.grid(True)
        plt.show()


    def compute_mean_vector(self, X): # Step 3
        return np.mean(X, axis=0)


    def compute_covariance_matrix(self, X, mean_vec): # Step 4
        """
        Computes the covariance matrix of the dataset X.
        """
        n, _ = X.shape
        X_centered = X - mean_vec
        cov_matrix = (X_centered.T @ X_centered) / (n - 1)
        return cov_matrix


    def eigen_decomposition(self, cov_matrix): # Step 5
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        return eigenvalues, eigenvectors
    

    def sort_eigens(self, eigenvalues, eigenvectors, k): # Step 6
        """
        Sorts eigenvalues in descending order along with their corresponding eigenvectors and selects the top k eigenvectors.
        """
        # 1. Make pairs of (eigenvalue, eigenvector)
        eig_pairs = [(eigenvalues[i], eigenvectors[:, i]) for i in range(len(eigenvalues))]

        # 2. Sort by absolute value of eigenvalue (descending)
        eig_pairs.sort(key=lambda x: np.abs(x[0]), reverse=True)

        # 3. Separate them back
        sorted_eigenvalues = [pair[0] for pair in eig_pairs]
        sorted_eigenvectors = [pair[1] for pair in eig_pairs]

        # 4. Select k eigenvectors
        sorted_eigenvalues = sorted_eigenvalues[:k]
        sorted_eigenvectors = sorted_eigenvectors[:k]

        return np.array(sorted_eigenvalues), np.array(sorted_eigenvectors).T


    def project_data(self, X, mean_vec, W): # Step 7
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
    
    
    def plot_pca_subspace(self, X_projected, labels=None): # Step 8
        """
        Plots the 2D or 3D data from the projected subspace.
        """
        num_components = X_projected.shape[1]
        plt.figure(figsize=(6, 5))
        
        if num_components == 2:
            if labels is not None and len(labels) == len(X_projected):
                for label in set(labels):
                    mask = (labels == label)
                    plt.scatter(X_projected[mask, 0],
                                X_projected[mask, 1],
                                label=str(label), alpha=0.7)
                plt.legend()
            else:
                plt.scatter(X_projected[:, 0], X_projected[:, 1], alpha=0.7)
            
            plt.xlabel("Principal Component 1")
            plt.ylabel("Principal Component 2")

        elif num_components == 3:
            from mpl_toolkits.mplot3d import Axes3D
            ax = plt.axes(projection='3d')
            if labels is not None and len(labels) == len(X_projected):
                for label in set(labels):
                    mask = (labels == label)
                    ax.scatter(X_projected[mask, 0],
                            X_projected[mask, 1],
                            X_projected[mask, 2],
                            label=str(label), alpha=0.7)
                ax.legend()
            else:
                ax.scatter(X_projected[:, 0], X_projected[:, 1], X_projected[:, 2], alpha=0.7)
            
            ax.set_xlabel("Principal Component 1")
            ax.set_ylabel("Principal Component 2")
            ax.set_zlabel("Principal Component 3")

        else:
            raise ValueError("Can only plot 2D or 3D PCA subspaces.")
        
        plt.title("Data in PCA Subspace")
        plt.grid(True)
        plt.show()

    def reconstruct_data(self, X_projected, mean_vec, W): # Step 9
        """
        Reconstructs the data from the projected subspace back to original dimension.
        """
        return X_projected @ W.T + mean_vec


def main():
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
    X_satimage = df_satimage.values
    pca.plot_original_dataset(X_satimage, labels_satimage, feature_indices=[0, 1])

    print("\n--- Step 2: Plotting Original Splice Dataset ---")
    X_splice = df_splice.values
    pca.plot_original_dataset(X_splice, labels_splice, feature_indices=[0, 1])

    ## --- Step 3: Compute Mean Vectors --- ##
    print("\n--- Step 3: Computing Mean Vectors ---")
    mean_vec_satimage = pca.compute_mean_vector(df_satimage)
    mean_vec_splice   = pca.compute_mean_vector(df_splice)

    ## --- Step 4: Compute Covariance Matrices --- ##
    print("\n--- Step 4: Computing Covariance Matrices ---")
    cov_matrix_satimage = pca.compute_covariance_matrix(df_satimage, mean_vec_satimage)
    print("Satimage Covariance Matrix:\n", cov_matrix_satimage)

    cov_matrix_splice = pca.compute_covariance_matrix(df_splice, mean_vec_splice)
    print("\nSplice Covariance Matrix:\n", cov_matrix_splice)

    ## --- Step 5: Calculate Eigenvectors --- ##
    print("\n--- Step 5: Calculating Eigenvectors and Eigenvalues ---")
    eigenvalues_satimage, eigenvectors_satimage = pca.eigen_decomposition(cov_matrix_satimage)
    print("Satimage Eigenvalues:\n", eigenvalues_satimage)
    print("\nSatimage Eigenvectors:\n", eigenvectors_satimage)
    
    eigenvalues_splice, eigenvectors_splice = pca.eigen_decomposition(cov_matrix_splice)
    print("\n\nSplice Eigenvalues:\n", eigenvalues_splice)
    print("\nSplice Eigenvectors:\n", eigenvectors_splice)
    
    ## --- Step 6: Sort Eigenvectors --- ##
    print("\n--- Step 6: Sorting Eigenvectors ---")
    k = 2
    
    sorted_eigenvalues_satimage, sorted_eigenvectors_satimage = pca.sort_eigens(eigenvalues_satimage, eigenvectors_satimage, k)
    print(f"Top {k} Satimage Sorted Eigenvalues:\n", sorted_eigenvalues_satimage)
    print(f"\nTop {k} Satimage Sorted Eigenvectors:\n", sorted_eigenvectors_satimage)
    
    sorted_eigenvalues_splice, sorted_eigenvectors_splice = pca.sort_eigens(eigenvalues_splice, eigenvectors_splice, k)
    print(f"\n\nTop {k} Splice Sorted Eigenvalues:\n", sorted_eigenvalues_splice)
    print(f"\nTop {k} Splice Sorted Eigenvectors:\n", sorted_eigenvectors_splice)

    ## --- Step 7: Derive New Datasets --- ##
    print("\n--- Step 7: Projecting Data onto New Subspace ---")
    projected_satimage = pca.project_data(df_satimage, mean_vec_satimage, sorted_eigenvectors_satimage)
    projected_splice   = pca.project_data(df_splice,   mean_vec_splice, sorted_eigenvectors_splice)

    ## --- Step 8: Plot New Subspaces --- ##
    print("\n--- Step 8: Plotting PCA Subspace for Satimage ---")
    X_projected_satimage = projected_satimage.values
    pca.plot_pca_subspace(X_projected_satimage, labels_satimage)

    print("\n--- Step 8: Plotting PCA Subspace for Splice ---")
    X_projected_splice = projected_splice.values
    pca.plot_pca_subspace(X_projected_splice, labels_splice)

    ## --- Step 9: Reconstruct and Plot Datasets --- ##
    print("\n--- Step 9: Reconstructing Data from PCA Subspace ---")
    reconstructed_satimage = pca.reconstruct_data(projected_satimage, mean_vec_satimage, sorted_eigenvectors_satimage)
    reconstructed_splice   = pca.reconstruct_data(projected_splice,   mean_vec_splice,   sorted_eigenvectors_splice)

    # Plot Reconstructed Data vs Original Data for comparison
    #print("\n--- Plotting Reconstructed vs Original Satimage Data ---")
    #pca.plot_reconstructed_data(df_satimage, reconstructed_satimage, feature_indices=[0, 1])

    #print("\n--- Plotting Reconstructed vs Original Splice Data ---")
    #pca.plot_reconstructed_data(df_splice, reconstructed_splice, feature_indices=[0, 1])


if __name__ == '__main__':
    main()