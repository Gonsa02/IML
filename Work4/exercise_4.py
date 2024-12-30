from PCA.PCA import imlPCA
from preprocessing.data_loader import DataLoader
from preprocessing.data_processor import DataProcessor
from sklearn.cluster import OPTICS
from sklearn.metrics import adjusted_rand_score
from clustering.kmeans.kmeans import KMeans

def load_and_preprocess_data(data_loader, data_processor, dataset_name):
    df, labels = data_loader.load_arff_data(dataset_name)
    df = data_processor.preprocess_dataset(df)
    return df, labels


def experiment(dataset_name, kmeans, optics):
    data_loader = DataLoader()
    data_processor = DataProcessor()
    pca = imlPCA()

    # Load and preprocess dataset
    df, labels = load_and_preprocess_data(data_loader, data_processor, dataset_name)

    # Apply PCA
    projected_data = pca.fit_transform(df)

    pca.plot_pca_subspace(projected_data, labels, dataset_name)

    max_score = 0
    best_seed = -1
    seeds = [0, 1, 2, 3, 4]

    # Perform KMeans clustering
    for seed in seeds:
        kmeans.seed = seed
        kmeans_labels = kmeans.fit_predict(projected_data)
        score = adjusted_rand_score(labels, kmeans_labels)
        if score > max_score:
            max_score = score
            best_seed = seed

    kmeans.seed = best_seed
    kmeans_labels = kmeans.fit_predict(projected_data)
    pca.plot_pca_subspace(projected_data, kmeans_labels, dataset_name)

    print(adjusted_rand_score(labels, kmeans_labels))
    
    optics_labels = optics.fit_predict(projected_data)
    pca.plot_pca_subspace(projected_data, optics_labels, dataset_name, legend=False)
    print(adjusted_rand_score(labels, optics_labels))

# Main workflow
def main_our_pca_clustering():

    # SATIMAGE

    # Best models Work3
    kmeans = KMeans(k=6, distance='cosine')

    optics = OPTICS(metric='euclidean',
                    algorithm='brute',
                    min_samples=2,
                    n_jobs=-1)
    
    experiment('satimage', kmeans, optics)

    # SPLICE

    # Best models Work3
    kmeans = KMeans(k=4, distance='euclidean')

    optics = OPTICS(metric='l1',
                    algorithm='brute',
                    min_samples=2,
                    n_jobs=-1)
    
    experiment('splice', kmeans, optics)

    # VOWEL

    kmeans = KMeans(k=6, distance='euclidean')

    optics = OPTICS(metric='euclidean',
                    algorithm='brute',
                    min_samples=4,
                    n_jobs=-1)
    
    experiment('vowel', kmeans, optics)


if __name__ == "__main__":
    main_our_pca_clustering()