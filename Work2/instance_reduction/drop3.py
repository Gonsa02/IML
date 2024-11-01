import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from instance_reduction.ennth import ENNTh


class DROP3:
    # k usually 1, 3 or 5 (section 4)
    def __init__(self, dataset, labels, k=7, k_ennth=7, threshold=0.8):
        self.T = dataset
        self.columns = dataset.columns
        self.T_labels = labels
        self.k = k
        self.k_ennth = k_ennth
        self.thresh = threshold

    # Step 1: Remove noisy instances (ENNTh algorithm)
    def _noise_filtering(self):
        ennth_algorithm = ENNTh(self.T, self.T_labels,
                                self.k_ennth, self.thresh)
        return ennth_algorithm.fit()

    def fit(self):
        self.S, self.S_labels = self._noise_filtering()
        self.S = self.S.to_numpy()
        self.S_labels = self.S_labels.to_numpy()

        # Distances between samples
        distances = squareform(pdist(self.S, metric=self._minkowski_distance))
        # for each sample, the idx of the distances in increasing order.
        nn_idxs_matrix = np.argsort(distances, axis=1)

        # distance of the first enemy for each sample
        min_enemy_distances = np.zeros(len(self.S_labels))

        for sample_idx, nn_idxs in enumerate(nn_idxs_matrix):
            nn_idxs = nn_idxs[1:]
            sample_label = self.S_labels[sample_idx]

            for idx in nn_idxs:
                if sample_label != self.S_labels[idx]:
                    min_enemy_distances[sample_idx] = distances[sample_idx, idx]
                    break

        # idx samples in decreasing order
        sorted_enemy_distances = np.argsort(-min_enemy_distances).astype(int)

        knn = {sample_idx: nn_idxs_matrix[sample_idx][1:self.k+2]
               for sample_idx in sorted_enemy_distances}
        associates = {sample_idx: set()
                      for sample_idx in sorted_enemy_distances}

        for sample_idx in sorted_enemy_distances:
            knn[sample_idx] = nn_idxs_matrix[sample_idx][1:self.k+2]  # k+1 NN
            for knn_idx in knn[sample_idx]:
                associates[knn_idx].add(sample_idx)

        deleted_samples = set()  # set for optimal search
        mask = np.ones(len(self.S), dtype=bool)

        for i, sample_idx in enumerate(sorted_enemy_distances):
            if (i % 100) == 0:
                print('Iteration:', i)
                print('   # deleted_samples:', len(deleted_samples))

            if not associates.get(sample_idx, []):
                print(f"""Skipping sample {sample_idx}
                      due to lack of associates.""")
                continue

            # WITH
            pred_labels = [np.bincount(
                self.S_labels[knn[a][:self.k+1]]).argmax() for a in associates[sample_idx]]
            true_labels = [self.S_labels[a] for a in associates[sample_idx]]
            accuracy_with = np.sum(np.array(pred_labels)
                                   == np.array(true_labels))

            pred_labels_without = [
                np.bincount(
                    self.S_labels[[nn for nn in knn[a][:self.k+2] if nn != sample_idx]]).argmax()
                for a in associates[sample_idx]
                if associates[sample_idx]]
            true_labels_without = [self.S_labels[a]
                                   for a in associates[sample_idx]]
            accuracy_without = np.sum(
                np.array(pred_labels_without) == np.array(true_labels_without))

            if accuracy_with >= accuracy_without:
                deleted_samples.add(sample_idx)
                mask[sample_idx] = False

                for a in associates[sample_idx]:
                    nns = [s for s in nn_idxs_matrix[a]
                           [1:] if s not in deleted_samples]
                    nns = nns[:self.k+2]

                    knn[a] = nns
                    for knn_idx in nns:
                        associates[knn_idx].add(a)

        new_S = self.S[mask]
        new_S_labels = self.S_labels[mask]

        return pd.DataFrame(new_S, columns=self.columns), pd.Series(new_S_labels)

    def _minkowski_distance(self, x1, x2, r=2):
        distance = 0.0
        for i in range(len(x1)):
            distance += (abs(x1[i] - x2[i])) ** r
        return distance ** (1 / r)
