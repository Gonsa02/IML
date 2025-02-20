import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


class ENNTh:

    def __init__(self, dataset, labels, k=7, threshold=0.8):
        self.dataset = dataset.to_numpy()
        self.labels = labels.to_numpy()
        self.unique_labels = np.unique(labels)
        self.k = k
        self.thresh = threshold
        self.columns = dataset.columns

        self.distances = squareform(pdist(
            dataset, metric=self._minkowski_distance))

        self.probabilities = []  # (label, prob)
        self._compute_probabilities()

    def fit(self):
        idxs_to_remove = []
        for sample_idx, (label, prob) in enumerate(self.probabilities):
            if (sample_idx % 100) == 0:
                print('Sample:', sample_idx+1)
                print('   # idxs_to_remove:', len(idxs_to_remove))

            if label != self.labels[sample_idx] or prob <= self.thresh:
                idxs_to_remove.append(sample_idx)

        new_dataset = np.delete(self.dataset, idxs_to_remove, axis=0)
        new_labels = np.delete(self.labels, idxs_to_remove, axis=0)

        return pd.DataFrame(new_dataset, columns=self.columns).reset_index(drop=True), pd.Series(new_labels).reset_index(drop=True)

    def _compute_probabilities(self):
        # Order distances
        nn_idxs_matrix = np.argsort(self.distances, axis=1)

        for sample_idx, nn_idxs in enumerate(nn_idxs_matrix):
            # Remove itself and choose only k nn.
            nn_idxs = nn_idxs[1:self.k+1]

            nn_distances = self.distances[sample_idx][nn_idxs]
            nn_labels = self.labels[nn_idxs]

            prob_x = []  # Pi(x)
            for label in self.unique_labels:
                prob_x.append(np.sum((nn_labels == label)/(1+nn_distances)))

            # Normalize probs: ensure values up to 1
            normalized_prob_x = np.array(prob_x) / np.sum(prob_x)

            max_prob_idx = np.argmax(normalized_prob_x)
            max_prob = normalized_prob_x[max_prob_idx]

            self.probabilities.append(
                (self.unique_labels[max_prob_idx], max_prob))

    def _minkowski_distance(self, x1, x2, r=2):
        distance = 0.0
        for i in range(len(x1)):
            try:
                diff = x1[i] - x2[i]
                distance += abs(diff) ** r
            except TypeError as te:
                print(f"""TypeError at index {
                      i}: cannot subtract x2[{i}] from x1[{i}].""")
                print(f"""  x1[{i}] type: {type(x1[i])
                                           }, x2[{i}] type: {type(x2[i])}""")
                print(f"  x1[{i}] = {x1[i]}, x2[{i}] = {x2[i]}")
                raise te

        return distance ** (1 / r)
