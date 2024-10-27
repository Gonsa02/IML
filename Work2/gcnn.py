import numpy as np
import pandas as pd

import itertools
from scipy.spatial.distance import cdist


class GCNN:
    def __init__(self, p, dataset, labels):
        self.prototypes = []    # sample
        self.prototypes_labels = []  # labels

        self.p = p
        self.dataset = dataset
        self.labels = np.array(labels)
        self.unique_labels = np.unique(labels)

        self.unabsorbed_mask = np.ones(len(labels), dtype=bool)

        # Distances between samples
        self.distances = cdist(
            dataset, dataset, metric=self._minkowski_distance)

    def _minkowski_distance(self, x1, x2, r=1):
        distance = 0.0
        for i in range(len(x1)):
            distance += (abs(x1[i] - x2[i])) ** r
        return distance ** (1 / r)

    # G1 step

    def _initiation(self):
        for label in self.unique_labels:
            samples_idxs = np.where(self.labels == label)[0]
            prototype_idx = self._select_prototype(samples_idxs)

            # Saving prototype
            self.prototypes.append(self.dataset.iloc[prototype_idx].to_list())
            self.prototypes_labels.append(label)

            # Mark prototype as absorbed
            self.unabsorbed_mask[prototype_idx] = False

    # G2 step
    def fit(self):
        self._initiation()

        total_executions = 0
        label_unabsorbed_idxs = self._absorption_check()

        while self.unabsorbed_mask.sum() > 0:
            if (total_executions % 50) == 0:
                print('Execution:', total_executions+1)
                print('   # Prototypes:', len(self.prototypes_labels))
                print('   # Unabsorbed samples:', self.unabsorbed_mask.sum())

            unabsorbed_idxs = np.where(self.unabsorbed_mask)[0]
            if len(unabsorbed_idxs) < 1:
                break

            for label, unabsorbed_idxs in label_unabsorbed_idxs.items():
                if len(unabsorbed_idxs) == 0:
                    continue

                prototype_idx = self._select_prototype(unabsorbed_idxs)
                self.prototypes.append(
                    self.dataset.iloc[prototype_idx].to_list())
                self.prototypes_labels.append(label)

                self.unabsorbed_mask[prototype_idx] = False

            label_unabsorbed_idxs = self._absorption_check()
            total_executions += 1

        print('END')
        print('Execution:', total_executions+1)
        print('   # Prototypes:', len(self.prototypes_labels))
        print('   # Unabsorbed samples:', self.unabsorbed_mask.sum())

        new_X = pd.DataFrame(self.prototypes, columns=self.dataset.columns).astype(
            self.dataset.dtypes)
        new_y = pd.DataFrame(self.prototypes_labels)

        return new_X, new_y

    def _select_prototype(self, samples_idx):
        # Only distances from samples of the same class
        sample_distances = self.distances[samples_idx][:, samples_idx]

        nearest_neighbor_idx = np.argsort(sample_distances, axis=1)
        # getting column 1 (0 it will be with itself).
        if nearest_neighbor_idx.shape[1] > 1:
            nearest_neighbor_idx = nearest_neighbor_idx[:, 1]
        else:
            nearest_neighbor_idx = nearest_neighbor_idx[:, 0]

        indexes, counts = np.unique(nearest_neighbor_idx, return_counts=True)
        nn_submatrix_idx = indexes[np.argmax(counts)]
        return samples_idx[nn_submatrix_idx]    # General index

    def _minimum_heterogeneous_distance(self):
        unabsorbed_idxs = np.where(self.unabsorbed_mask)[0]
        unabsorbed_labels = self.labels[self.unabsorbed_mask]

        if len(np.unique(unabsorbed_labels)) > 1:
            min_dist = np.inf

            for i, j in itertools.combinations(unabsorbed_idxs, 2):
                if self.labels[i] != self.labels[j]:
                    distance = self.distances[i, j]
                    if distance < min_dist:
                        min_dist = distance

            return min_dist

        return 0

    def _absorption_check(self):
        unabsorbed_idxs = np.where(self.unabsorbed_mask)[0]
        label_unabsorbed_idx = {}  # (label, samples_idx)
        delta = self._minimum_heterogeneous_distance()

        for label in self.unique_labels:
            # unabsorved samples idx that matches the label
            unabsorbed_label_idx = unabsorbed_idxs[self.labels[unabsorbed_idxs] == label]

            if len(unabsorbed_label_idx) == 0:
                continue

            # Homogeneous prototypes
            hom_idxs = np.where(np.array(self.prototypes_labels) == label)[0]
            hom_distances = self.distances[unabsorbed_label_idx][:, hom_idxs]
            hom_diff = np.min(hom_distances, axis=1)

            # Heterogeneous prototypes
            het_idxs = np.where(np.array(self.prototypes_labels) != label)[0]
            het_distances = self.distances[unabsorbed_label_idx][:, het_idxs]
            het_diff = np.min(het_distances, axis=1)

            absorbed = (het_diff - hom_diff) > (self.p * delta)

            label_unabsorbed_idx[label] = unabsorbed_label_idx[~absorbed]
            self.unabsorbed_mask[unabsorbed_label_idx[absorbed]] = False

        return label_unabsorbed_idx
