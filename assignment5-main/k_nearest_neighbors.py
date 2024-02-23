# Submitted by: |Joseph Shepherd, joseshep| |Venkata Naga Sreya Kolachalama, vekola|       

import numpy as np
from utils import euclidean_distance, manhattan_distance

class KNearestNeighbors:
    def __init__(self, n_neighbors=5, weights='uniform', metric='l2'):
        if weights not in ['uniform', 'distance'] or metric not in ['l1', 'l2'] or not isinstance(n_neighbors, int):
            raise ValueError('The provided class parameter arguments are not recognized.')

        self.n_neighbors = n_neighbors
        self.weights = weights
        self._X = None
        self._y = None
        self._distance = euclidean_distance if metric == 'l2' else manhattan_distance

    def fit(self, X, y):
        """
        Fits the model to the provided data matrix X and targets y.
        Args:
            X: A numpy array of shape (n_samples, n_features) representing the input data.
            y: A numpy array of shape (n_samples,) representing the true class values for each sample in the input data.
        Returns:
            None.
        """
        self._X = X
        self._y = y

    def predict(self, X):
        """
        Predicts class target values for the given test data matrix X using the fitted classifier model.
        Args:
            X: A numpy array of shape (n_samples, n_features) representing the test data.
        Returns:
            A numpy array of shape (n_samples,) representing the predicted target class values for the given test data.
        """
        epsilon = 1e-9
        predictions = []

        for sample in X:
            #Calculate distances between the sample and training samples
            distances = [self._distance(sample, x) for x in self._X]

            #Get the indices of the kNN
            sorted_indices = np.argsort(distances)
            k_nearest_indices = sorted_indices[:self.n_neighbors]

            if self.weights == 'uniform':
                #Simple majority method
                neighbors_labels = self._y[k_nearest_indices]
                predicted_label = np.bincount(neighbors_labels).argmax()
            elif self.weights == 'distance':
                #Weighting based on distance
                weights = 1 / ((np.array(distances)[k_nearest_indices]) + epsilon)
                weighted_votes = np.bincount(self._y[k_nearest_indices], weights)
                predicted_label = weighted_votes.argmax()

            predictions.append(predicted_label)

        return np.array(predictions)
