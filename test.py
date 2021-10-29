#!/usr/bin/python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import pandas as pd
import unittest 
from k_means_from_scratch import KMeansFromScratch

class TestKMeansFromScratchMethods(unittest.TestCase):
    """
    Unit test in order to ensure that our previous class methods run always as defined,
    implemented, and correctly.

    Attributes:
        model(KMeansFromScratch): The K-means implementation from scratch
        data(np.array)
    """
    model = KMeansFromScratch(n_clusters=3, n_iterations=3, random_state=47)
    data = pd.read_csv('./data/abalone.csv')[['LongestShell', 'Diameter']].values

    def test_euclidean_distance(self):
        """
        """
        X = [2, 2]
        Y = [3, 2]
        self.assertEqual(self.model.euclidean_distance(X, Y), 1.0)

    def test_get_centroids(self):
        """
        """
        centroids = self.model.get_centroids(self.data)
        self.assertEqual(len(centroids), self.model.n_clusters)

    def test_get_min_index(self):
        """
        """
        self.assertEqual(self.model.get_min_index(self.data[-1]), 1)

    def test_clustering(self):
        """
        """
        centroids = self.model.get_centroids(self.data)
        clusters = self.model.clustering(self.data, centroids)

        self.assertEqual(len(clusters), self.model.n_clusters)

    def test_get_centroids_mean(self):
        """
        """
        clusters = self.model.clustering(self.data, self.model.get_centroids(self.data))
        centroids = self.model.get_centroids_mean(clusters)

        self.assertEqual(len(centroids), self.model.n_clusters)

    def test_fit(self):
        """
        """
        clusters = self.model.fit(self.data)
        self.assertEqual(len(clusters), 3)


if __name__ == '__main__':
    """
    Run
    """
    unittest.main()
