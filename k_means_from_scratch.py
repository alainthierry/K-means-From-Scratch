#!/usr/bin/python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import pandas as pd

class KMeansFromScratch(object):
    """
    Implementation of K-means, a clustering machine learning algorithm in the case of
    Unsupervised learning from scratch !
    
    Attributes:
        n_clusters(integer): The number of cluster chosen
        n_iterations(integer): The number of iterations to run the algorithm
        random_state(integer)
        centroids_(list): This is a class attribute that contains the centroid values after
        clusters_(list): List of clustered data
        training(fit)
    """
    centroids_ = []
    
    def __init__(self, n_clusters, n_iterations, random_state):
        self.n_clusters = n_clusters
        self.n_iterations = n_iterations
        self.random_state = random_state
        
    def euclidean_distance(self, X, Y):
        """
        Compute euclidean distance between X and Y

        Arg(s):
            X(array): List of coordinates
            Y(array): List of coordinates
        Return(s):
            distance(float): The distance between X and Y
        """
        try:
            distance = np.linalg.norm(Y - X)
            return distance
        
        except Exception as e:
            print(f"For distance computing, I got this{e}")
            
    def get_centroids(self, dataset):
        """
        Getting randomly centroids values

        Arg(s):
            dataset(numpy.array): The whole observations to cluster
        Return(s):
            centroids(nump.array): The randomly picked centroids
        """
        np.random.seed(self.random_state)
        centroids = []
        rows = dataset.shape[0]

        for _ in range(self.n_clusters):
            centroid_index = np.random.randint(0, rows)

            centroids.append(dataset[centroid_index])

        return np.array(centroids)
    
    def get_min_index(slef, array):
        """
        Given an array of at least two values, return the index of the minimim valu that
        it contains

        Arg(s):
            array(array): The list of values where to get the index of the minimum value
        Return(s):
            index(integer): The index of the minimum value that is in the array
        """

        if len(array) >= 2:
            min_value = array[0]
            index = 0

            for i in range(len(array)):
                if min_value > array[i]:
                    min_value = array[i]
                    index = i

            return index
        else:
            return("Warning ! The array must contain at least two values ... !")
    
    def get_centroids_mean(self, data):
        """
        Getting the centroid as the mean of each previous cluster as the new centroid

        Arg(s):
            data(np.array): This is especially the return of clustering function
        Return(s):
            centroids(np.array): The centroids as means of the previously clustered
        """

        centroids = []

        for k_cluster in range(self.n_clusters):
            centroids.append(data[k_cluster].mean(axis=0))

        return np.array(centroids)

        
    def clustering(self, data, centroids):
        """
        Clustering data points using the euclidean distance between the observs and the
        centroid points

        Arg(s):
            data(np.array): The observations to cluster
            centroids(nump.array): The randomly picked centroids
        Return(s):
            data_per_cluster(list): The clusters of the whole observations
            cluster_indexes(list): The clusters' indexes
        """

        temp = {}
        for k_cluster in range(self.n_clusters):
            liste = []

            for observation in data:
                liste.append(self.euclidean_distance(observation, centroids[k_cluster]))

            temp[f'k_{k_cluster}'] = liste

        """
        In this data set below(distances), every column represents a k cluster.
        The row represents. The distance between one observation and the whole k clusters.
        """
        distances = pd.DataFrame(data = temp).values

        """
        Clustering observation, data
        """
        cluster_indexes = []
        for row in distances:
            cluster_indexes.append(self.get_min_index(row))

        data_per_cluster = []
        for k_cluster in range(self.n_clusters):
            classified_observs = []

            for index in range(len(cluster_indexes)):
                if k_cluster == cluster_indexes[index]:
                    classified_observs.append(data[index])

            data_per_cluster.append(np.array(classified_observs))

        return data_per_cluster, cluster_indexes
    
    def fit(self, data):
        """
        Run K-means clustering n_iterations times

        Arg(s):
            data(np.array): The data to cluster
        Return(s):
            (list): The clusters after fitting
            cluster_indexes(list): The clusters' indexes
        """
        try:
            centroids = self.get_centroids(data)
            clusters, cluster_indexes = self.clustering(data, centroids)
            
            if self.n_iterations <= 0:
                print("The number of iterations must be at least 3 ... !")

            elif self.n_iterations == 1:
                KMeansFromScratch.centroids_ = centroids
                KMeansFromScratch.clusters_ = clusters
                
                return clusters, cluster_indexes
            else:
                for _ in range(self.n_iterations):
                    centroids = self.get_centroids_mean(clusters)
                    clusters, cluster_indexes = self.clustering(data, centroids) 
                    
                KMeansFromScratch.centroids_ = centroids
                KMeansFromScratch.clusters_ = clusters
                
                return clusters, cluster_indexes

        except Exception as e:
            print(f"""This {e} has been returned ! The variable data must have the wrong
            data structure ... !\n Please check the fit function args type by running help(fit)
            ... !""")
            
    def inertia(self):
        """
        This computes the inertia value, the lower is the inertia, the better the model is.
        Sum squares of the difference of each data point and its closest centroid.
        
        Return(s):
            inertia_value(float): The inertia value
        """
        centroids = KMeansFromScratch.centroids_
        clusters = KMeansFromScratch.clusters_
        inertia_value = 0
        
        for index in range(self.n_clusters):
            
            for cluster_row in clusters[index]:
                inertia_value +=np.linalg.norm(cluster_row - centroids[index])**2
                
        return inertia_value
    
    
    def predict(self, new_entry):
        """
        Predicting a new data point after training the model
        
        Arg(s):
            new_entry(np.array): The new data point to predict using the built model
        Return(s):
            clusters(list): List of clusters
        """
        try:
            clusters = []
            centroids = KMeansFromScratch.centroids_
            
            if len(centroids) != 0:
                clusters = self.clustering(new_entry, centroids)[1]
                
                return clusters
            else:
                print(""" Oops ! I did it again...!
                Please, fit your model by providing the data to the fit method before predicting... !""")
        except Exception as e:
            print(f"This {e} occurs !")


if __name__ == '__main__':

    X = np.array([2, 2])
    Y = np.array([3, 2])

    data = pd.read_csv('./data/data.csv').values

    model = KMeansFromScratch(n_clusters=4, n_iterations=500, random_state=47)
    clusters = model.fit(data)

    print(f"Centroids \n{model.centroids_}, centroids length \n{len(clusters)}")
    print(f"Clusters\n{clusters[1]}")
    print(f"The inertia is {model.inertia()}")

	