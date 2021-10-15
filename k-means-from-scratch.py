#/usr/bin/python3

import math
import numpy as np
import pandas as pd

## Compute euclidean distance between X and Y

def euclidean_distance(X, Y):
    """
    Compute euclidean distance between X and Y
    
    Arg(s):
        X(array):
        Y(array):
    Return(s):
        distance(float): The distance between X and Y
    """
    try:
        distance = 0
        if len(X) == len(Y):
            for (x, y) in zip(X, Y):
                distance +=(y - x)**2

            return np.round(math.sqrt(distance), 3)
        else:
            print("The xs vectors do not have the same length... !")
            
    except Exception as e:
        print('The vectors must be arry or list ... !')


## Get randomly the controid values

def get_centroids(dataset, n_cluster=3, random_state=47):
    """
    Get randomly the controid values
    
    Arg(s):
        dataset(numpy.array):
        n_cluster(int):
        random_state(int):
    Return(s):
        centroids(nump.array):
    """
    np.random.seed(random_state)
    centroids = []
    rows = dataset.shape[0]
    
    for _ in range(n_cluster):
        picked_centroid = np.random.randint(0, rows)
        
        centroids.append(dataset[picked_centroid])
    return np.array(centroids)

def get_min_index(array):
    """
    Given an array of at least two values, return the index of the minimim valu that it contains
    
    Arg(s):
        array(array):
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
        return ("Warning ! The array must contain at least two values ... !")

if __name__ == '__main__':
    X = [2, 3]
    Y = [4, 5, 2, 1.5]

    df = pd.read_csv("/home/alain/Documents/MSD/Projects/Kmeans/sklearn/abalone.csv")
    header = ['LongestShell', 'Diameter', 'Height', 'WholeWeight', 'ShellWeight', 'Rings']
    df = df[header]
    values = df.values
    print(get_centroids(values))
    print(get_min_index(Y))

	