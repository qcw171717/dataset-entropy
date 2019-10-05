import math
import random
import numpy as np
from statistics import stdev
from scipy.stats import norm
from scipy.spatial import distance
from sklearn.neighbors import KDTree

def calculate_dataset_entropy(dataset_x, labels):
    '''
        Calculate the dataset entropy with the features given
    '''
    # Randomly initialize 1/10 of points within range of dataset features
    rand_points = []
    for _ in range(math.ceil(dataset_x.shape[0]/10)):
        rand_points.append(rand_init_point(dataset_x))
    
    entropies = []
    for point in rand_points:
        this_point_prob = calculate_point_probability(dataset_x, point)
        this_point_entropy = calculate_point_entropy(dataset_x, labels, point)
        entropies.append(this_point_prob * this_point_entropy)
    
    return sum(entropies)/len(entropies)

def calculate_point_probability(dataset_x, point):
    '''
        Returns the probability of a point (possibily randomly generated) occuring in dataset_x
    '''
    assert dataset_x.shape[1] == point.size
    # sample len(dataset_x) number of pairs of points in the dataset to compare cosin distances
    # an implementation of CLT that generates a likelyhood score
    # define random variable X=cosine distance between 2 points
    # first estimate population mean and standard deviation
    sample_size = dataset_x.shape[0]
    summed_distances = []
    for _ in range(sample_size):
        random_idx1 = random.randint(0, dataset_x.shape[0]-1)
        random_idx2 = random.randint(0, dataset_x.shape[0]-1)
        while random_idx1 == random_idx2:
            random_idx1 = random.randint(0, dataset_x.shape[0]-1)
            random_idx2 = random.randint(0, dataset_x.shape[0]-1)
        summed_distances.append(distance.cosine(dataset_x[random_idx1], dataset_x[random_idx2]))
    mean_distance = sum(summed_distances)/sample_size
    stdev_distance = stdev(summed_distances, mean_distance)
    if stdev_distance == 0:
        return 0 if distance.cosine(dataset_x[0], point) != mean_distance else 1
    
    # then find sample mean and standard deviation in the normal distribution
    n = max(math.ceil(dataset_x.shape[0]/10), 100)
    sample_mean = mean_distance
    sample_stdev = stdev_distance / math.sqrt(n)
    sampled_distances = []
    for _ in range(n):
        random_idx = random.randint(0, dataset_x.shape[0]-1)
        sampled_distances.append(distance.cosine(dataset_x[random_idx], point))
    point_mean_distance = sum(sampled_distances)/n
    
    point_probability = norm(sample_mean, sample_stdev).pdf(point_mean_distance)
    return point_probability

def rand_init_point(dataset_x):
    ''' 
        Returns a randomly initiallized point satisfying:
        1. dataset_x.shape[1] == len(point)
        2. point[i] <= dataset_x[:, some_idx].max()
        3. point[i] >= dataset_x[:, some_idx].min()
    '''
    point = []
    for i in range(0, dataset_x.shape[1]):
        feature_min = dataset_x[:, i].min()
        feature_max = dataset_x[:, i].max()
        point.append(random.uniform(feature_min, feature_max))
    return np.array(point)

def calculate_point_entropy(dataset_x, labels, point, num_n=10):
    '''
        Returns the entropy of this point
        Entropy of a point is defined to be the impurity of the num_n number of neighbors of this point
    '''
    kdt = KDTree(dataset_x, leaf_size=30, metric='euclidean')
    indices = kdt.query(point, k=num_n, return_distance=False)

    # In the binary case, take average of the labels, worse if it's closer to 0.5
    nearest_labels = [labels[i] for i in indices]
    label_average = sum(nearest_labels)/len(nearest_labels)
    point_entropy = 1/abs(label_average-0.5)

    return point_entropy
