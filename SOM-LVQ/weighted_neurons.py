import numpy as np
from random import randrange
import sys
sys.path.insert(0, '../SOM-LVQ')
import LVQ

# Create a random subsample from the dataset with replacement
def subsample(x, y, ratio):
    sample_x = list()
    sample_y = list()
    n_sample = round(len(x) * ratio)
    while len(sample_x) < n_sample:
        index = randrange(len(x))
        sample_x.append(x[index])
        sample_y.append(y[index])
    return [np.asarray(sample_x), np.asarray(sample_y)]

# Bootstrap Aggregation Algorithm
def bagging(x_train, y_train, x_test, sample_size, n_clfs):
    clfs = list()
    for i in range(n_clfs):
        sample = subsample(x_train, y_train, sample_size)
        lvq = LVQ.LVQ(sample[0], sample[1], n_classes=4, p_vectors=[], n_neurons=25)
        y_pred = [lvq.predict(instance) for instance in x_test]
        clfs.append(y_pred)
    return(np.asarray(clfs))

# Make a prediction with a list of bagged clfs
def bagging_predict(predictions, n_row, n_clfs):
    return [np.bincount([predictions[j][i] for j in range(n_clfs)]).argmax() for i in range(n_row)]

# This function used for SOM and SOM_LVQ because their sets of neurons have topology structure, 
# the propa array is 3D array
def weighted_neurons_predict(models, mappings, propa, x):
    n_models = len(models)
    pos = [] # array to store postion of BMUs
    l = [] # array to store the number of samples in each neurons
    for i in range(n_models):
        tmp_pos = models[i].find_closest(x)[0] # find_closest function return [position, closest]
        pos.append(tmp_pos)
        tmp_l = len(mappings[i][tmp_pos])
        l.append(tmp_l)
    soft_predict = []
    weights = [i/np.sum(l) for i in l] # li / (l1 + l2 + ...)
    for i in range(propa[0].shape[0]):
        tmp_propa = 0
        for j in range(n_models):
            tmp_propa += weights[j] * propa[j][i][pos[j][0]][pos[j][1]]
        soft_predict.append(tmp_propa)
    return np.argmax(np.array(soft_predict))
    
# This function used for LVQ, the propa array is 2D array
def weighted_neurons_predict_LVQ(models, mappings, propa, x):
    n_models = len(models)
    pos = [] # array to store postion of BMUs
    l = [] # array to store the number of samples in each neurons
    for i in range(n_models):
        tmp_pos = models[i].find_closest(x)[0] # find_closest function return [position, closest]
        pos.append(tmp_pos)
        tmp_l = len(mappings[i][tmp_pos])
        l.append(tmp_l)
    soft_predict = []
    weights = [i/np.sum(l) for i in l] # li / (l1 + l2 + ...)
    for i in range(propa[0].shape[0]):
        tmp_propa = 0
        for j in range(n_models):
            tmp_propa += weights[j] * propa[j][i][pos[j]]
        soft_predict.append(tmp_propa)
    return np.argmax(np.array(soft_predict))
    