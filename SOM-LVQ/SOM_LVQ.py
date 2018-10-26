import numpy as np
from collections import defaultdict
"""
    Version 0: Minimalistic implementation of the Learning Vector Quantization (LVQ)
    based on Python Data Science Cookbook

    Version 1: LVQ based on SOM with SOM topology
"""

# class of prototype vectors
class prototype(object):
    """
    Define prototype, prototype is a vector with weights(p_vector) and label(class_id)
    """
    def __init__(self, class_id, p_vector, epsilon=0.9):
        self.class_id = class_id
        self.p_vector = p_vector
        self.epsilon = epsilon
    def update(self, u_vector, increment = True):
        """
        The function to update the prototype vector of the closest point

        If the class label of the prototype vector is the same as the input data point, we will
        increment the prototype vector with the difference between the prototype vector and data
        point.
        If the class label is different, we will decrement the prototype vector with the difference
        between the prototype vector and data point.
        """
        if increment:
            # Move the prototype closer to input vector
            self.p_vector = self.p_vector + self.epsilon * (u_vector - self.p_vector)
        else:
            # Move the prototype away from input vector
            self.p_vector = self.p_vector - self.epsilon * (u_vector - self.p_vector)

class SOM_LVQ(object):
    def __init__(self, x, y, n_classes, n_neurons, p_vectors, epsilon=0.9, epsilon_dec_factor=0.001):
        """
        Initialize a LVQ network.

        Parameters
        -------
        x, y : the data and label
        n_classes: the # of distinctive classes
        n_neurons: the # of prototype vectors for each class
        epsilon: learning rate
        epsilon_dec_factor: decrease factor for learning rate

        p_vectors: the set of prototype vectors
        """
        self.x = x
        self.y = y
        self.n_classes = n_classes
        self.n_neurons = n_neurons
        self.epsilon = epsilon
        self.epsilon_dec_factor = epsilon_dec_factor
        self.p_vectors = p_vectors
        if(len(self.p_vectors) == 0):
            p_vectors = []
            for i in range(n_classes):
                # select class i
                y_subset = np.where(y == i)
                # select tuple for chosen class
                x_subset = x[y_subset]
                # get R random indices between 0 and len(x_subset)
                samples = np.random.randint(0, len(x_subset), n_neurons)
                # select p_vectors, they are chosen randomly from the samples x
                for sample in samples:
                    s = x_subset[sample]
                    p = prototype(i, s, epsilon)
                    p_vectors.append(p)
        self.p_vectors = p_vectors
        self.labels = np.zeros((len(np.unique(y)), self.p_vectors.shape[0], self.p_vectors.shape[1]))
        self.propa = np.zeros((len(np.unique(y)), self.p_vectors.shape[0], self.p_vectors.shape[1]))
    def find_closest(self, in_vector):
        """
        Find the closest prototype vector for a given vector

        Parameters
        -------
        in_vector: the given vector
        proto_vectors: the set of prototype vectors
        """
        proto_vectors = self.p_vectors
        closest = None
        position = ()
        closest_distance = 9999999
        for i in range(proto_vectors.shape[0]):
            for j in range(proto_vectors.shape[1]):
                distance = np.linalg.norm(in_vector - proto_vectors[i][j].p_vector)
                if distance < closest_distance:
                    closest_distance = distance
                    closest = proto_vectors[i][j]
                    position = (i,j)
        return [position, closest]
    
    def find_runnerup(self, in_vector):
        """
        Find the second closest prototype vector for a given vector

        Parameters
        -------
        in_vector: the given vector
        proto_vectors: the set of prototype vectors
        """
        proto_vectors = self.p_vectors
        closest_p_vector = self.find_closest(in_vector)
        runnerup = closest_p_vector
        closest_distance = 99999
        for i in range(proto_vectors.shape[0]):
            for j in range(proto_vectors.shape[1]):
                distance = np.linalg.norm(in_vector - proto_vectors[i][j].p_vector)
                if (distance < closest_distance) and (proto_vectors[i][j] != closest_p_vector):
                    closest_distance = distance
                    runnerup = proto_vectors[i][j]
        return runnerup
    def predict(self, test_vector):
        """
        Predict label for a given input

        Parameters
        -------
        test_vector: input vector
        """
        return self.find_closest(test_vector)[1].class_id
    def fit(self, x, y):
        """
        Perform iteration to adjust the prototype vector 
        in order to classify any new incoming points using existing data points

        Parameters
        -------
        x: input
        y: label
        """
        while self.epsilon >= 0.01:
            rnd_i = np.random.randint(0, len(x))
            rnd_s = x[rnd_i]
            target_y = y[rnd_i]
            
            self.epsilon = self.epsilon - self.epsilon_dec_factor
            
            closest_pvector = self.find_closest(rnd_s)[1]
            
            if target_y == closest_pvector.class_id:
                closest_pvector.update(rnd_s)
            else:
                closest_pvector.update(rnd_s, False)
            closest_pvector.epsilon = self.epsilon
        return self.p_vectors

    def train_LVQ2(self, x, y):
        """
        First improvement for LVQ, update both the winner and the runner up vector

        Parameters
        -------
        x: input
        y: label
        """
        while self.epsilon >= 0.01:
            rnd_i = np.random.randint(0, len(x))
            rnd_s = x[rnd_i]
            target_y = y[rnd_i]
            
            self.epsilon = self.epsilon - self.epsilon_dec_factor
            
            closest_pvector = self.find_closest(rnd_s)[1]
            second_closest_pvector = self.find_runnerup(rnd_s)
            compare_distance = np.linalg.norm(closest_pvector.p_vector - rnd_s)/np.linalg.norm(second_closest_pvector.p_vector - rnd_s)
            
            if target_y == second_closest_pvector.class_id and target_y != closest_pvector.class_id and compare_distance > 0.8 and compare_distance < 1.2:
                closest_pvector.update(rnd_s, False)
                second_closest_pvector.update(rnd_s)
            elif target_y == closest_pvector.class_id:
                closest_pvector.update(rnd_s)
            elif target_y != closest_pvector.class_id:
                closest_pvector.update(rnd_s, False)
            closest_pvector.epsilon = self.epsilon
        return self.p_vectors

    def train_LVQ_neighbors(self, x, y):
        while self.epsilon >= 0.01:
            rnd_i = np.random.randint(0, len(x))
            rnd_s = x[rnd_i]
            target_y = y[rnd_i]
            
            self.epsilon = self.epsilon - self.epsilon_dec_factor
            
            index = self.find_closest(rnd_s)[0]
            # if (index == ()): 
            #     continue
            selfx = self.p_vectors.shape[0]
            selfy = self.p_vectors.shape[1]
            update_p_vectors = []
            x_down = -1 if (index[0] > 0) else 0
            x_up = 1 if (index[0] < selfx-1) else 0
            y_down = -1 if (index[1] > 0) else 0
            y_up = 1 if (index[1] < selfy-1) else 0    
            for dx in range(x_down, x_up + 1):
                for dy in range(y_down, y_up + 1):
                    update_p_vectors.append(self.p_vectors[index[0] + dx][index[1] + dy])
                
            for p in update_p_vectors:
                if target_y == p.class_id:
                    p.update(rnd_s)
                else:
                    p.update(rnd_s, False)
                p.epsilon = self.epsilon
        return self.p_vectors
    def win_map_LVQ(self, x):
        """
            Returns a dictionary wm where wm[(i,j)] is a list with all the patterns
            that have been mapped in the position i,j.
        """
        win_map = defaultdict(list)
        for ix in x:
            win_map[self.find_closest(ix)[0]].append(ix)
        return win_map

    def labelingLVQ(self):
        """
        Count the number of samples of each label for each neuron
        """
        numLabels = len(np.unique(self.y))
        for i, x in enumerate(self.x):
            w = self.find_closest(x)[0]
            for nl in range(numLabels):
                if self.y[i] == nl:
                    self.labels[nl, w[0], w[1]] += 1
        return self.labels

    def propabilityLVQ(self):
        """
        Calculate propa np array

        Parameters
        -------
        """
        self.labels = self.labelingLVQ()
        for i in range(self.labels.shape[0]):
            for j in range(self.labels.shape[1]):
                for k in range(self.labels.shape[2]):
                    total = sum(self.labels[i, j, k] for i in range(self.labels.shape[0]))
                    if total == 0. :
                        continue
                    else:
                        self.propa[i, j, k] = self.labels[i, j, k] / total
                        self.propa[i, j, k] = round(self.propa[i, j, k], 2)
        return self.propa
def init_LVQ_pvectors(som, taggings, x_train, y_train):
    """
        Returns a 2D array of p_vectors for SOM_LVQ learning algorithms
        The weights and labels are initialized based on the weights and labels of trained SOM
    """
    p_vectors = np.ndarray(shape = (som.x, som.y), dtype = prototype)
    for i in range(som.x):
        for j in range(som.y):
            p_vectors[i][j] = prototype(taggings[i][j], som.weights[(i,j)])
    return p_vectors


