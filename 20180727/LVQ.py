import numpy as np

"""
    Minimalistic implementation of the Learning Vector Quantization (LVQ)
    based on Python Data Science Cookbook
"""

# class of prototype vectors
class prototype(object):
    """
    Define prototype, prototype is a vector with weights(p_vector) and label(class_id)
    """
    def __init__(self, class_id, p_vector, epsilon):
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

class LVQ(object):
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
    def find_closest(self, in_vector, proto_vectors):
        """
        Find the closest prototype vector for a given vector

        Parameters
        -------
        in_vector: the given vector
        proto_vectors: the set of prototype vectors
        """
        closest = None
        position = None
        closest_distance = 99999
        for i in range(len(proto_vectors)):
            distance = np.linalg.norm(in_vector - proto_vectors[i].p_vector)
            if distance < closest_distance:
                closest_distance = distance
                closest = proto_vectors[i]
                position = i
        return [position, closest]
    
    def find_runnerup(self, in_vector, proto_vectors):
        """
        Find the second closest prototype vector for a given vector

        Parameters
        -------
        in_vector: the given vector
        proto_vectors: the set of prototype vectors
        """
        closest_p_vector = self.find_closest(in_vector, proto_vectors)
        runnerup = closest_p_vector
        closest_distance = 99999
        for p_v in proto_vectors:
            distance = np.linalg.norm(in_vector - p_v.p_vector)
            if (distance < closest_distance) and (p_v != closest_p_vector):
                closest_distance = distance
                runnerup = p_v
        return runnerup
    def predict(self, test_vector):
        """
        Predict label for a given input

        Parameters
        -------
        test_vector: input vector
        """
        return self.find_closest(test_vector, self.p_vectors)[1].class_id
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
            
            closest_pvector = self.find_closest(rnd_s, self.p_vectors)[1]
            
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
            
            closest_pvector = self.find_closest(rnd_s, self.p_vectors)
            second_closest_pvector = self.find_runnerup(rnd_s, self.p_vectors)
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