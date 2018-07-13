import numpy as np
# class of prototype vectors
class prototype(object):
    def __init__(self, class_id, p_vector, epsilon):
        self.class_id = class_id
        self.p_vector = p_vector
        self.epsilon = epsilon
    
def update(p_vector, u_vector, epsilon, increment = True):
    if increment:
        # Move the prototype closer to input vector
        p_vector = p_vector + epsilon * (u_vector - p_vector)
    else:
        # Move the prototype away from input vector
        p_vector = p_vector - epsilon * (u_vector - p_vector)
    return p_vector

# function to find the closest prototype vector for a given vector
def find_closest(in_vector, proto_vectors):
    closest = None
    closest_distance = 99999
    for p_v in proto_vectors:
        distance = np.linalg.norm(in_vector - p_v.p_vector)
        if distance < closest_distance:
            closest_distance = distance
            closest = p_v
    return closest

# function to find a class tag for a vector sample
def find_class_id(test_vector, p_vectors):
    return find_closest(test_vector, p_vectors).class_id


# function to choose R initial prototype for each class
def initiate_prototype(x, y, n_classes, R, epsilon):
    p_vectors = []
    for i in range(n_classes):
        # select class i
        y_subset = np.where(y == i)
        # select tuple for chosen class
        x_subset = x[y_subset]
        # get R random indices between 0 and 50
        samples = np.random.randint(0, len(x_subset), R)
        # select p_vectors, they are chosen randomly from the samples x
        for sample in samples:
            s = x_subset[sample]
            p = prototype(i, s, epsilon)
            p_vectors.append(p)
    return p_vectors

# function to train LVQ 2.1

def train(x, y, epsilon, epsilon_dec_factor, p_vectors):
    while epsilon >= 0.01:
        rnd_i = np.random.randint(0, 149)
        rnd_s = x[rnd_i]
        target_y = y[rnd_i]
        
        epsilon = epsilon - epsilon_dec_factor
        
        closest_pvector = find_closest(rnd_s, p_vectors)
        
        if target_y == closest_pvector.class_id:
            closest_pvector = update(closest_pvector, rnd_s, epsilon)
        else:
            closest_pvector = update(closest_pvector, rnd_s, False, epsilon)
        closest_pvector.epsilon = epsilon
    return p_vectors
