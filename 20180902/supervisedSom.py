import numpy as np
from pylab import bone, pcolor, colorbar, plot, show


class supervisedSom(object):
    def __init__(self, som, x_train, y_train):
        """
        Initialize a supervised SOM

        Parameters
        -------
        som: trained unsupervised som
        x_train: x_train set
        y_train: y_train set

        Note that automatically initialized parameters are:
        labels: a 3-dimensional np array denoting the number of each label's samples
        propa: a 3-dimensional np array similar to labels but in proportion (%)
        taggings: a 2-dimensional np array with chosen labels resulted from labels and propa
        """
        self.som = som
        self.x_train = x_train
        self.y_train = y_train
        self.labels = np.zeros((len(np.unique(y_train)), som.x, som.y))
        self.propa = np.zeros((self.labels.shape[0], self.labels.shape[1], self.labels.shape[2]))
        self.taggings = np.zeros((som.x, som.y))

    def visualSom(self):
        """
        Visualizing a SOM with marked labels of multiple classes

        Parameters
        -------
        """
        numLabels = len(np.unique(self.y_train))
        bone()
        pcolor(self.som.distance_map().T)
        colorbar()
        markers = ['v', 's', 'o', '4']
        colors = ['r', 'g', 'b', 'y']
        for i, x in enumerate(self.x_train):
            w = self.som.winner(x)
            plot(w[0] + 0.5,
                w[1] + 0.5,
                markers[self.y_train[i]],
                markeredgecolor = colors[self.y_train[i]],
                markerfacecolor = 'None',
                markersize = 10,
                markeredgewidth = 2)
            for nl in range(numLabels):
                if self.y_train[i] == nl:
                    self.labels[nl, w[0], w[1]] += 1
        show()
        return self.labels

    def propabilitySom(self):
        """
        Calculate propa np array

        Parameters
        -------
        """
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
    
    def taggingSom(self):
        """
        Calculate tagging np array

        Parameters
        -------
        """
        x = self.labels.shape[1]
        y = self.labels.shape[2]
        tag = self.labels.shape[0]
        for i in range(x):
            for j in range(y):
                tmp = np.array([self.labels[t][i][j] for t in range(tag)])
                sort = tmp.argsort()
                if (tmp[sort[tag-1]] <= tmp[sort[tag-2]] * 5): # must be improve
                    self.taggings[i][j] = np.random.choice([sort[tag-1], sort[tag-2]])
                else:
                    self.taggings[i][j] = sort[tag-1]
        return self.taggings

    def find_closest(self, in_vector):
        """
        Find the closest neuron in som for a given vector

        Parameters
        -------
        in_vector: the given vector
        """
        closest_distance = 99999
        x = self.labels.shape[1]
        y = self.labels.shape[2]
        for i in range(x):
            for j in range(y):
                if (sum(self.labels[k, i, j] for k in range(self.labels.shape[0])) == 0):
                    continue
                else:
                    distance = np.linalg.norm(in_vector - self.som.weights[i][j])
                    if distance < closest_distance:
                        closest_distance = distance
                        closest = (i, j)         
        return closest

    def predict(self, test_vector):
        """
        Find the label

        Parameters
        -------
        test_vector: the given vector
        """
        position = self.find_closest(test_vector)
        return self.taggings[position[0], position[1]]
