import numpy as np
from matplotlib import pyplot as plt


class Perceptron:
    def __init__(self):
        self.weights = None
        self.converged = None
        self.train_acc = {}
        self.epochs = 0
        self.bias = None
        self.margin = None
        self.norm_weights = None

    def sgn(self, z):
        return np.where(z < 0.0, -1, +1)

    def algorithm(self, X, Y, max_epochs=100):
        """
        Perceptron algorithm given in the text book page 190
        Inputs:
        X - normalized data matrix wit size (#data points, # features)
        Y - True output {+1, -1}
        max_epochs - maximum number of epochs


        This function update the
        weights of the perceptron
        margin parameter which measures how far the correctly classified points
        are from the decision boundary.
        """
        w = np.zeros((X.shape[1]+1))
        n = X.shape[0]

        # while self.epochs != max_epochs and not self.converged:
        while not self.converged:
            self.converged = True

            for index in range(n):
                if Y[index] != self.sgn(np.dot(w[:-1], X[index, :])+w[-1]):
                    w += np.append(np.multiply(Y[index], X[index]), Y[index])
                    self.converged = False

            self.epochs += 1

            # train accuracy
            self.train_acc[self.epochs] = np.sum([Y == self.sgn(np.dot(X, w[:-1]) + w[-1])]) / len(Y)

        self.weights = w
        w = w / np.linalg.norm(w)
        self.norm_weights=w
        self.margin = np.absolute(np.min(np.dot(X, w[:-1]) + w[-1]))

    def train_accuracy(self):
        print(f"train set converged = {self.converged} for epochs={self.epochs} "
              f"with accuracy={self.train_acc[self.epochs] * 100}%")
        plt.figure()
        plt.plot(self.train_acc.values())
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.title('Accuracy of training set at each epoch')
        plt.show()

    def test_accuracy(self, X, Y):
        result = [self.sgn(np.dot(X, self.weights[:-1] ) + self.weights[-1]) == Y]
        accuracy = np.sum(result) / len(Y) * 100
        print(f"accuracy of test set = {accuracy}%")
