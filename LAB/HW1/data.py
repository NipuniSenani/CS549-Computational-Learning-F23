import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt

# todo:  1. importing data as numpy arrays.
x_data = np.loadtxt(fname="train-a1-449.txt",
                    dtype=float,
                    delimiter=" ",
                    usecols=np.arange(0, 1024))
# print(x_data[0, :])

y_data = np.loadtxt(fname="train-a1-449.txt",
                    dtype=str,
                    delimiter=" ",
                    usecols=1024)
# print(y_data[0])

# convert output data to 1, -1
# i.e: 'Y' = 1 and 'N' = -1
y_data = np.where(y_data == 'Y', 1, -1)

# todo: 2. normalize and separate data to training set and test set.
x_data = preprocessing.normalize(x_data)

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=9)


# todo: 3. write a function to perceptron algorithm.

def sgn(z):
    return np.where(z < 0.0, -1, +1)


def perceptron(X, Y):
    epochs = 0
    w = np.zeros((X.shape[1]))
    b = 0
    converged = False
    train_acc = {}

    while not converged and (epochs < 100):

        converged = True

        for index in range(X.shape[0]):
            if Y[index] != sgn(np.dot(w, X[index]) + b):
                w = w + np.multiply(Y[index], X[index])
                b += Y[index]
                converged = False

        epochs += 1

        # train accuracy
        train_acc[epochs] = np.sum([Y == sgn(np.dot(X, w) + b)])/len(Y)

    print(f"train set converged = {converged} for epochs={epochs} with accuracy={train_acc[epochs]}")
    plt.figure()
    plt.plot(train_acc.values())
    plt.show()
    return w, b


# todo: 4: train the perceptron using training set.
trained_w, trained_b = perceptron(X_train, y_train)

# todo: 5: apply it to test set.
# do broadcasting
y_predicted = sgn(np.dot(X_test, trained_w) + trained_b)
result = [y_test == y_predicted]

accuracy = np.sum(result) / len(y_test) * 100
print(f"accuracy of test set = {accuracy}")

"""
If y and yi have the same sign (both +1 or both -1),
then the product xiyi will be positive, and the weight vector w will be adjusted in the direction of the input vector x. 
This helps to move the decision boundary closer to the correct classification.

If y and yi have opposite signs (one is +1, and the other is -1), 
then the product xiyi will be negative, and the weight vector w will be adjusted away from the input vector x.
This is done to reduce the misclassification error
"""
