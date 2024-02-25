import sys
import random
import numpy as np


def sgn(z):
    if (z < 0.0):
        return -1
    else:
        return 1


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 main.py <data-file> <label-file>")
        exit(1)

    # read data points
    datafile = open(sys.argv[1], "r")
    Points = datafile.read()
    datafile.close()

    Points = Points.split('\n')
    Points = Points[:-1]
    # print(Points)

    # construct numeric data points
    X = []
    for index in range(len(Points)):
        point = Points[index]
        point = point.split(' ')

        # tmp = []
        # for i in range(len(point)):
        #	tmp.append(float(point[i]))
        # print(tmp)
        tmp = list(map(float, point))
        X.append(tmp)
    print(X)

    # read labels
    labelfile = open(sys.argv[2], "r")
    Labels = labelfile.read()
    labelfile.close()
    # print(Labels)

    Labels = Labels.split('\n')
    Labels = Labels[:-1]

    # construct numeric labels
    Y = []
    for index in range(len(Labels)):
        tmp = int(Labels[index])
        Y.append(tmp)
    print(Y)

    # Perceptron learning algorithm
    dim = len(X[0])
    w = np.zeros(dim)
    print("Normal = ", w)
    for index in range(len(X)):
        if Y[index] != sgn(np.dot(w, X[index])):
            w = w + np.multiply(Y[index], X[index])
            print("Normal = ", w)


if __name__ == "__main__":
    main()
