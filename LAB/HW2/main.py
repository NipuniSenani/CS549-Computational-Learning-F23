import numpy as np

from nn import NN

# Todo 1. load data-
X_test = np.loadtxt("a2-test-data.txt",
                    delimiter=" ")
# Scale input data row wise
X_test = X_test / np.linalg.norm(X_test, keepdims=True, axis=1)

with open("a2-test-label.txt", 'r') as test_label_file:
    y_test = np.array(eval(test_label_file.readline()))

X_train = np.loadtxt("a2-train-data.txt",
                     delimiter=" ",
                     usecols=range(1000))
# Scale input data row wise
X_train = X_train / np.linalg.norm(X_train, keepdims=True, axis=1)

y_train = np.loadtxt("a2-train-label.txt")

nn = NN()


#  Todo 7. Create NN model-
def nn_model(X, Y, n_h, num_iterations, print_cost=False):
    """
    n - dimension of the data set
    m - numer of data points in the training set

    structure of the Neural Network
    n_x - input layer - size n
    n_h - hidden layer - size h
    n_y - output layer - size 1

    :param X: Training data set data points as columns
    :param Y: Training data label set
    :param n_h: hidden layer dimension
    :param num_iterations: epochs
    :param print_cost: dfault false, if true ,will print the cost at each 1000 epoch
    :return: trained parameters (weights and biases of NN)
    """

    [m, n] = X.shape
    Y = np.reshape(Y, (1, Y.shape[0]))

    n_x = n
    n_h = n_h
    n_y = 1

    parameters = nn.initialize_parameters(n_x, n_h, n_y)

    for i in range(0, num_iterations):
        nn_out, cash = nn.forward_propagation(X.T, parameters)
        cost = nn.cost_function(nn_out, Y, parameters)
        grads = nn.backward_propagation(parameters, cash, X.T, Y)
        parameters = nn.update_parameters(parameters, grads, learning_rate=0.1)

        accuracy = np.mean(nn_out == Y)
        if print_cost and i % 1000 == 0:
            print(f"iteration: {i}  accuarcy:{accuracy}   cost: {cost}")

        if accuracy == 1.0:
            print(f"iteration: {i}  accuarcy:{accuracy}   cost: {cost}")
            break
    return parameters


n_h = 32
parameters = nn_model(X_train, y_train, n_h=n_h, num_iterations=120000, print_cost=True)

# Todo 8. Apply for test data
nn_out, cash = nn.forward_propagation(X_test.T, parameters)
cost = nn.cost_function(nn_out, y_test, parameters)
print(f"Test data accuracy: {np.mean(nn_out == y_test)} and cost {cost}")

"""
What to submit
Collect the Boolean predictions of your classifier on the testing set in a single text file (as a sequence of +1 and -1 separated by spaces in a single line). 
Also store your net as a text file using the following format. 
The first line should be an integer that represents the number of hidden units. 
The second line should be the sequence of weights used by the output units. 
The subsequent lines should be the sequence of weights used by each hidden unit on the inputs (ordered in the same manner as the data).
So, if you use M hidden units, this file should have M+2 lines. 
The last M lines should have 1000 entries in them. 
Submit via email the two text files along with your source code for the homework.
"""
with open('HW2_results_test_set.txt', "w") as f:
    f.write(" ".join(map(str, nn_out)))
with open('HW2_results_test_set.txt', "a") as f:
    f.write("\n ")
    f.write(f"Test data accuracy: {np.mean(nn_out == y_test)} and cost {cost}\n")
# ---------------------------------------------------------------------------------------#
with open('HW2_results_weights.txt', "w") as f:
    f.write(f"{n_h}\n")

# with open('HW2_results_weights.txt', 'a') as f:
#     f.write(f"weights used by the output units \n")
#     np.savetxt(f, parameters['w2'], delimiter=' ', fmt='%f', newline='\n')
#     f.write(f"weights used by the hidden units \n")
#     np.savetxt(f, parameters['w1'], delimiter=' ', fmt='%f', newline='\n')
#     f.write(f"bias used by the output units \n")
#     np.savetxt(f, parameters['b2'], delimiter=' ', fmt='%f', newline='\n')
#     f.write(f"biases used by the hidden units \n")
#     np.savetxt(f, parameters['b1'], delimiter=' ', fmt='%f', newline='\n')

with open('HW2_results_weights.txt', 'a') as f:
    f.write(f"weights and biases used by the output units \n")
    np.savetxt(f, np.concatenate((parameters['w2'], parameters['b2']), axis=1), delimiter=' ', fmt='%f')
    f.write(f"weights and biases used by the hidden units \n")
    np.savetxt(f, np.concatenate((parameters['w1'], parameters['b1']), axis=1), delimiter=' ', fmt='%f', newline='\n')
