

# Todo 2. Initiaize model parameters-

def initialize_parameters(n_x, n_h, n_y):
    """
    Weight matrices and bias sizes

    w1 - size (n_h x n-x)
    b1 - size (n_h x 1)
    w2 - size (n_y x n_h)
    b2 - size (n_y x 1)

    :param n_x: Input layer dimension
    :param n_h: Hidden Layer dimension
    :param n_y: Output layer dimension
    :return: weights and biases as dictionary
    {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}
    """
    w1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    w2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    p = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}

    return p


# -------------------------------------------------#
def relu(x):
    return np.maximum(0, x)


# -------------------------------------------------#

# Todo 3.Forward propagation-

def forward_propagation(X, parameters):
    """
    input
    X - size( # features, # data points)
    parameters - a dictionary with all weights and biases
    {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}


    z1 - hidden layer output before activation function -
    a1 - hidden layer output after activation applied
    z2 - output layer values before activation
    a2 - output layer values after activation


    output
    cashe - a dictionary with outputs of each layer before aand after activation applied.
    {'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}

    a2 - output of the nn
    """
    w1 = parameters['w1']
    w2 = parameters['w2']
    b1 = parameters['b1']
    b2 = parameters['b2']

    z1 = np.dot(w1, X) + b1
    a1 = relu(z1)
    z2 = np.dot(w2, z1) + b2
    a2 = np.sign(np.tanh(z2))

    cash = {'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}
    return a2, cash


# Todo 4.Compute Loss-

def cost_function(nn_out, true_y, parameters):
    """
    cost calculate by summing binary Cross_Entropy loss
    (log-likelihood method for error estimate)
    :param nn_out: NN output
    :param true_y: true output
    :param parameters:a dictionary with all weights and biases
    {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}
    :return: cost value of current iteration
    """
    m = parameters['w1'].shape[1]

    se = (1 / 2) * np.power(true_y - nn_out, 2)
    cost = (1 / m) * np.sum(se)

    return cost


# Todo 5. BackPropagation/Gradient Descent (GD)-

def backward_propagation(parameters, cash, X, Y):
    """

    :param parameters:weights and biases
    :param cash: each layer output before and after activation
    :param X: trained data set data point as a column
    :param Y: train set labels
    :return: grad of each weight and bias
    """
    m = parameters['w1'].shape[1]

    w1 = parameters['w1']
    w2 = parameters['w2']
    b1 = parameters['b1']
    b2 = parameters['b2']

    z1 = cash['z1']
    a1 = cash['a1']
    z2 = cash['z2']
    a2 = cash['a2']

    dz2 = a2 - Y
    dw2 = (1 / m) * np.dot(dz2, a1.T)
    db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)
    dz1 = np.multiply(np.dot(w2.T, dz2), 1 - np.power(a1, 2))
    dw1 = (1 / m) * np.dot(dz1, X.T)
    db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)

    grads = {'dw1': dw1, 'db1': db1, 'dw2': dw2, 'db2': db2}

    return grads


# Todo 6. Update the parameters-

def update_parameters(parameters, grads, learning_rate=0.1):
    # w1 = parameters['w1']
    # w2 = parameters['w2']
    # b1 = parameters['b1']
    # b2 = parameters['b2']

    dw1 = grads['dw1']
    db1 = grads['db1']
    dw2 = grads['dw2']
    db2 = grads['db2']

    parameters['w1'] = parameters['w1'] - learning_rate * dw1
    parameters['b1'] = parameters['b1'] - learning_rate * db1
    parameters['w2'] = parameters['w2'] - learning_rate * dw2
    parameters['b2'] = parameters['b2'] - learning_rate * db2
    return parameters