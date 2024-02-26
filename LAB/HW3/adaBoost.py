import numpy as np
from sklearn.tree import DecisionTreeClassifier


class AdaBoost:
    """
    Algorithm taken from
    The Elements of Statistical Learning (Data Mining, Inference, and Prediction) by Trevor Hastie, Robert Tibshirani,
    Jerome Friedman

    second edition
    page: 358
    ch 10
    Algorithm 10.1 AdaBoost.M1.

    Website reference:https://towardsdatascience.com/adaboost-from-scratch-37a936da3d50
    """

    def __init__(self):
        self.G_M = []
        self.score = None
        self.M = None
        self.alphas = []
        self.training_errors = []

    # Defininig independent chunks of the algorithm

    # Todo 1: functon for compute error
    def compute_error(self, y, y_pred, w_i):
        """
        Calculate the error rate of a weak classifier m.

        Inputs:
        :param y: actual target value
        :param y_pred: predicted value by weak classifier
        :param w_i: individual weights for each observation

        Note that all arrays should be the same length

       :return:error rate of a weak classifier m

        """
        return (sum(w_i * (np.not_equal(y, y_pred)).astype(int))) / sum(w_i)

    # Todo 2: functon for compute alpha
    def compute_alpha(self, error):
        """
        Calculate the weight of a weak classifier m in the majority vote of the final classifier
        :param error: error rate from weak classifier m
        :return: alpha - the weight of a weak classifier m
        """

        return np.log((1 - error) / error)

    # Todo 3: functon for compute weights
    def update_weights(self, w_i, alpha, y, y_pred):
        """
        Update individual weights w_i after a boosting iteration.
        :param w_i: individual weights for each observation
        :param alpha: weight of weak classifier m used to estimate y_pred
        :param y: actual target value
        :param y_pred: predicted value by weak classifier

        :return: updated w_i weight after a boosting iteration
        """
        return w_i * np.exp(alpha * (np.not_equal(y, y_pred)).astype(int))

    def model_fit(self, X, y, M=100):
        """
        Fit the Ada Boost model
        :param X: indpendent variable
        :param y: actual target values {-1, +1} type-vector(array)
        :param M: number of Boosting rounds

        :return: a set of stumps, alphas and training errors that we store for further use in the predict method
        """
        #
        # Clear before calling
        self.alphas = []
        self.training_errors = []
        self.M = M

        for m in range(M):
            # Todo(1) Initialize the observation weights wi = 1/N, i = 1, 2, . . . ,N.
            if m == 0:
                w_i = np.ones(len(y)) * 1 / len(y)  # At m = 0, weights are all the same and equal to 1 / N
            else:
                # Todo(2d) Update w_i
                w_i = self.update_weights(w_i, alpha_m, y, y_pred)

            # Todo(2a) Fit a classifier G(x) to the training data using weights w_i.
            G_m = DecisionTreeClassifier(max_depth=1)  # Stump: Two terminal-node classification tree
            G_m.fit(X, y, sample_weight=w_i)
            y_pred = G_m.predict(X)

            self.G_M.append(G_m)  # Save to a list weak classifier m

            # Todo(2b) Compute error to compute alpha in the next step
            error_m = self.compute_error(y, y_pred, w_i)
            self.training_errors.append(error_m)

            # Todo(2c) Compute alpha
            alpha_m = self.compute_alpha(error_m)
            self.alphas.append(alpha_m)

        assert len(self.G_M) == len(self.alphas)

    def predict(self, X, y):
        """
        Predict using fitted model
        :param X: Independent variables
        :return: Ada Boost prediction of the classifier for X
        """

        # Initialise array with weak predictions for each observation
        weak_preds = np.zeros((len(X), self.M))

        # Predict class label for each weak classifier, weighted by alpha_m
        for m in range(self.M):
            y_pred_m = self.G_M[m].predict(X) * self.alphas[m]  # DecisionTreeClassifier prediction
            weak_preds[:, m] = y_pred_m

        # Calculate final predictions
        y_pred = (np.sign(weak_preds.sum(axis=1))).astype(int)  # row vector

        # Accuracy of test data
        self.score = float(np.sum((y == y_pred).astype(int)) / len(y))
        return y_pred
