import numpy as np
from sklearn.svm import SVC


class AdaBoost97SVM:

    def __init__(self):

        self.score = None
        self.H_T = []
        self.T = None
        self.training_errors = []
        self.alphas = []

    # Defininig independent chunks of the algorithm

    # Todo 1: functon for compute error
    def compute_error(self, y, y_pred, D_m):
        """
        Calculate the error rate of a weak classifier m.

        Inputs:
        :param y: actual target value
        :param y_pred: predicted value by weak classifier
        :param w_i: individual weights for each observation

        Note that all arrays should be the same length

       :return:error rate of a weak classifier m

        """
        return (sum(D_m * (np.not_equal(y, y_pred)).astype(int))) / sum(D_m)

    # Todo 2: functon for compute alpha
    def compute_alpha(self, error):
        """
        Calculate the weight of a weak classifier m in the majority vote of the final classifier
        :param error: error rate from weak classifier m
        :return: alpha - the weight of a weak classifier m
        """

        return 0.5 * np.log((1 - error) / error)

    # Todo 3: functon for compute weights
    def update_weights(self, D_m, alpha, y, y_pred):
        """
        Update individual weights w_i after a boosting iteration.
        :param D_m: individual weights for each observation
        :param alpha: weight of weak classifier m used to estimate y_pred
        :param y: actual target value
        :param y_pred: predicted value by weak classifier

        :return: updated w_i weight after a boosting iteration
        """
        u = y * y_pred
        Z_t = sum(D_m * np.exp((-alpha) * u))
        return D_m * np.exp((-alpha) * u) / Z_t

    def model_fit(self, X, y, T=100):
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
        self.T = T

        for t in range(T):
            # Todo(1) Initialize the observation weights D_m = 1/m, i = 1, 2, . . . ,M.
            if t == 0:
                D_m = np.ones(len(y)) * 1 / len(y)  # At m = 0, weights are all the same and equal to 1 / N
            else:
                # Todo(2d) Update w_i
                D_m = self.update_weights(D_m, alpha_t, y, y_pred)

            # Todo(2a) Fit a classifier G(x) to the training data using weights w_i.
            H_t = SVC(kernel='linear', C=0.025)
            H_t.fit(X, y, sample_weight=D_m)
            y_pred = H_t.predict(X)

            self.H_T.append(H_t)  # Save to a list weak classifier m

            # Todo(2b) Compute error to compute alpha in the next step
            error_t = self.compute_error(y, y_pred, D_m)
            self.training_errors.append(error_t)

            # Todo(2c) Compute alpha
            alpha_t = self.compute_alpha(error_t)
            self.alphas.append(alpha_t)

        assert len(self.H_T) == len(self.alphas)

    def predict(self, X, y):
        """
        Predict using fitted model
        :param X: Independent variables
        :return: Ada Boost prediction of the classifier for X
        """

        # Initialise array with weak predictions for each observation
        weak_preds = np.zeros((len(X), self.T))

        # Predict class label for each weak classifier, weighted by alpha_m
        for t in range(self.T):
            y_pred_t = self.H_T[t].predict(X) * self.alphas[t]  # DecisionTreeClassifier prediction
            weak_preds[:, t] = y_pred_t

        # Calculate final predictions
        y_pred = (np.sign(weak_preds.sum(axis=1))).astype(int)  # row vector

        # Accuracy of test data
        self.score = float(np.sum((y == y_pred).astype(int)) / len(y))
        print(f"test set is predicted correctly(SVM)  {np.sum((y == y_pred).astype(int))}  out of {len(y)}")
        return y_pred
