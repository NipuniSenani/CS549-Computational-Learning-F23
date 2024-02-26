import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split

from adaBoost97 import AdaBoost97
from adaBoost97SVM import AdaBoost97SVM

# Todo :Load the digits dataset

"""
This dataset is made up of 1797 8x8 images.
Each image, like the one shown below, is of a hand-written digit. 
In order to utilize an 8x8 figure like this,
 we’d have to first transform it into a feature vector with length 64.
"""
digits = datasets.load_digits()
# ----------------------------------------------------#
X = digits.data

y = [1.0 if digit % 2 == 1 else -1.0 for digit in digits.target]

# ----------------------------------------------------#

print(f"dataset X shape: {digits.data.shape}")
print(f"dataset y(target) shape: {digits.target.shape}")
print(f"dataset y(target) classes: {digits.target_names}")

print(f"digits : \n{digits.target[:10]} \nodd-even classification: \n{y[:10]}")

# Todo : Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=2)
# Todo; Number of weak Classifiers
T = 10
# Todo : Fit the Ada Boost97 model
adaBoost97 = AdaBoost97()
adaBoost97.model_fit(X_train, y_train, T=T)

u=5
# Todo : get the Ada Boost97 classifier prediction
y_pred97 = adaBoost97.predict(X=X_test, y=y_test)


# Todo : Fit the Ada Boost97SVM model
adaBoost97R = AdaBoost97SVM()
adaBoost97R.model_fit(X_train, y_train, T=T)

# Todo : get the Ada Boost97SVM classifier prediction
y_pred97R = adaBoost97R.predict(X=X_test, y=y_test)
#
# Todo: Sklearn AdaBoost model(default weak learner type:DecisionTreeClassifier with depth 1(slump))
sk_adaBoost = AdaBoostClassifier(n_estimators=T, random_state=0)
sk_adaBoost.fit(X_train, y_train)
y_pred_sk = sk_adaBoost.predict(X=X_test[:u, :])

print(f"predictions \n AdaBoost97 : \n{y_pred97[:u]} "
      f"\nAdaBoost97SVM : \n{y_pred97R[:u]}\n SK Adabt : \n{y_pred_sk}")
print(f" Accuracy of the test set 97 :{adaBoost97.score}")
print(f" Accuracy of the test set 97SVM :{adaBoost97R.score}")
print(f" Accuracy of the test set  Sklearn AB:{sk_adaBoost.score(X_test, y_test)}")

plt.figure()

x = np.arange(1, T + 1)
# Plot data on each subplot
plt.subplot(1, 2, 1)
plt.plot(adaBoost97.training_errors, color='blue')
plt.title('AdaBoost97-DecisionTree')
plt.subplot(1, 2, 2)
plt.plot(adaBoost97R.training_errors, color='red')
plt.title('AdaBoost97-SVM')

# Adjust layout to prevent clipping of titles
plt.tight_layout()

# Show the plots
plt.show()
