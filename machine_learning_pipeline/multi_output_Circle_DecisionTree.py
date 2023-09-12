"""
https://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression_multioutput.html#sphx-glr-auto-examples-tree-plot-tree-regression-multioutput-py

Decision Tree: it learn the local linear regression approximating the circle.
if the maximum depth is too high, the decision trees learns too fine details of the training data and learn from the noise.
"""

# multi output problems

import matplotlib.pyplot as plt
import numpy as np

from sklearn.tree import DecisionTreeRegressor

rng = np.random.RandomState(1)

X = np.sort(
    200 * rng.randn(100, 1) - 100, axis=0
)

y = np.array(
    [np.pi * np.sin(X).ravel(), np.pi * np.cos(X).ravel()]
).T

# print("Training data:", X)
# print("Labelled data:", y)

y[::5, :] += 0.5 - rng.rand(20, 2)
# print("Prepocessed Y:", y)

# fit regression Model
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_3 = DecisionTreeRegressor(max_depth=8)

regr_1.fit(X, y)
regr_2.fit(X, y)
regr_3.fit(X, y)

# predicting

X_test = np.arange(-100.0, 100.0, 0.01)[:, np.newaxis]
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)
y_3 = regr_3.predict(X_test)

# Plotting the figures

plt.figure()
s = 25

plt.scatter(
    y[:, 0],
    y[:, 1],
    c="navy", s=s, edgecolors="black", label="data",
)

plt.scatter(
    y_1[:, 0],
    y_1[:, 1],
    c="cornflowerblue", s=s, edgecolors="black", label="Max_depth_2"
)
plt.scatter(
    y_2[:, 0],
    y_2[:, 1],
    c="red", s=s, edgecolors="black", label="Max_depth_5"
)

plt.scatter(
    y_3[:, 0],
    y_3[:, 1],
    c="orange", s=s, edgecolors="black", label="Max_depth_8"
)

plt.xlim([-6, 6])
plt.ylim([-6, 6])
plt.xlabel("target 1")
plt.ylabel("target 2")
plt.title("Multi-output Decision Tree Regression")
plt.legend(loc="best")
plt.show()
