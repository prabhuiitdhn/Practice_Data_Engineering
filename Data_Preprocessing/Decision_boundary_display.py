import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.tree import DecisionTreeClassifier
iris = load_iris()
feature_1, feature_2 = np.meshgrid(np.linspace(iris.data[:, 0].min(), iris.data[:, 0].max()),
                                   np.linspace(iris.data[:, 1].min(), iris.data[:, 1].max()))

grid = np.vstack([feature_1.ravel(), feature_2.ravel()]).T
tree = DecisionTreeClassifier().fit(iris.data[:, :2], iris.target)
y_pred = np.reshape(tree.predict(grid), feature_1.shape)
display = DecisionBoundaryDisplay(xx0=feature_1, xx1 = feature_2, response = y_pred)
display.plot()
display.ax_.scatter(iris.data[:, 0], iris.data[:, 1], c = iris.target, edgecolor = "black")
plt.show()