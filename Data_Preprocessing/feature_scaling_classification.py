"""
https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html#sphx-glr-auto-examples-preprocessing-plot-scaling-importance-py

Feature scaling is the process of normalizing the range of features in a dataset. This is done to ensure that all features are on a similar scale, which can help to improve the performance of machine learning algorithms.

Normalization: This involves transforming the features so that they have a range of 0 to 1. This is often done by subtracting the minimum value from each feature and then dividing by the maximum value minus the minimum value.
Standardization: This involves transforming the features so that they have a mean of 0 and a standard deviation of 1. This is done by subtracting the mean from each feature and then dividing by the standard deviation.

- Used for to ease the convergence, to create completely different model fit compared to the fit with unscaled data.


Algorithm which uses feature scaling:
1. Linear regression
2. Logistic regression
3. Support vector machines
4. K-nearest neighbors
5. Principal component analysis
6. nueral networks

"""
# Example:
# part1 : if we fit the data with normalised and unscaled data then the model becomes completely different
# part2: How PCA is impacted by normalisation of features.

# # # # # # # # # #  Loading and preparing the data # # # # # # # # # # # # # # #

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
pd.set_option('display.max_columns', None)

# loading the data
X, y = load_wine(return_X_y=True,  # it return as (data, Target)
                 as_frame=True  # it returns the data as pandas data frame
                 )

# preprocessing the data as standardlisaition
scaler = StandardScaler().set_output(transform="pandas")

# splitting the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print("features in wine data:\n", X)
# alcohol  malic_acid   ash  alcalinity_of_ash  magnesium  total_phenols flavanoids  nonflavanoid_phenols
# proanthocyanins  color_intensity   hue od280/od315_of_diluted_wines  proline

# Transforming the data
scaled_X_train = scaler.fit_transform(X_train)

# # # # # Effect of rescaling on a k-neighbor models  # # # # # # # #

import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay  # Decisions boundary visualization.
# sklearn.inspection: Module includes tools for model inspection

from sklearn.neighbors import KNeighborsClassifier

# For sake of the visualisation we will only taking two features "proline, Hue"
X_plot = X[["proline", "hue"]]
X_plot_scaled = scaler.fit_transform(X_plot)

clf = KNeighborsClassifier(n_neighbors=20)


def fit_and_plot_model(X_plot, y, clf, ax):
    clf.fit(X_plot, y)
    disp = DecisionBoundaryDisplay.from_estimator(
        clf,
        X_plot,
        response_method="predict",
        alpha=0.5,
        ax=ax,
    )
    disp.ax_.scatter(X_plot["proline"], X_plot["hue"], c=y, s=20, edgecolor="k")
    disp.ax_.set_xlim((X_plot["proline"].min(), X_plot["proline"].max()))
    disp.ax_.set_ylim((X_plot["hue"].min(), X_plot["hue"].max()))
    return disp.ax_

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
fit_and_plot_model(X_plot, y, clf, ax1)
ax1.set_title("KNN without scaling")
fit_and_plot_model(X_plot_scaled, y, clf, ax2)
ax2.set_xlabel("scaled proline")
ax2.set_ylabel("scaled hue")
_ = ax2.set_title("KNN with scaling")
plt.show()