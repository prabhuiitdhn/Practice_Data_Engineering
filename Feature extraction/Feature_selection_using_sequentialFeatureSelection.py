"""
https://scikit-learn.org/stable/auto_examples/feature_selection/plot_select_from_model_diabetes.html#model-based-and-sequential-feature-selection

SequentialFeatureSelection: which relies on greedy approach
SFS is a greedy procedure where, at each iteration, we choose the best new feature to add to our selected features based a cross-validation score.
we start with 0 features and choose the best single feature with the highest score. The procedure is repeated until we reach the desired number of selected features.


"""
from time import time
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import RidgeCV

# getting the diabetes dataset
diabetes = load_diabetes()

# getting the data and target
X, y = diabetes.data, diabetes.target

# features name for diabetes
feature_names = np.array(diabetes.feature_names)

# this ridge regression linear model estimator for selecting the features based on the sequentially adding the features
ridge = RidgeCV(alphas=np.logspace(-6, 6, num=5)).fit(X, y)

tic_fwd = time() # Start time for featureselector forward direction
sfs_forward = SequentialFeatureSelector(
    ridge, n_features_to_select=2, direction="forward"
).fit(X, y)

toc_fwd = time() # End time for featureselector forward direction

tic_bwd = time() # # Start time for featureselector Backward direction
sfs_backward = SequentialFeatureSelector(
    ridge, n_features_to_select=2, direction="backward"
).fit(X, y)
toc_bwd = time() # # End time for featureselector forward direction
print(
    "Features selected by forward sequential selection: "
    f"{feature_names[sfs_forward.get_support()]}"
)
print(f"Done in {toc_fwd - tic_fwd:.3f}s")
print(
    "Features selected by backward sequential selection: "
    f"{feature_names[sfs_backward.get_support()]}"
)
print(f"Done in {toc_bwd - tic_bwd:.3f}s")