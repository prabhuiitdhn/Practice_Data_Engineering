"""
https://scikit-learn.org/stable/auto_examples/feature_selection/plot_select_from_model_diabetes.html#sphx-glr-auto-examples-feature-selection-plot-select-from-model-diabetes-py

Two approaches for feature selection
1. selectfrommodel: based on feature importance
2. SequentialFeatureSelection: which relies on greedy approach

 Diabetes dataset, which consists of 10 features collected from 442 diabetes patients.

"""

# loading the data
import pandas as pd
pd.set_option('display.max_columns', None)
from sklearn.datasets import load_diabetes

diabetes = load_diabetes(return_X_y=False, as_frame=False)
# as_frame is nothing but pandas data frame
# return_X_y: If True, returns ``(data, target)`` instead of a Bunch object.

# print("Data description:")
# print(diabetes.DESCR)  # It is describing the data
X, y = diabetes.data, diabetes.target

# # # # # # feature importance from coefficients To get an idea of the importance of the features, ######## we are
# going to use the RidgeCV estimator. The features with the highest absolute coef_ value are considered the most
# important.

import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import RidgeCV
# RidgeCV: Ridge regression with built-in cross-validation.
# Ridge regression is a model tuning method that is used to analyse any data that suffers from multi-collinearity

ridge = RidgeCV(alphas=np.logspace(-6, 6, num=5)).fit(X, y)
importance = np.abs(ridge.coef_)
feature_names = np.array(diabetes.feature_names)
plt.bar(height=importance, x=feature_names)
plt.title("Feature importances via coefficients")
plt.show()

# # # # # SElecting features based on the importance
from time import time
from sklearn.feature_selection import SelectFromModel
# SelectFromModel: Meta-transformer for selecting features based on importance weights.
threshold = np.sort(importance)[-3] + 0.01
tic= time()
selectFromModel = SelectFromModel(ridge, threshold).fit(X, y)
toc = time()
print(f"Features selected by SelectFromModel: {feature_names[selectFromModel.get_support()]}")
print(f"Done in {toc - tic:.3f}s")
