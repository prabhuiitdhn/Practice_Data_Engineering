"""

https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-py

In this example we will compare the impurity based feature importance of RandomForestClassifier with the
permutation importance on the dataset using permuatation importance.

The impurity-based feature importance of random forests suffers from being computed on statistics derived from the
training dataset, the importances can be high even for features that are not predictive of the target variable,
 as long as the model has the capacity to use them to overfit.


The impurity-based feature importance ranks the numerical features to be the most important features. As a result,
    the non-predictive random_num variable is ranked as one of the most important features!

This problem stems from two limitations of impurity-based feature importances:
    impurity-based importances are biased towards high cardinality features;
    impurity-based importances are computed on training set statistics and
    therefore do not reflect the ability of feature to be useful to make predictions that generalize to the test set
    (when the model has enough capacity).
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

X, y = fetch_openml(
    "titanic",  # Fetching the titanic dataset
    version=1,
    as_frame=True,  # the data is a pandas DataFrame including columns with appropriate dtypes
    return_X_y=True,  # - return as (data, target)
    parser="pandas"  # parser.
)

rng = np.random.RandomState(seed=42)

# We further include two random variables that are not correlated in any way with the target variable
X["random_cat"] = rng.randint(3, size=X.shape[0])
X["random_num"] = rng.randn(X.shape[0])

categorical_columns = ["pclass", "sex", "embarked", "random_cat"]  # this columns are available in the Titanic dataset.
numerical_columns = ["age", "sibsp", "parch", "fare", "random_num"]

X = X[categorical_columns + numerical_columns]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=42
)

# We will define the predictive model based on random forest, so, we will make preprocessing steps.

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

# To preprocess the categorical feature use OrdinalEncoder

categorical_encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1,
    encoded_missing_value=-1
    # handle_unknown : {'error', 'use_encoded_value'}, default='error'
    # When set to 'error' an error will be raised in case an unknown
    # categorical feature is present during transform. When set to
    # 'use_encoded_value', the encoded value of unknown categories will be
    # set to the value given for the parameter `unknown_value`.
)

numerical_pipeline = SimpleImputer(strategy="mean")

preprocessing_pipeline = ColumnTransformer(
    [
        ("cat", categorical_encoder, categorical_columns),
        ("num", numerical_pipeline, numerical_columns)
    ],
    verbose_feature_names_out=False
)

random_forest = Pipeline(
    [
        ("preprocess", preprocessing_pipeline),
        ("classifier", RandomForestClassifier(random_state=42))
    ]
)

random_forest.fit(X_train, y_train)

print(random_forest)

# # # Accuracy of the model
# print("Random forest Training accuracy:", random_forest.score(X_train, y_train))
# print("Rendom forest testing accuracy:", random_forest.score(X_test, y_test))
# print("Random forest features:", random_forest.feature_names_in_)
# print("Random forest number of features:", random_forest.n_features_in_)

# Check which features having the highest importance.

import pandas as pd

feature_names = random_forest[:-1].get_feature_names_out()

impurity_features_importances = pd.Series(
    random_forest[-1].feature_importances_, index=feature_names
).sort_values(ascending=True)

axis = impurity_features_importances.plot.barh()
axis.set_title("Random forest feature importance.")
axis.figure.tight_layout()
plt.show()

# # Check with permutation importance.

from sklearn.inspection import permutation_importance

# permutation_importance: Permutation importance for feature evaluation

result = permutation_importance(
    estimator=random_forest,
    X=X_test,
    y=y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=2
)
# returns
# importances_mean : ndarray of shape (n_features, ); Mean of feature importance over `n_repeats`.
# importances_std : ndarray of shape (n_features, ); Standard deviation over `n_repeats`.
# importances : ndarray of shape (n_features, n_repeats); raw permutation importance scores.

sorted_importances_idx = result.importances_mean.argsort()

importances = pd.DataFrame(
    result.importances[sorted_importances_idx].T,
    columns=X.columns[sorted_importances_idx]
)

ax = importances.plot.box(vert=False, whis=10)
ax.set_title("Permutation Importances (test set)")
ax.axvline(x=0, color="k", linestyle="--")
ax.set_xlabel("Decrease in accuracy score")
ax.figure.tight_layout()
plt.show()

# TO CHECK THE IMPORANCE OF FEATURE IN THE TRAINING SET as we have added random_cat, random_num
# as categorical features and numerical features.

result = permutation_importance(
    random_forest, X_train, y_train, n_repeats=10, random_state=42, n_jobs=2
)

sorted_importances_idx = result.importances_mean.argsort()
importances = pd.DataFrame(
    result.importances[sorted_importances_idx].T,
    columns=X.columns[sorted_importances_idx],
)
ax = importances.plot.box(vert=False, whis=10)
ax.set_title("Permutation Importances (train set)")
ax.axvline(x=0, color="k", linestyle="--")
ax.set_xlabel("Decrease in accuracy score")
ax.figure.tight_layout()
