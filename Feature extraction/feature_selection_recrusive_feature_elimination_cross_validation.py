"""

https://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_with_cross_validation.html#sphx-glr-auto-examples-feature-selection-plot-rfe-with-cross-validation-py

Recursive feature elimination: This method starts with all the features and then eliminates features one at a time,
based on their importance to the model.

A Recursive Feature Elimination: example with automatic tuning of the number of features selected with cross-validation.

"""

# We build a classification task using 3 informative features. The introduction of 2 additional redundant (i.e.
# correlated) features has the effect that the selected features vary depending on the cross-validation fold.
# The remaining features are non-informative as they are drawn at random.

# # # # Data preparation

from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=500,
    n_classes=8,
    n_features=15,
    n_informative=3,
    n_redundant=2,
    n_repeated=0,
    n_clusters_per_class=1,
    class_sep=0.8,
    random_state=0
)

# Model training and selection
# RFECV: Recursive feature elimination with cross validation

from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

min_feature_to_select = 1
classification_model = LogisticRegression()
cross_validation = StratifiedKFold(n_splits=5)

rfecv = RFECV(
    estimator=classification_model,
    step=1,
    cv= cross_validation,
    scoring="accuracy",
    min_features_to_select=min_feature_to_select,
    n_jobs=2
)

rfecv.fit(X, y)
print(f"Optimal number of features: {rfecv.n_features_}")


# # # # # Plot number of features VS. cross-validation scores¶

import matplotlib.pyplot as plt

n_scores = len(rfecv.cv_results_["mean_test_score"])
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Mean test accuracy")
plt.errorbar(
    range(min_feature_to_select, n_scores + min_feature_to_select),
    rfecv.cv_results_["mean_test_score"],
    yerr=rfecv.cv_results_["std_test_score"],
)
plt.title("Recursive Feature Elimination \nwith correlated features")
plt.show()