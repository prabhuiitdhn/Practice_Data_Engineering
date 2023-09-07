"""
The use of forest of trees to evaluate the importance of features on an AI classification task.
"""

# Data generation and model fitting

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=3,
    n_redundant=0,
    n_repeated=0,
    n_classes=2,
    random_state=0,
    shuffle=False
)

# Stratified sampling is a method of random sampling where a population is divided into smaller groups called strata.
# The strata are formed based on shared characteristics, such as race, gender, or educational attainment.
# Once divided, each subgroup is randomly sampled using another probability sampling method. It can be beneficial
# when the population has diverse subgroups and researchers want to be sure that the sample includes all of them.

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    stratify=y,
    random_state=42
)

# A random forest classifier will be fitted to compute the feature importance.
# RandomForest techniques: It is ensemble technique to 'solve the classification, regression problem, and also used for
# feature selection in a nutshell' in machine learning which works through a many decision-Trees in the forest,
# and the trees are in random forest are uncorrelated [uses bagging and feature randomness to build individual Tree]
# so that each individual trees will have their own features to work with which will have higher score, but at the end,
# each tress produces result and using Majority voting/averaging.techniques, score can be decided.


forest = RandomForestClassifier(
    random_state=0
)
forest.fit(X_train, y_train)
print("Forest:", forest)

# Feature importance based on mean decrease in impurity
feature_names = [f"feature {i}" for i in range(X.shape[1])]
print("Feature name:", feature_names)

importance = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
print("Forest importance:", forest.feature_importances_)

forest_importance = pd.Series(importance, index=feature_names)

fig, ax = plt.subplots()
forest_importance.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
plt.show()

# Feature importance based on feature permutation:Permutation feature importance overcomes limitations of the
# impurity-based feature importance: they do not have a bias toward high-cardinality features and can be computed on
# a left-out test set.

from sklearn.inspection import permutation_importance

# permutation_importance: The permutation importance of a feature is calculated as follows.
#     First, a baseline metric, defined by :term:`scoring`, is evaluated on a
#     dataset defined by the `X`. Next, a feature column from the validation set
#     is permuted and the metric is evaluated again. The permutation importance
#     is defined to be the difference between the baseline metric and metric from
#     permutating the feature column.
# sklearn.inspection: module includes tools for model inspection.

start_time = time.time()
result = permutation_importance(
    forest, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
)
elapsed_time = time.time() - start_time
print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

forest_importances = pd.Series(result.importances_mean, index=feature_names)
fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.show()