"""
FeatureUnion: for combining multiple features extracted from the datasets.

This example shows how to use FeatureUnion to combine features obtained by PCA and univariate selection.

Combining features using this transformer has the benefit that it allows cross validation and grid searches over the whole process.
"""

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.svm import SVC
# `sklearn.decomposition` module includes matrix decomposition algorithms, including among others PCA, NMF or ICA.
# Most of the algorithms of this module can be regarded as dimensionality reduction techniques.

from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import FeatureUnion, Pipeline

# FeatureUnion: Concatenates results of multiple transformer objects.

iris = load_iris()
X, y = iris.data, iris.target

# this dataset is too high dimensional, so it is better to work with PCA
pca = PCA(n_components=2)

# If it have more dimension then it is better to select 1
selection = SelectKBest(k=1)

# Build estimator from PCA and Univariate selection:

# FeatureUnion: This combines the multiple features from the dataset
# THIS IS WHERE CONCATENATION PERFORMS
combined_feature = FeatureUnion([
    ("pca", pca),
    ("univ_select", selection)
])

# DO searchOverK, n_component and C
# Use combined features to transform dataset:
X_features = combined_feature.fit(X, y).transform(X)
print("Combined space has", X_features.shape[1], "features")
svm = SVC(kernel="linear")

pipeline = Pipeline(
    steps=[
        ("features", combined_feature),
        ("svm", svm)
    ]
)

param_grid = dict(
    features__pca__n_components=[1, 2, 3],
    features__univ_select__k=[1, 2],
    svm__C=[0.1, 1, 10]
)

# GridSearchCV: Exhaustive search over specified parameter values for an estimator.
#     verbose : int
#         Controls the verbosity: the higher, the more messages.
#
#         - >1 : the computation time for each fold and parameter candidate is
#           displayed;
#         - >2 : the score is also displayed;
#         - >3 : the fold and candidate parameter indexes are also displayed
#           together with the starting time of the computation.

grid_search = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    verbose=10
)

grid_search.fit(X, y)
print("Best_estimators:")
print(grid_search.best_estimator_) # Gives the estimator which really performed well out the SVC__C param in param_grid
 