"""
https://scikit-learn.org/stable/auto_examples/compose/plot_compare_reduction.html#sphx-glr-auto-examples-compose-plot-compare-reduction-py

Example:
    1. Construct a pipeline PCA followed by SVM
    2. Demonstrate the use of GridSearchCV, pipeline
        - Optimised over different classes of estimatores in single cv run
        - unsupervised PCA, NMF dimensionality reductions are compared to univariate feature selection during GridSearch
        - Instantiated with Memory of pipeline which caches the pipeline in local directory

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA, NMF

# NMF: Find two non-negative matrices, i.e. matrices with all non-negative elements, (W, H)
#     whose product approximates the non-negative matrix X. This factorization can be used
#     for example for dimensionality reduction, source separation or topic extraction.

from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

X, y = load_digits(return_X_y=True)

pipeline = Pipeline(
    steps=
    [
        ("Scaling", MinMaxScaler()),
        # By specifying remainder='passthrough' , all remaining columns that were not specified in transformers ,
        # but present in the data passed to fit will be automatically passed through.
        ("reduce_dimension", "passthrough"),
        ("Classification", LinearSVC(dual=True, max_iter=10))
    ]
)

N_FEATURES_OPTIONS = [2, 4, 8]
C_OPTIONS = [1, 10, 100, 1000]

param_grid = [
    {
        # # Parameters of pipelines can be set using '__' separated parameter names: in this example:
        # reduce_dimension is pipeline parameters on PCA and n_components is the internal parameters for PCA so,
        # reduce_dimension__n_components becomes the parameters for pipeline.

        "reduce_dimension": [PCA(iterated_power=7), NMF(max_iter=10)],
        "reduce_dimension__n_components": N_FEATURES_OPTIONS,
        "Classification__C": C_OPTIONS

    },
    {
        "reduce_dimension": [SelectKBest(mutual_info_classif)],
        "reduce_dimension__k": N_FEATURES_OPTIONS,
        "Classification__C": C_OPTIONS,
    }
]

reducer_labels = ["PCA", "NMF", "KBest(mutual_info_classif)"]
grid_search = GridSearchCV(pipeline,
                           param_grid=param_grid,
                           n_jobs=1)

with ignore_warnings(category=ConvergenceWarning):
    grid_search.fit(X, y)
print("Best estimator:", grid_search.best_estimator_)
print("Best score:", grid_search.best_score_)
print("Best parameter:", grid_search.best_params_)

# # PLOT of comparing features reduction techniques

mean_scores = np.array(
    # CV_results will store all the results in dictionary format: dict with keys as column headers and values as
    # columns, that can be imported into a pandas ``DataFrame``.
    grid_search.cv_results_["mean_test_score"]
)

# scores are in the order of param_grid iteration, which is alphabetical
mean_scores = mean_scores.reshape(len(C_OPTIONS), -1, len(N_FEATURES_OPTIONS))

# select score for best C
mean_scores = mean_scores.max(axis=0)

# create a dataframe to ease plotting
mean_scores = pd.DataFrame(
    mean_scores.T, index=N_FEATURES_OPTIONS, columns=reducer_labels
)

ax = mean_scores.plot.bar()
ax.set_title("Comparing feature reduction techniques")
ax.set_xlabel("Reduced number of features")
ax.set_ylabel("Digit classification accuracy")
ax.set_ylim((0, 1))
ax.legend(loc="upper left")

plt.show()

# USING MEMORY AS PIPELINE TO SAVE THE PIPELINE IN CACHE It is sometimes worthwhile storing the state of a specific
# transformer since it could be used again. Using a pipeline in GridSearchCV triggers such situations. Therefore,
# we use the argument memory to enable caching.

from shutil import rmtree
from joblib import Memory

location = 'cached_pipeline'
memory = Memory(
    location = location,
    verbose=10
)


cached_pipe = Pipeline(
    [("reduce_dim", PCA()), ("classify", LinearSVC(dual=False, max_iter=10))],
    memory=memory,
)

# This time, a cached pipeline will be used within the grid search


# Delete the temporary cache before exiting
memory.clear(warn=False)
# rmtree(location)