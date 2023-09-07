"""
https://scikit-learn.org/stable/auto_examples/compose/plot_digits_pipe.html#sphx-glr-auto-examples-compose-plot-digits-pipe-py
The PCA does an unsupervised dimensionality reduction, while the logistic regression does the prediction.
GridSearchCV to set the dimensionality of the PCA
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from  sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Define a pipeline to search for the best combination of PCA truncation
# and classifier regularization.
pca = PCA()
scaler = StandardScaler()
logistic = LogisticRegression(
    max_iter=1000,
    tol=0.1
)

pipeline = Pipeline(
    [
        ("Scaler", scaler),
        ("pca", pca),
        ("logistic", logistic)
    ]
)

X_digit, y_digits = load_digits(
    return_X_y=True
)

param_grid = {
    # # Parameters of pipelines can be set using '__' separated parameter names: in this example:
    # reduce_dimension is pipeline parameters on PCA and n_components is the internal parameters for PCA so,
    # reduce_dimension__n_components becomes the parameters for pipeline.
    "pca__n_components": [5, 15, 30, 45, 60],
    "logistic__C": np.logspace(-4, 4, 4),
}

grid_search = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    verbose=10,
    n_jobs=2
)

grid_search.fit(X_digit, y_digits)
print("Best estimator using Grid search.")
print(grid_search.best_estimator_)
print("Best parameters using Grid search.")
print(grid_search.best_params_)
print("Best score using Grid search")
print(grid_search.best_score_)

# PLot the pca spectrum
pca.fit(X_digit)


fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(6, 6))
ax0.plot(
    np.arange(1, pca.n_components_ + 1), pca.explained_variance_ratio_, "+", linewidth=2
)
ax0.set_ylabel("PCA explained variance ratio")

ax0.axvline(
    grid_search.best_estimator_.named_steps["pca"].n_components,
    linestyle=":",
    label="n_components chosen",
)
ax0.legend(prop=dict(size=12))

# For each number of components, find the best classifier results
results = pd.DataFrame(grid_search.cv_results_)
components_col = "param_pca__n_components"
best_clfs = results.groupby(components_col).apply(
    lambda g: g.nlargest(1, "mean_test_score")
)

best_clfs.plot(
    x=components_col, y="mean_test_score", yerr="std_test_score", legend=False, ax=ax1
)
ax1.set_ylabel("Classification accuracy (val)")
ax1.set_xlabel("n_components")

plt.xlim(-1, 70)

plt.tight_layout()
plt.show()


