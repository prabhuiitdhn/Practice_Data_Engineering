"""
https://scikit-learn.org/stable/auto_examples/feature_selection/plot_feature_selection.html#sphx-glr-auto-examples-feature-selection-plot-feature-selection-py

Example of using uni-variate feature selection to improve the classification accuracy.


In this example, some noisy (non informative) features are added to the iris dataset. Support vector machine (SVM) is
used to classify the dataset both before and after applying univariate feature selection. For each feature,
we plot the p-values for the univariate feature selection and the corresponding weights of SVMs. With this,
we will compare model accuracy and examine the impact of univariate feature selection on model weights


"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

iris = load_iris()
print("Iris dataset Description:")
print(iris.DESCR)

X, y = iris.data, iris.target

# Some noisy data not correlated
E = np.random.RandomState(42).uniform(0, 0.1, size=(X.shape[0], 20))

# adding noisy data to informative features

X = np.hstack((X, E))

# split the data : Stratified sampling is a sampling technique used in statistics and data analysis. It involves
# dividing a population into subgroups, or strata, based on certain characteristics or attributes that are important
# to the analysis. Then, samples are randomly selected from each stratum in proportion to their representation in the
# overall population. The goal of stratified sampling is to ensure that each subgroup is adequately represented in
# the sample, which can lead to more accurate and representative results, especially when there are significant
# variations between subgroups.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    stratify=y,
    random_state=0
)

# univariate feature selection

from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(score_func=f_classif, k=4)
selector.fit(X_train, y_train)
scores = -np.log10(selector.pvalues_)
scores /= scores.max()

# # # # # # # PLotting the feature as per the score it gets using f_classi
import matplotlib.pyplot as plt

X_indices = np.arange(X.shape[-1])
plt.figure(1)
plt.clf()  # Clear the current figure.
plt.bar(X_indices - 0.05, scores, width=0.2)
plt.title("Feature univariate score")
plt.xlabel("Feature number")
plt.ylabel(r"Univariate score ($-Log(p_{value})$)")
plt.show()

# # # # Compare with SVMs

from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler

# # # WITHOUT UNIVARIATE FEATURE SELECTION
clf_without_univariate = make_pipeline(
    MinMaxScaler(),
    LinearSVC(dual='auto')
)
clf_without_univariate.fit(X_train, y_train)

print(
    "Classification accuracy without selecting features: {:.3f}".format(
        clf_without_univariate.score(X_test, y_test)
    )
)

svm_weights = np.abs(clf_without_univariate[-1].coef_).sum(axis=0)
svm_weights /= svm_weights.sum()

# # # # With univariate feature selection

clf_with_univariate = make_pipeline(
    SelectKBest(f_classif, k= 4),
    MinMaxScaler(),
    LinearSVC(dual='auto')
)

clf_with_univariate.fit(X_train, y_train)

print(
    "Classification accuracy after univariate feature selection: {:.3f}".format(
        clf_with_univariate.score(X_test, y_test)
    )
)

svm_weights_with_univariate = np.abs(clf_with_univariate[-1].coef_).sum(axis=0)
svm_weights_with_univariate /= svm_weights_with_univariate.sum()


# COmparison plot
plt.bar(
    X_indices - 0.45, scores, width=0.2, label=r"Univariate score ($-Log(p_{value})$)"
)

plt.bar(X_indices - 0.25, svm_weights, width=0.2, label="SVM weight")

plt.bar(
    X_indices[selector.get_support()] - 0.05,
    svm_weights_with_univariate,
    width=0.2,
    label="SVM weights after selection",
)

plt.title("Comparing feature selection")
plt.xlabel("Feature number")
plt.yticks(())
plt.axis("tight")
plt.legend(loc="upper right")
plt.show()

"""Without univariate feature selection, the SVM assigns a large weight to the first 4 original significant features, 
but also selects many of the non-informative features. Applying univariate feature selection before the SVM increases 
the SVM weight attributed to the significant features, and will thus improve classification. """