"""
https://scikit-learn.org/stable/auto_examples/feature_selection/plot_feature_selection_pipeline.html#sphx-glr-auto-examples-feature-selection-plot-feature-selection-pipeline-py

this is basically an example for how to make pipeline when we have feature selection as separate steps in the model training.

& feature selection can be easily integrated within machine learning pipeline

# We should never extract the feature_selection for all the data, we should only be using in Training dataset.
# this problem can be handled using sklearn pipeline
"""

# make_classification: A toy dataset for the classification problem.
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# this is for binary classification problem
X, y = make_classification(
    n_samples=20,
    n_features=4,
    n_redundant=0,
    n_classes=2,
    n_clusters_per_class=2,
    random_state=42
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    random_state=42
)

from sklearn.feature_selection import SelectKBest, f_classif

# SelectKBest: Select features according to the k highest scores.
# f_classif: Compute the ANOVA F-value for the provided sample. It retursn F-value and p_value
# F_value : variance_of_first_dataset/ variance_of_second_dataset

from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

anova_filter = SelectKBest(f_classif, k=3)  # It selects the best 3 features using f_score
classification_model = LinearSVC(dual="auto")

# make_pipeline takes steps here anova_filter and classification_model are the steps for make_pipeline
anova_svm = make_pipeline(
    anova_filter,
    classification_model
)

print("Steps for anova_svm:")
print(anova_svm)

# Training
anova_svm.fit(X_train, y_train)

from sklearn.metrics import classification_report
y_pred = anova_svm.predict(X_test)
print(classification_report(y_test, y_pred))

# Knowing the parameters of the classifier
print(anova_svm[-1].coef_)

# However, we do not know which features were selected from the original dataset. We could proceed by several
# manners. Here, we will invert the transformation of these coefficients to get information about the original space.
print("features with non-zero coefficients are selected.")
print(anova_svm[:-1].inverse_transform(anova_svm[-1].coef_))