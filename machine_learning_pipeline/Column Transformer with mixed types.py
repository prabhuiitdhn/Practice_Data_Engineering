"""
https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html#sphx-glr-auto-examples-compose-plot-column-transformer-mixed-types-py

This is basically for applying different preprocessing and feature extraction pipelines to different subset of features

When the dataset contains numeric features and categorical one, then in this case, numeric features have to be scaled and categorical features has to be one-hot encoded.

in this example:
    the numeric data is standard-scaled after mean-imputation.
    The categorical data is one-hot encoded via OneHotEncoder, which creates a new category for missing values.
    We further reduce the dimensionality by selecting categories using a chi-squared test.

Openml datasets: https://www.openml.org/search?type=data&status=active
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# RandomizedSearchCV: Randomized search on hyperparameters.
# fetch_openml: String identifier of the dataset. Note that OpenML can have multiple datasets with the same name. It
# can find using string name or ID
np.random.seed(0)

X, y = fetch_openml(
    "titanic",
    version=1,
    as_frame=True,
    return_X_y=True,
    parser="pandas"
)

# Use ColumnTransformer by selecting column by names
# we will train out classifier with numeric features: [age, and fare]
# categorical features
# embarked: categories encoded as strings {'C', 'S', 'Q'};
# sex: categories encoded as strings {'female', 'male'};
# pclass: ordinal integers {1, 2, 3}. # IT CAN BE TREATED AS NUMBERIC or CATEGORICAL

numeric_features = ["age", "fare"]

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_features = ["embarked", "sex", "pclass"]
categorical_transformer = Pipeline(
    steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ("selector", SelectPercentile(chi2, percentile=50))
    ]
)

preprocessor_categorical_numeric = ColumnTransformer(
    transformers=[
        ("numeric_transformer", numeric_transformer, numeric_features),
        ("categorical_features", categorical_transformer, categorical_features)
    ]
)

# append classifier for preprocessing pipeline

classification_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor_categorical_numeric),
        ("Classifier", LogisticRegression())
    ]
)

# SPlitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print("Training data info:", X_train.info())
classification_model.fit(X_train, y_train)

print("Score")
print(classification_model.score(X_test, y_test))

print("Pipeline")
print(classification_model)


# # # # # # # # # # # # # when the data have to be handled as column data type, If we want some columns to be
# considered as category. we will have to convert them into categorical columns

from sklearn.compose import make_column_selector as selector

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, selector(dtype_exclude="category")), # Excluding the dtype
        ("cat", categorical_transformer, selector(dtype_include="category")), # including the dtype
    ]
)
clf_dtype = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", LogisticRegression())]
)


clf_dtype.fit(X_train, y_train)
print("model score: %.3f" % clf_dtype.score(X_test, y_test))
print(clf_dtype)
print(selector(dtype_exclude="category")(X_train))
print(selector(dtype_include="category")(X_train))