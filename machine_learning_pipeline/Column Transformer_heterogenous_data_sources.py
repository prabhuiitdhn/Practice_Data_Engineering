"""
https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer.html#sphx-glr-auto-examples-compose-plot-column-transformer-py

Sometime dataset contains different components that requires different feature extraction and preprocessing pipelines
i.e.

1. your dataset consists of heterogeneous data types (e.g. raster images and text captions),
2. your dataset is stored in a pandas.DataFrame and different columns require different processing pipelines.
"""

# This example demonstrates how to use ColumnTransformer on a dataset containing different types of features.
# The choice of features is not particularly helpful, but serves to illustrate the technique.

import numpy as np
from sklearn.compose import ColumnTransformer
# sklearn.compose: Meta-estimators for building composite models with transformers
# ColumnTransformer: Applies transformers to columns of an array or pandas DataFrame.

from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD

# sklearn.decomposition: `sklearn.decomposition` module includes matrix decomposition algorithms,
# including among others PCA, NMF or ICA. Most of the algorithms of
# this module can be regarded as dimensionality reduction techniques.

# TruncatedSVD: This transformer performs linear dimensionality reduction by means of truncated SVD.

from sklearn.feature_extraction import DictVectorizer

# sklearn.feature_extraction` module deals with feature extraction from raw data. It currently includes methods to
# extract features from text and images. DictVectorizer: Transforms lists of feature-value mappings to vectors.

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

categories = ["sci.med", "sci.space"]  # Only these categories of dataset we are loading from new groups
X_train, y_train = fetch_20newsgroups(
    random_state=1,
    subset="train",
    categories=categories,
    remove=("footer", "quotes"),
    return_X_y=True
)

X_test, y_test = fetch_20newsgroups(
    random_state=1,
    subset="test",
    categories=categories,
    remove=("footer", "quotes"),
    return_X_y=True
)

print("This shows the body of the new post:")
print(X_train[0])


# Creating transformers

def subject_body_extractor(posts):
    # construct object dtype array with two columns
    # first column = 'subject' and second column = 'body'
    features = np.empty(shape=(len(posts), 2), dtype=object)
    for i, text in enumerate(posts):
        # temporary variable `_` stores '\n\n'
        headers, _, body = text.partition("\n\n")
        # store body text in second column
        features[i, 1] = body

        prefix = "Subject:"
        sub = ""
        # save text after 'Subject:' in first column
        for line in headers.split("\n"):
            if line.startswith(prefix):
                sub = line[len(prefix):]
                break
        features[i, 0] = sub

    return features


# A FunctionTransformer forwards its X (and optionally y) arguments to a user-defined function or function object and
# returns the result of this function.


subject_body_transformer = FunctionTransformer(subject_body_extractor)


# Create a transformer that extracts the length of the text and the number of sentences.
def text_stats(posts):
    return [{"length": len(text), "num_sentences": text.count(".")} for text in posts]


text_stats_transformer = FunctionTransformer(text_stats)

# Creating pipeline


pipeline = Pipeline(
    [
        ("Subject Body Extractor", subject_body_transformer),
        ("union",
         ColumnTransformer([
             ("subject", TfidfVectorizer(min_df=50), 0),
             (
                 "Body_bow", Pipeline([
                     ("tfidf", TfidfVectorizer()),
                     ("best", TruncatedSVD(n_components=50))
                 ]
                 ),
                 1
             ),
             # Pipeline for pulling text stats from post's body
             (
                 "body_stats", Pipeline([
                     ("stats", text_stats_transformer),
                     ("vect", DictVectorizer())
                 ]
                 ), 1
             )
         ],
             # weight above ColumnTransformer features
             transformer_weights={"subject": 0.8, "body_bow": 0.5, "body_stats": 1.0},
         ),
         ),
        # Use a SVC classifier on the combined features
        ("svc", LinearSVC(dual=False))
    ]
    , verbose=True
)

print("Check the pipeline:")
print(pipeline)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print("Classification report:\n\n{}".format(classification_report(y_test, y_pred)))