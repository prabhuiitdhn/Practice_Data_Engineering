"""
multi-output estimator to complete images.
The goal is to predict the lower half of a face given its upper half.

First column shows the True faces.
nex columns show how that algorithms i.e randomized Tree, K nearest neighbors, Linear regression, ridgeression
complete the lower half of those faces.


"""

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_olivetti_faces
# Olivetti faces data-set from AT&T
from sklearn.ensemble import ExtraTreesRegressor
# ExtraTreesRegressor: This class implements a meta estimator that fits a number of
#     randomized decision trees (a.k.a. extra-trees) on various sub-samples
#     of the dataset and uses averaging to improve the predictive accuracy
#     and control over-fitting.

from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils.validation import check_random_state

data, targets = fetch_olivetti_faces(return_X_y=True)
# return Dictionary-like object, with the following attributes.
# It return the image of 64X64 pixels with Labels associated to each face image from 0-39
train = data[targets < 30]  # this is training the model
test = data[targets >= 30]  # this is for testing the model

n_faces = 5
rng = check_random_state(4)

face_ids = rng.randint(
    test.shape[0],
    size=(n_faces,)
)

test = test[face_ids, :]

n_pixels = data.shape[1]
X_train = train[:, : (n_pixels + 1) // 2]  # Upper half of the face.
y_train = train[:, n_pixels // 2:]  # lower half of the face.
X_test = test[:, : (n_pixels + 1) // 2]
y_test = test[:, n_pixels // 2:]

# Estimators

Estimators = {
    'Extra Trees:': ExtraTreesRegressor(
        n_estimators=10,
        max_features=32,
        random_state=0
    ),
    "k_nn": KNeighborsRegressor(),
    "Linear regression": LinearRegression(),
    "Ridge CV": RidgeCV()
}

y_test_predict = dict()
for name, estimator in Estimators.items():
    estimator.fit(X_train, y_train)
    y_test_predict[name] = estimator.predict(X_test)

# plot the completed faces

image_shape = (64, 64)
n_cols = 1 + len(Estimators)
plt.figure(figsize=(2 * n_cols, 2.26 * n_faces))

plt.suptitle("face completion with output estimators", size=16)

for i in range(n_faces):
    true_face = np.hstack((X_test[i], y_test[i]))
    if i:
        sub = plt.subplot(n_faces, n_cols, 1 * n_cols + 1)
    else:
        sub = plt.subplot(n_faces, n_cols, i * n_cols + 1, title="True faces")

    sub.axis("off")
    sub.imshow(
        true_face.reshape(image_shape),
        cmap=plt.cm.gray,
        interpolation="nearest"
    )

    for j, est in enumerate(sorted(Estimators)):
        completed_faces = np.hstack((X_test[i], y_test_predict[est][i]))
        if i:
            sub = plt.subplot(n_faces, n_cols, i*n_cols+2+j)
        else:
            sub = plt.subplot(n_faces, n_cols, i * n_cols + 2 + j, title=est)
        sub.axis("off")
        sub.imshow(
            completed_faces.reshape(image_shape),
            cmap=plt.cm.gray,
            interpolation="nearest"
        )

plt.show()