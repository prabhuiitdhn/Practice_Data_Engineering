"""

https://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_digits.html#sphx-glr-auto-examples-feature-selection-plot-rfe-digits-py

"""
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.feature_selection import RFE
from sklearn.svm import SVC

# load the dataset

digits = load_digits()
print("Data set description:")
print(digits.DESCR)

X = digits.images.reshape(len(digits.images), -1)
y= digits.target

# Create RFE object and rank each pixel

svc = SVC(kernel="linear", C=1) # C: regularisation term
rfe = RFE(
    estimator=svc,
    n_features_to_select=1,
    step=1
)

rfe.fit(X, y)
ranking = rfe.ranking_.reshape(digits.images[0].shape)
# print(rfe.predict(X_test)) # for this we need to split the data using sklearn.model_selection import train_test_split

# Plot pixel ranking
plt.matshow(ranking, cmap=plt.cm.Blues)
plt.colorbar()
plt.title("Ranking of pixels with RFE")
plt.show()
