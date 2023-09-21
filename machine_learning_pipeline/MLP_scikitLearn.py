"""
https://scikit-learn.org/stable/modules/neural_networks_supervised.html


"""

# Classification
from sklearn.neural_network import MLPClassifier, BernoulliRBM, MLPRegressor

# "BernoulliRBM": Bernoulli Restricted Boltzmann Machine (RBM).A Restricted Boltzmann Machine with binary visible units
#                 and binary hidden units. Parameters are estimated using Stochastic Maximum
#                 Likelihood (SML), also known as Persistent Contrastive Divergence (PCD)
# "MLPClassifier":This model optimizes the log-loss function using LBFGS or stochastic gradient descent.
# "MLPRegressor": This model optimizes the squared error using LBFGS or stochastic gradient descent

X = [[0, 0], [1, 1]]
y = [0, 1]

clf = MLPClassifier(
    hidden_layer_sizes=(5, 2),
    activation="relu",
    solver="lbfgs",
    alpha=1e-05,
    random_state=1,
    batch_size="auto",
    learning_rate="constant",
    learning_rate_init=0.001,
)

print("Training the model:")
clf.fit(X, y)  # Train the model
print("Training done.")
print("Model Description:")
print("parameter used for Training the model:", clf.get_params())
print("Score on this model:", clf.score(X, y))
print("Solver is being used for Training:", clf.solver)
print("metadata Routing:", clf.get_metadata_routing())
# get_metadata_routing: This guide demonstrates how metadata such as sample_weight can be routed and passed
# along to estimators, scorers, and CV splitters through meta-estimators such as Pipeline and GridSearchCV .
# In order to pass metadata to a method such as fit or score , the object consuming the metadata, must request it.
print("No of input features:", clf.n_features_in_)
print("Weight matrix:", clf.coefs_) # clf.coefs_ contains the weight matrices that constitute the model parameters:
print(clf.predict([[2, 2], [-1, -2]]))

print("Shape of coefficient:")
for coef in clf.coefs_:
    print(coef.shape)

# MLPClassifier supports multi-class classification by applying Softmax as the output function.

