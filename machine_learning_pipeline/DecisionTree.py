"""
https://scikit-learn.org/stable/modules/tree.html

Decision tree are non-parametric supervised learning method used for classification and regression

goal is create the model which predicts the value of target variable by learning simple decsion rules inferred from data
A tree can be seen as a piecewise constant approximation.

The deeper the tree, the more complex the decision rules and the fitter the model.

advantages:
1. Simple to understand and to interpret. Trees can be visualized.
2. Requires little data preparation. Other techniques often require data normalization,
    dummy variables need to be created and blank values to be removed. Note however that this module does not support missing values.
3. The cost of using the tree (i.e., predicting data) is logarithmic in the number of data points used to train the tree.
4. Able to handle both numerical and categorical data
5. Able to handle multi-output problems
6. Uses a white box model. If a given situation is observable in a model, the explanation for the condition is easily explained by boolean logic.
7. possible to validate a model using statistical tests. That makes it possible to account for the reliability of the model.
8. Performs well even if its assumptions are somewhat violated by the true model from which the data were generated.

Disadvantages:
1. Decision-tree learners can create over-complex trees that do not generalize the data well. This is called overfitting.
  Mechanisms such as pruning,
  setting the minimum number of samples required at a leaf node
  or setting the maximum depth of the tree are necessary to avoid this problem.
2. Decision trees can be unstable because small variations in the data might result in a completely different tree being generated. This problem is mitigated by using decision trees within an ensemble.

3. Predictions of decision trees are neither smooth nor continuous, but piecewise constant approximations as seen in the above figure. Therefore, they are not good at extrapolation.

4. The problem of learning an optimal decision tree is known to be NP-complete,
   Consequently, practical decision-tree learning algorithms are based on heuristic algorithms such as the greedy algorithm where locally optimal decisions are made at each node. Such algorithms cannot guarantee to return the globally optimal decision tree.
   This can be mitigated by training multiple trees in an ensemble learner, where the features and samples are randomly sampled with replacement.

5. There are concepts that are hard to learn because decision trees do not express them easily, such as XOR, parity or multiplexer problems.

6. Decision tree learners create biased trees if some classes dominate. It is therefore recommended to balance the dataset prior to fitting with the decision tree.


"""
# DecisionTreeClassifier is a class capable of performing multi-class classification on a dataset.

from sklearn import tree

# "tree.BaseDecisionTree": Base class for decision trees
# "tree.DecisionTreeClassifier":A decision tree classifier.
# "tree.DecisionTreeRegressor":A decision tree regressor.
# "tree.ExtraTreeClassifier":     Extra-trees differ from classic decision trees in the way they are built.
#     When looking for the best split to separate the samples of a node into two
#     groups, random splits are drawn for each of the `max_features` randomly
#     selected features and the best split among those is chosen. When
#     `max_features` is set 1, this amounts to building a totally random
#     decision tree.
# tree.export_graphviz: Export a decision tree in DOT format.
# tree.plot_tree: Plot a decision tree.
# tree.export_text: Build a text report showing the rules of a decision tree.


# tree.DecisionTreeClassifier() : take two array in needed x: shape[n_sample, n_features], y:shape[n_labels]
# parameters:
# criterion: The function to measure the quality of a split.{"gini", "entropy", "log_loss"}
# splitter:The strategy used to choose the split at each node.{"best", "random"}
# max_depth: The maximum depth of the tree
# min_samples_split: The minimum number of samples required to split an internal node:
# min_samples_leaf: The minimum number of samples required to be at a leaf node.
# min_weight_fraction_leaf: The minimum weighted fraction of the sum total of weights
# max_features: The number of features to consider when looking for the best split
# random_state: Controls the randomness of the max_features
# max_leaf_nodes: Grow a tree with ``max_leaf_nodes`` in best-first fashion.
# min_impurity_decrease: A node will be split if this split induces a decrease of the impurity >= to this value.
# class_weight: Weights associated with classes in the form ``{class_label: weight}``.
# ccp_alpha:cost complexity pruning;The subtree with the largest cost complexity that is < ``ccp_alpha`` will be chosen.
# Simple Example:
X = [[0, 0], [1, 1]]
y = [0, 1]
clf = tree.DecisionTreeClassifier()

clf = clf.fit(X, y)
clf.predict([[2, 2]])
print(clf.predict([[2, 2]]))
# print(tree.plot_tree(clf))
# print(tree.export_graphviz(clf))


# # The tree can be exported in graph form using export_graphviz.
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()
X, y = iris.data, iris.target
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)
print(tree.plot_tree(clf))

import graphviz

dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("iris")
dot_data = tree.export_graphviz(
    clf, out_file=None,
    feature_names=iris.feature_names,
    filled=True,
    rounded=True,
    special_characters=True
)

graph = graphviz.Source(dot_data)
print(graph)

# the tree can also be exported in textual format with function export_text
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
iris = load_iris()
decision_tree = DecisionTreeClassifier(random_state=0, max_depth=2)
decision_tree = decision_tree.fit(iris.data, iris.target)
r = export_text(
    decision_tree,
    feature_names=iris['feature_names']
)
print(r)


# # regression.
# Decision tree can also be used for regression problems, using DecisionTreeRegressor class.
from sklearn import tree
x = [[0, 0], [2, 2]]
y = [0.5, 2.5]
clf = tree.DecisionTreeRegressor()
clf = clf.fit(x, y)
clf.predict([1,1])

# # A multi-output problems
"""
A multi-output problem is a supervised learning problem with several outputs to predict, that is when Y is a 2d array of shape (n_samples, n_outputs).

When there is no correlation between the outputs, a very simple way to solve this kind of problem is 
to build n independent models, i.e. one for each output, and then to use those models to independently predict each one-
of the n outputs. However,because it is likely that the output values related to the same input are themselves correlated,
an often better way is to build a single model capable of predicting simultaneously all n outputs.

& decision trees can be used for multi-output problems. for this, following changes can be required.

to make the multi-output problems, following changes needs to be done.
1. Store n output values in leaves instead of 1
2. Use splitting criteria that compute the average reduction across all n outputs.
"""

