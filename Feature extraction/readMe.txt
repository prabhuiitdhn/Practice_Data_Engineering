
Feature selection is a process of selecting the subset of features from the dataset which are most relevant to the target variable.
This can be done for a variety of reasons, such as to improve the performance of a machine learning model,to reduce the computational complexity of the model, or to make the model more interpretable.


Two main types of feature selection:
1. Filter methods: These methods select features based on their statistical properties, such as their correlation with the target variable or their variance.
        - Missing value ration: Identifying features with a high proportion of missing values can adversely affect model performance and lead to biased results.
        - Information gain: Quantifying a feature's ability to reduce uncertainty about the target variable, making it an essential tool for classification tasks by measuring the discriminatory power of each attribute.
        - Low Variance Filter: Isolating/removing features with minimal variation across the dataset, as they often contribute little to modeling and can be removed without significant information loss.
2. Wrapper methods: These methods select features by iteratively searching through the space of possible feature subsets and evaluating the performance of the model on a held-out validation set.
        - Forward Feature Selection: Iteratively adding features and evaluating model performance to identify the best subset that maximizes predictive power.
        - Backward Feature Selection: The reverse of forward selection, this method starts with all features and eliminates the least important ones until optimal performance is achieved.

3. Embedded methods: Embedded methods perform feature selection during the training process of a machine learning algorithm.
        - Regularisation: Regularization techniques like LASSO (L1 regularization) and Ridge Regression (L2 regularization) can lead to automatic feature selection by shrinking coefficient towards 0
        - Random forest importance: Tree-based algorithms like Random Forests and Gradient Boosting often rank features based on their importance, allowing for feature selection.
4. feature importance: Many machine learning algorithms provide feature importance scores that indicate how much each feature contributes to the model's predictions.
5. Dimensionality reduction: It actually reduces the dimension of Original features and It selects the feature based on variance. Highest variance will be selected first
6. Hybrid methods: Combining aspects of filter and wrapper methods for a balanced approach
    - Filter and Wrapper Methods: Employing filter techniques to preselect a subset of features, then fine-tuning using a wrapper method to maximize model performance.
    - sophisticated algorithms like the 'Boruta algorithm' and exhaustive feature selection techniques can be employed for exhaustive search and comprehensive feature evaluation.


Most common filter technique for feature selection:
It depends of Machine learning task, Filter methods are faster and easier to implement.
1. Univariate selection: This method selects features one at a time, and it is based on their correlation with the target variable.
                        - Chi-Square test
                        - Mutual information
                        - Variance thresholds
                        - Co-relation

2. Recursive feature elimination: This method starts with all the features and then eliminates features one at a time, based on their importance to the model.
3. Chi-Sqaured test: This method selects features based on their statistical significance, as measured by the chi-squared test.


Most common wrapper methods:
These methods can be computationally expensive but tend to find optimal subsets for specific algorithms.
1. Forward selection: This method starts with an empty set of features and then adds features one at a time, based on their improvement to the model's performance.
2. Backward elimination: This method starts with all the features and then eliminates features one at a time, based on their negative impact on the model's performance.
3. Genetic algorithm: This method uses a genetic algorithm to search for the best feature subset.

Benefits of feature selection:
1. It can improve the performance of Machine Learning Models
2. Reduce the computational complexity of ML model


One technique is ANOVA [Analysis of variance]: It assumes certain assumptions like normality of data, Homogenity of variance.
- It is a statistical technique commonly used in machine learning and statistics to analyze the variance between group means and assess whether there are statistically significant differences among the means of multiple groups or treatments.
- ANOVA helps determine if the variation between groups is larger than the variation within groups, suggesting that the groups are not all the same.

It can be used for
1. feature selection:
    - ANOVA can be used to assess the relationship between individual features and the target variable in a classification or regression problem.
    - It helps identify features that have a significant impact on the target variable.
    - Features with higher F-values (ANOVA statistic) are considered more important.
    - F_value : variance_of_first_dataset/ variance_of_second_dataset

2. Comparing multiple models:
    - ANOVA can be used to compare the performance of different machine learning models or algorithms.
    - It can help determine whether the differences in predictive performance among the models are statistically significant.

3. Hyper parameter tuning:
    - When comparing the performance of models with different hyperparameters, ANOVA can help identify whether a particular set of hyperparameters leads to significantly better results.

4. Experimental Analysis:
    - In experimental design, ANOVA can be used to analyze the effects of different factors or treatments on an outcome.


A random forest classifier will be fitted to compute the feature importance.
RandomForest techniques: It is ensemble technique to 'solve the classification, regression problem, and also used for
    feature selection in a nutshell' in machine learning which works through a many decision-Trees in the forest,
    and the trees are in random forest are uncorrelated [uses bagging and feature randomness to build individual Tree]
    so that each individual trees will have their own features to work with which will have higher score, but at the end,
    each tress produces result and using Majority voting/averaging.techniques, score can be decided.

from sklearn.ensemble import RandomForestClassifier, BaseEnsemble

"BaseEnsemble": Base class for all ensemble classes.
"RandomForestClassifier": A random forest is a meta estimator that fits a number of decision tree classifiers on
                          various sub-samples of the dataset and uses averaging to improve the predictive accuracy and
                          control over-fitting.

"RandomForestRegressor":
"RandomTreesEmbedding": An unsupervised transformation of a dataset to a high-dimensional sparse representation.
"ExtraTreesClassifier": a meta estimator that fits a number of randomized decision trees (a.k.a. extra-trees) on
                        various sub-samples of the dataset and uses averaging to improve the predictive accuracy
                        and control over-fitting.
"ExtraTreesRegressor":
"BaggingClassifier": It is a meta estimator which fits a base classifier each on random subsets of datasets,
and aggregate the result using voting/averaging. It basically reduced the variance of black box estimator.
When random subsets of the dataset are drawn as random subsets of the samples this algorithm is known as Pasting [1]_.
If samples are drawn with replacement, then the method is known as Bagging [2]_.
When random subsets of the datasetare drawn as random subsets of the features, then the method is known as Random Subspaces [3]_.
Finally, when base estimators are built on subsets of both samples and features, then the method is known as Random Patches[4]_.

# "BaggingRegressor",

# "IsolationForest":  The IsolationForest 'isolates' observations by randomly selecting a feature and then randomly
# selecting a split value between the maximum and minimum values of the selected feature.
# "GradientBoostingClassifier",:     This algorithm builds an additive model in a forward stage-wise fashion; it
# allows for the optimization of arbitrary differentiable loss functions. In each stage ``n_classes_`` regression
# trees are fit on the negative gradient of the loss function, e.g. binary or multiclass log loss. Binary
# classification is a special case where only a single regression tree is induced.

# "GradientBoostingRegressor",

# "AdaBoostClassifier":     An AdaBoost [1] classifier is a meta-estimator that begins by fitting a
#     classifier on the original dataset and then fits additional copies of the
#     classifier on the same dataset but where the weights of incorrectly
#     classified instances are adjusted such that subsequent classifiers focus
#     more on difficult cases.

# "AdaBoostRegressor",

# "VotingClassifier": Soft Voting/Majority Rule classifier for unfitted estimators.

# "VotingRegressor",

# "StackingClassifier": Stacked generalization consists in stacking the output of individual estimator and use a
# classifier to compute the final prediction. Stacking allows to use the strength of each individual estimator by
# using their output as input of a final estimator.

# "StackingRegressor",

# "HistGradientBoostingClassifier": This estimator has native support for missing values (NaNs). During training,
# the tree grower learns at each split point whether samples with missing values should go to the left or right
# child, based on the  potential gain. When predicting, samples with missing values are assigned to the left or right
# child consequently. If no missing values were encountered for a given feature during training, then samples with
# missing values are mapped to whichever child has the most samples.

# "HistGradientBoostingRegressor",


