# Feature Selection

Feature selection is a process of selecting the subset of features from the dataset which are most relevant to the target variable. This can be done for a variety of reasons, such as to improve the performance of a machine learning model, to reduce the computational complexity of the model, or to make the model more interpretable.

---

## Types of Feature Selection

### 1. Filter Methods

These methods select features based on their statistical properties, such as their correlation with the target variable or their variance.

- **Missing Value Ratio**: Identifying features with a high proportion of missing values can adversely affect model performance and lead to biased results.
- **Information Gain**: Quantifying a feature's ability to reduce uncertainty about the target variable, making it an essential tool for classification tasks by measuring the discriminatory power of each attribute.
- **Low Variance Filter**: Isolating/removing features with minimal variation across the dataset, as they often contribute little to modeling and can be removed without significant information loss.

### 2. Wrapper Methods

These methods select features by iteratively searching through the space of possible feature subsets and evaluating the performance of the model on a held-out validation set.

- **Forward Feature Selection**: Iteratively adding features and evaluating model performance to identify the best subset that maximizes predictive power.
- **Backward Feature Selection**: The reverse of forward selection, this method starts with all features and eliminates the least important ones until optimal performance is achieved.

### 3. Embedded Methods

Embedded methods perform feature selection during the training process of a machine learning algorithm.

- **Regularisation**: Regularization techniques like LASSO (L1 regularization) and Ridge Regression (L2 regularization) can lead to automatic feature selection by shrinking coefficients towards 0.
- **Random Forest Importance**: Tree-based algorithms like Random Forests and Gradient Boosting often rank features based on their importance, allowing for feature selection.

### 4. Feature Importance

Many machine learning algorithms provide feature importance scores that indicate how much each feature contributes to the model's predictions.

### 5. Dimensionality Reduction

It actually reduces the dimension of original features and selects the feature based on variance. Highest variance will be selected first.

### 6. Hybrid Methods

Combining aspects of filter and wrapper methods for a balanced approach.

- **Filter and Wrapper Methods**: Employing filter techniques to preselect a subset of features, then fine-tuning using a wrapper method to maximize model performance.
- Sophisticated algorithms like the **Boruta algorithm** and exhaustive feature selection techniques can be employed for exhaustive search and comprehensive feature evaluation.

---

## Most Common Filter Techniques

> It depends on the Machine Learning task. Filter methods are faster and easier to implement.

### 1. Univariate Selection

Selects features one at a time based on their correlation with the target variable.

- Chi-Square test
- Mutual information
- Variance thresholds
- Correlation

### 2. Recursive Feature Elimination

Starts with all the features and then eliminates features one at a time, based on their importance to the model.

### 3. Chi-Squared Test

Selects features based on their statistical significance, as measured by the chi-squared test.

---

## Most Common Wrapper Methods

> These methods can be computationally expensive but tend to find optimal subsets for specific algorithms.

1. **Forward Selection**: Starts with an empty set of features and adds features one at a time, based on their improvement to the model's performance.
2. **Backward Elimination**: Starts with all the features and eliminates features one at a time, based on their negative impact on the model's performance.
3. **Genetic Algorithm**: Uses a genetic algorithm to search for the best feature subset.

---

## Benefits of Feature Selection

1. It can improve the performance of Machine Learning Models.
2. Reduce the computational complexity of ML model.

---

## ANOVA (Analysis of Variance)

ANOVA assumes certain assumptions like **normality of data** and **homogeneity of variance**.

It is a statistical technique commonly used in machine learning and statistics to analyze the variance between group means and assess whether there are statistically significant differences among the means of multiple groups or treatments. ANOVA helps determine if the variation between groups is larger than the variation within groups, suggesting that the groups are not all the same.

### ANOVA Use Cases

#### 1. Feature Selection

- ANOVA can be used to assess the relationship between individual features and the target variable in a classification or regression problem.
- It helps identify features that have a significant impact on the target variable.
- Features with higher F-values (ANOVA statistic) are considered more important.
- `F_value = variance_of_first_dataset / variance_of_second_dataset`

#### 2. Comparing Multiple Models

- ANOVA can be used to compare the performance of different machine learning models or algorithms.
- It can help determine whether the differences in predictive performance among the models are statistically significant.

#### 3. Hyperparameter Tuning

- When comparing the performance of models with different hyperparameters, ANOVA can help identify whether a particular set of hyperparameters leads to significantly better results.

#### 4. Experimental Analysis

- In experimental design, ANOVA can be used to analyze the effects of different factors or treatments on an outcome.

---

## Random Forest Classifier for Feature Importance

A random forest classifier will be fitted to compute the feature importance.

**RandomForest** is an ensemble technique to solve classification, regression problems, and also used for feature selection. It works through many decision trees in the forest, and the trees are uncorrelated (uses bagging and feature randomness to build individual trees) so that each individual tree will have their own features to work with which will have higher score. At the end, each tree produces a result and using **Majority Voting/Averaging** techniques, the final score can be decided.

### Scikit-learn Ensemble Classes

```python
from sklearn.ensemble import RandomForestClassifier, BaseEnsemble
```

| Class | Description |
|-------|-------------|
| `BaseEnsemble` | Base class for all ensemble classes |
| `RandomForestClassifier` | A meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting |
| `RandomForestRegressor` | Random forest for regression tasks |
| `RandomTreesEmbedding` | An unsupervised transformation of a dataset to a high-dimensional sparse representation |
| `ExtraTreesClassifier` | A meta estimator that fits a number of randomized decision trees (extra-trees) on various sub-samples and uses averaging to improve accuracy and control over-fitting |
| `ExtraTreesRegressor` | Extra trees for regression tasks |
| `BaggingClassifier` | A meta estimator which fits a base classifier each on random subsets of datasets, and aggregates the result using voting/averaging. Reduces the variance of a black box estimator |
| `IsolationForest` | Isolates observations by randomly selecting a feature and then randomly selecting a split value between the max and min values of the selected feature |
| `GradientBoostingClassifier` | Builds an additive model in a forward stage-wise fashion; allows optimization of arbitrary differentiable loss functions |
| `GradientBoostingRegressor` | Gradient boosting for regression tasks |
| `AdaBoostClassifier` | A meta-estimator that fits a classifier on the original dataset and then fits additional copies where weights of incorrectly classified instances are adjusted |
| `AdaBoostRegressor` | AdaBoost for regression tasks |
| `VotingClassifier` | Soft Voting/Majority Rule classifier for unfitted estimators |
| `VotingRegressor` | Voting for regression tasks |
| `StackingClassifier` | Stacks the output of individual estimators and uses a classifier to compute the final prediction |
| `StackingRegressor` | Stacking for regression tasks |
| `HistGradientBoostingClassifier` | Has native support for missing values (NaNs). During training, the tree grower learns at each split point whether samples with missing values should go to the left or right child |
| `HistGradientBoostingRegressor` | Histogram-based gradient boosting for regression |

### Bagging Variants

| Variant | Description |
|---------|-------------|
| **Pasting** | Random subsets of the dataset are drawn as random subsets of the samples |
| **Bagging** | Samples are drawn with replacement |
| **Random Subspaces** | Random subsets of the dataset are drawn as random subsets of the features |
| **Random Patches** | Base estimators are built on subsets of both samples and features |
