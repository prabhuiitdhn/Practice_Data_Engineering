"""
https://scikit-learn.org/stable/auto_examples/feature_selection/plot_f_test_vs_mi.html#sphx-glr-auto-examples-feature-selection-plot-f-test-vs-mi-py

The F-test statistic is a statistical measure used to compare the variances of two or more groups or samples. It
assesses whether the means of the groups are significantly different by comparing the variability within each group
to the variability between the groups. The F-test is commonly used in analysis of variance (ANOVA) and regression
analysis to determine if the variation in the data can be attributed to different factors.

1. Null Hypothesis: The null hypothesis states that there is no significant difference between the means of the
groups being compared.

2. Alternative Hypothesis: The alternative hypothesis states that there is a significant difference between at least
one pair of means

3. Calculating Variability:The F-test statistic is calculated as the ratio of the between-group variability to the
within-group variability.

4. F-statistic calculation: The formula for the F-test statistic is: F = (Between-Group Variability) / (Within-Group
Variability)

5. Critical value and p-value: The F statistic is compared to a critical value from the F-distribution to determine
statistical significance. Alternatively, the p-value associated with the F statistic can be calculated,
which indicates the probability of obtaining the observed results if the null hypothesis is true.

6. Decision: If the p-value is lower than a chosen significance level (e.g., 0.05), the null hypothesis is rejected
in favor of the alternative hypothesis, indicating that there is a significant difference between the group means. If
the p-value is not lower than the significance level, the null hypothesis is not rejected, suggesting that there is
not enough evidence to conclude a significant difference.

Assume x1, x2, x3 are three features are available and all are distributed uniformly over [0,1] range and target
depends on it as follows

y = x1 + sin(6*pi*x2) = 0.1 * N(0,1) # in this case we can say that x3 is so irrelevant in this case.

F-test captures only linear dependency, it rates x_1 as the most discriminative feature. On
the other hand, mutual information can capture any kind of dependency between variables and it rates x_2 as the most
discriminative feature, which probably agrees better with our intuitive perception for this example.


"""

# Examples shows  the difference between F2 score and mutual information.

import matplotlib.pyplot as plt
import numpy as np

from sklearn.feature_selection import f_regression, mutual_info_regression

# f_regression: Univariate linear regression tests returning F-statistic and p-values.

# Mutual information: (MI) [1]_ between two random variables is a non-negative value, which measures the dependency
                    # between the variables. It is  equal to zero if and only if two random variables are independent, and higher values
                    # mean higher dependency.

np.random.seed(0)
X = np.random.rand(1000, 3)
# y = x1 + sin(6* pi* x2) + 0.1 * N(0, 1)
y = X[:, 0] + np.sin(6 * np.pi * X[:, 1]) + 0.1 * np.random.randn(1000)


f_test, p_value = f_regression(X, y) # It returns the f_test, p_value for all 3 features of X
# 4. F-statistic calculation: The formula for the F-test statistic is: F = (Between-Group Variability) / (Within-Group Variability)

f_test /= np.max(f_test) # normalising the f_test values

mi = mutual_info_regression(X, y) # It return the value which says the dependency between the,m
mi /= np.max(mi)

plt.figure(figsize=(15, 5))

for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.scatter(X[:, i], y, edgecolor="black", s=20) # plotting all the features of X
    plt.xlabel("$x_{}$".format(i + 1), fontsize=14)
    if i == 0:
        plt.ylabel("$y$", fontsize=14)
    plt.title("F-test={:.2f}, MI={:.2f}".format(f_test[i], mi[i]), fontsize=16)
plt.show()
