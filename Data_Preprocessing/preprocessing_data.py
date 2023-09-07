"""
https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py

Feature 0 (median income in a block) and feature 5 (average house occupancy) of the California Housing dataset
have very different scales and contain some very large outliers These two characteristics lead to difficulties to
visualize the data and, more importantly, they can degrade the predictive performance of many machine learning
algorithms.

Unscaled data can also slow down or even prevent the convergence of many gradient-based estimators.

estimators are designed with the assumption that each feature takes values close to zero or more importantly that all features vary on comparable scales.

metric-based and gradient-based estimators often assume approximately standardized data (Centered features with unit variances.)

This example uses different scalers, transformers, and normalizers to bring the data within a pre-defined range.

1. MaxAbsScaler: Scale each feature by its maximum absolute value.
                This estimator scales and translates each feature individually such
                that the maximal absolute value of each feature in the
                training set will be 1.0. It does not shift/center the data, and
                thus does not destroy any sparsity.

2. MinMaxScaler : Transform features by scaling each feature to a given range.

                This estimator scales and translates each feature individually such
                that it is in the given range on the training set, e.g. between
                zero and one.

                The transformation is given by::

                    X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
                    X_scaled = X_std * (max - min) + min

                where min, max = feature_range.
3. Normalizer: normalization refers to a per sample transformation instead of a per feature transformation.
                Normalize samples individually to unit norm.

                Each sample (i.e. each row of the data matrix) with at least one
                non zero component is rescaled independently of other samples so
                that its norm (l1, l2 or inf) equals one.

4. PowerTransformer: non-linear transformations in which data is mapped to a normal distribution to stabilize variance and minimize skewness.
                Apply a power transform featurewise to make data more Gaussian-like.
                Power transforms are a family of parametric, monotonic transformations
                that are applied to make data more Gaussian-like. This is useful for
                modeling issues related to heteroscedasticity (non-constant variance),
                or other situations where normality is desired.

5. Quantile transformer: non-linear transformation in which distance between marginal outliers and inliers are shrunk.
                        Transform features using quantiles information.
                        This method transforms the features to follow a uniform or a normal
                        distribution. Therefore, for a given feature, this transformation tends
                        to spread out the most frequent values. It also reduces the impact of
                        (marginal) outliers: this is therefore a robust preprocessing scheme.

6. RobustScaler: Scale features using statistics that are robust to outliers.
                This Scaler removes the median and scales the data according to
                the quantile range (defaults to IQR: Interquartile Range).
                The IQR is the range between the 1st quartile (25th quantile)
                and the 3rd quartile (75th quantile).

                Centering and scaling happen independently on each feature by
                computing the relevant statistics on the samples in the training
                set. Median and interquartile range are then stored to be used on
                later data using the :meth:`transform` method.

                Standardization of a dataset is a common requirement for many
                machine learning estimators. Typically this is done by removing the mean
                and scaling to unit variance. However, outliers can often influence the
                sample mean / variance in a negative way. In such cases, the median and
                the interquartile range often give better results.

7. StandardScaler:Standardize features by removing the mean and scaling to unit variance.

                        The standard score of a sample `x` is calculated as:

                            z = (x - u) / s

                        where `u` is the mean of the training samples or zero if `with_mean=False`,
                        and `s` is the standard deviation of the training samples or one if
                        `with_std=False`.
8. minmax_scale: Transform features by scaling each feature to a given range.

"""

import matplotlib as mpl
import numpy as np
from matplotlib import cm  # cm is for the color map
from matplotlib import pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import (
    MaxAbsScaler,  # Scale each feature by its maximum absolute value
    MinMaxScaler,  # Transform features by scaling each feature to a given range
    Normalizer,  # Normalize samples individually to unit norm.
    PowerTransformer,  # Apply a power transform feature wise to make data more Gaussian-like/ normal distribution
    QuantileTransformer,  # Transform features uses quantiles information;distance between outliers, inliers are shrunk.
    RobustScaler,  # Scale features using statistics that are robust to outliers.
    StandardScaler,  # Standardize features by removing the mean and scaling to unit variance.
    minmax_scale  # Transform features by scaling each feature to a given range.
)

# data set
dataset = fetch_california_housing()
X_full, y_full = dataset.data, dataset.target  # dataset.data shape (20640, 8) and dataset.target shape (20640,)
feature_name = dataset.feature_names  # list of length 8 features

# after extracting the features from feature name. we can map the features
feature_mapping = {
    "MedInc": "Median income in block",
    "HouseAge": "Median house age in block",
    "AveRooms": "Average number of rooms",
    "AveBedrms": "Average number of bedrooms",
    "Population": "Block population",
    "AveOccup": "Average house occupancy",
    "Latitude": "House block latitude",
    "Longitude": "House block longitude",
}

features = ["MedInc", "AveOccup"]
features_index = [feature_name.index(feature) for feature in features]

X = X_full[:, features_index]

distributions = [
    ("Unscaled data", X),
    ("Data after standard scaling:", StandardScaler().fit_transform(X)),
    ("Data after min-max scaling:", MinMaxScaler().fit_transform(X)),
    ("Data after max-abs scaling:", MaxAbsScaler().fit_transform(X)),
    ("Data after robust scaling:", RobustScaler(quantile_range=(25, 75)).fit_transform(X)),
    ("Data after power transformation:", PowerTransformer(method="yeo-johnson").fit_transform(X)),
    ("Data after power transformation (Box-Cox)", PowerTransformer(method="box-cox").fit_transform(X)),
    ("Data after quantile transformation (uniform pdf)",
     QuantileTransformer(output_distribution="uniform").fit_transform(X)),
    ("Data after quantile transformation (gaussian pdf)",
     QuantileTransformer(output_distribution="normal").fit_transform(X)),
    ("Data after sample-wise L2 normalizing", Normalizer().fit_transform(X))
]

# Scale the output between 0 to 1 for the color bar
y = minmax_scale(y_full)

# as plasma [it is used for color map] & it does not exist in matplotlib < 1.5
# so used cm function in matplotlib which gives color map.
cmap = getattr(cm, "plasma_r", cm.hot_r)


def create_axes(title, figsize=(16, 6)):
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title)

    # define the axis for the first plot
    left, width = 0.1, 0.22
    bottom, height = 0.1, 0.7
    bottom_h = height + 0.15
    left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.1]
    rect_histy = [left_h, bottom, 0.05, height]

    ax_scatter = plt.axes(rect_scatter)
    ax_histx = plt.axes(rect_histx)
    ax_histy = plt.axes(rect_histy)

    # define the axis for the zoomed-in plot
    left = width + left + 0.2
    left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.1]
    rect_histy = [left_h, bottom, 0.05, height]

    ax_scatter_zoom = plt.axes(rect_scatter)
    ax_histx_zoom = plt.axes(rect_histx)
    ax_histy_zoom = plt.axes(rect_histy)

    # define the axis for the colorbar
    left, width = width + left + 0.13, 0.01

    rect_colorbar = [left, bottom, width, height]
    ax_colorbar = plt.axes(rect_colorbar)

    return (
        (ax_scatter, ax_histy, ax_histx),
        (ax_scatter_zoom, ax_histy_zoom, ax_histx_zoom),
        ax_colorbar,
    )


def plot_distribution(axes, X, y, hist_nbins=50, title="", x0_label="", x1_label=""):
    ax, hist_X1, hist_X0 = axes

    ax.set_title(title)
    ax.set_xlabel(x0_label)
    ax.set_ylabel(x1_label)

    # The scatter plot
    colors = cmap(y)
    ax.scatter(X[:, 0], X[:, 1], alpha=0.5, marker="o", s=5, lw=0, c=colors)

    # Removing the top and the right spine for aesthetics
    # make nice axis layout
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))

    # Histogram for axis X1 (feature 5)
    hist_X1.set_ylim(ax.get_ylim())
    hist_X1.hist(
        X[:, 1], bins=hist_nbins, orientation="horizontal", color="grey", ec="grey"
    )
    hist_X1.axis("off")

    # Histogram for axis X0 (feature 0)
    hist_X0.set_xlim(ax.get_xlim())
    hist_X0.hist(
        X[:, 0], bins=hist_nbins, orientation="vertical", color="grey", ec="grey"
    )
    hist_X0.axis("off")



def make_plot(item_idx):
    title, X = distributions[item_idx]
    ax_zoom_out, ax_zoom_in, ax_colorbar = create_axes(title)
    axarr = (ax_zoom_out, ax_zoom_in)
    plot_distribution(
        axarr[0],
        X,
        y,
        hist_nbins=200,
        x0_label=feature_mapping[features[0]],
        x1_label=feature_mapping[features[1]],
        title="Full data",
    )

    # zoom-in
    zoom_in_percentile_range = (0, 99)
    cutoffs_X0 = np.percentile(X[:, 0], zoom_in_percentile_range)
    cutoffs_X1 = np.percentile(X[:, 1], zoom_in_percentile_range)

    non_outliers_mask = np.all(X > [cutoffs_X0[0], cutoffs_X1[0]], axis=1) & np.all(
        X < [cutoffs_X0[1], cutoffs_X1[1]], axis=1
    )
    plot_distribution(
        axarr[1],
        X[non_outliers_mask],
        y[non_outliers_mask],
        hist_nbins=50,
        x0_label=feature_mapping[features[0]],
        x1_label=feature_mapping[features[1]],
        title="Zoom-in",
    )

    norm = mpl.colors.Normalize(y_full.min(), y_full.max())
    mpl.colorbar.ColorbarBase(
        ax_colorbar,
        cmap=cmap,
        norm=norm,
        orientation="vertical",
        label="Color mapping for values of y",
    )


if __name__ == "__main__":
    make_plot(1)
