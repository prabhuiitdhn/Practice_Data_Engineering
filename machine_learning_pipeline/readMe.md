# Machine Learning Pipeline

> Reference: [sklearn.pipeline.Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)

## Overview

**Pipeline** of transforms with a final estimator.

Sequentially apply a list of transforms and a final estimator. Intermediate steps of the pipeline must be **transforms** — they must implement `fit` and `transform` methods. The final estimator only needs to implement `fit`. The transformers in the pipeline can be cached using the `memory` argument.

The purpose of the pipeline is to assemble several steps that can be **cross-validated together** while setting different parameters.

---

## Creating a Pipeline

```python
from sklearn.pipeline import make_pipeline

make_pipeline(
    steps,    # List of (name, transform) tuples that are chained sequentially
    memory,   # Used to cache fitted transformers. If a string, it is the path to the caching directory.
    verbose,  # If True, the time elapsed while fitting each step will be printed.
)
```

### Attributes

| Attribute | Description |
|-----------|-------------|
| `named_steps` | Access the steps by name |
| `classes_` | The class labels |
| `n_features_in_` | Number of features seen during first step `fit` method |
| `feature_names_in_` | Names of features seen during first step `fit` method |

---

## Example

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

X, y = make_classification(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)
```
