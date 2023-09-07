from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
import pickle


# generate regression dataset
X, y = make_regression(
    n_samples=100, n_features=3, noise=0.1, random_state=1
)

# training linear regression model
linear_regression = LinearRegression()
linear_regression.fit(X, y)

# summary of the model
print("****** Initial model parameter.******")
print('Model intercept :', linear_regression.intercept_)
print('Model coefficients : ', linear_regression.coef_)
print('Model score : ', linear_regression.score(X, y))

# Serialising this model using pickle.dump
with open("linear_regression.pkl", "wb") as f:
    pickle.dump(linear_regression, f)

# deserialised
with open("linear_regression.pkl", "rb") as f:
    unpickled_linear_model = pickle.load(f)

# summary of the model
print("****** After unpickled the model. ******")
print('Model intercept :', unpickled_linear_model.intercept_)
print('Model coefficients : ', unpickled_linear_model.coef_)
print('Model score : ', unpickled_linear_model.score(X, y))