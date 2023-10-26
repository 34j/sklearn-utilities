import pandas as pd
from sklearn.datasets import make_regression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from sklearn_utilities.pandas.multioutput import SmartMultioutputEstimator


def test_SmartMultioutputEstimator():
    # Generate some random data
    X, y = make_regression(n_samples=100, n_features=5, n_targets=3)

    # convert to pandas DataFrame
    X, y = pd.DataFrame(X), pd.DataFrame(y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Create a SmartMultioutputEstimator object
    # with a GaussianProcessRegressor estimator
    kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0)) + WhiteKernel(
        noise_level=1, noise_level_bounds=(1e-10, 1e1)
    )
    estimator = SmartMultioutputEstimator(
        GaussianProcessRegressor(
            kernel=kernel, alpha=0.0, normalize_y=True, n_restarts_optimizer=10
        )
    )

    # Fit the estimator to the training data
    estimator.fit(X_train, y_train)

    # Predict the output for the test data
    y_pred, y_std = estimator.predict(X_test, return_std=True)

    # Check that the output has the correct shape
    assert y_pred.shape == y_test.shape
    assert y_std.shape == y_test.shape

    mean_squared_error(y_test, y_pred)
