from lightgbm import LGBMRegressor
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from sklearn_utilities.recursive_fit_subtract_regressor import (
    RecursiveFitSubtractRegressor,
)


def test_regressor():
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    y_pred_baseline = (
        LGBMRegressor(learning_rate=0.01).fit(X_train, y_train).predict(X_test)
    )
    y_pred = (
        RecursiveFitSubtractRegressor(
            estimators=[
                LGBMRegressor(learning_rate=0.01),
                LGBMRegressor(learning_rate=0.01),
            ]
        )
        .fit(X_train, y_train)
        .predict(X_test)
    )
    assert y_pred.shape == y_test.shape
    assert type(y_pred_baseline) == type(y_pred)  # noqa
    assert mean_squared_error(y_pred_baseline, y_test) > mean_squared_error(
        y_pred, y_test
    )
