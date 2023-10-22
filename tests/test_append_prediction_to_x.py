from numpy import ndarray
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from sklearn_utilities.append_prediction_to_x import (
    AppendPredictionToX,
    AppendPredictionToXSingle,
)

X, y = load_diabetes(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)
y_pred = LinearRegression().fit(X_train, y_train).predict(X_train)
assert isinstance(y_pred, ndarray)


def test_single() -> None:
    X_t = AppendPredictionToXSingle(estimator=LinearRegression()).fit_transform(
        X_train, y_train
    )
    assert X_t.shape == (X_train.shape[0], X_train.shape[1] + 1)
    assert X_t.columns[-1] == "y_pred_LinearRegression_0"
    # assert assert_array_almost_equal(X_t["y_pred_LinearRegression_0"], y_pred)


def test_multiple() -> None:
    X_t = AppendPredictionToX(estimators=LinearRegression()).fit_transform(
        X_train, y_train
    )
    assert X_t.shape == (X_train.shape[0], X_train.shape[1] + 1)
    assert X_t.columns[-1] == "y_pred_LinearRegression_0_0"
    # assert assert_array_almost_equal(X_t["y_pred_LinearRegression_0_0"], y_pred)
