from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from sklearn_utilities.append_x_prediction_to_x import AppendXPredictionToX

X, y = load_diabetes(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)


def test_x() -> None:
    X_t = AppendXPredictionToX(estimator=LinearRegression()).fit_transform(X_train)
    assert X_t.shape == (X_train.shape[0], X_train.shape[1] * 4)
