from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from sklearn_utilities.pandas.dataframe_wrapper import DataFrameWrapper


def test_regressor() -> None:
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    est = LinearRegression()
    est = DataFrameWrapper(est)
    est.fit(X_train, y_train)
    y_pred = est.predict(X_test)
    assert y_pred.shape == y_test.shape
    assert isinstance(y_pred, type(y_test))
