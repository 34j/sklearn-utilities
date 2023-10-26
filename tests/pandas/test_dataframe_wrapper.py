from pandas import concat
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from sklearn_utilities.pandas.dataframe_wrapper import DataFrameWrapper

X, y = load_diabetes(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)


def test_regressor() -> None:
    est = DataFrameWrapper(LinearRegression())
    est.fit(X_train, y_train)
    print(est)
    y_pred = est.predict(X_test)
    assert y_pred.shape == y_test.shape
    assert isinstance(y_pred, type(y_test))
    assert y_pred.index.equals(y_test.index)
    assert y_pred.name == y_train.name
    assert y_pred.name == y_test.name


def test_regressor_multi() -> None:
    Y_train = concat([y_train, y_train], axis=1)
    Y_test = concat([y_test, y_test], axis=1)
    est = DataFrameWrapper(LinearRegression()).fit(X_train, Y_train)
    Y_pred = est.predict(X_test)
    assert Y_pred.shape == Y_test.shape
    assert isinstance(Y_pred, type(Y_train))
    assert Y_pred.index.equals(Y_test.index)
    assert Y_pred.columns.equals(Y_train.columns)
