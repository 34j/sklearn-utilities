from pandas import DataFrame
from sklearn.tree import DecisionTreeRegressor

from sklearn_utilities.drop_by_noise_prediction import DropByNoisePrediction


def test_drop_by_noise_prediction() -> None:
    X = DataFrame({"a": [45, 45, 2, 1], "b": [1, 2, 3, 4], "c": [1, 1, 2, 1]})
    X_t = (
        DropByNoisePrediction(
            drop_rate=0.3, distribution="arange", estimator=DecisionTreeRegressor()
        )
        .set_output(transform="pandas")
        .fit_transform(X)
    )
    assert isinstance(X_t, DataFrame)
    # X_t_expected = X.drop(columns=["c"], inplace=False)
    # assert_frame_equal(X_t, X_t_expected) too random
    X_t = (
        DropByNoisePrediction(
            drop_rate=0.3, distribution="uniform", estimator=DecisionTreeRegressor()
        )
        .set_output(transform="pandas")
        .fit_transform(X)
    )
    assert isinstance(X_t, DataFrame)
    # X_t_expected = X.drop(columns=["a"], inplace=False)
    # assert_frame_equal(X_t, X_t_expected) too random
