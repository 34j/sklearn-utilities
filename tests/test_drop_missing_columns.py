from pandas import DataFrame
from pandas.testing import assert_frame_equal

from sklearn_utilities.drop_missing_columns import DropMissingColumns


def test_drop() -> None:
    X = DataFrame({"a": [1, 2, 3, 4], "b": [1, 2, None, None], "c": [1, 2, 3, None]})
    est = DropMissingColumns(threshold_not_missing=0.5).set_output(transform="pandas")
    X_t1 = est.fit_transform(X)
    X_t2 = est.fit_transform(X, X["a"])
    X_t_expected = DataFrame({"a": [1, 2, 3, 4]})
    assert_frame_equal(X_t1, X_t_expected)
    assert_frame_equal(X_t2, X_t_expected)
