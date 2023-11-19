from typing import Any

from pandas import DataFrame
from pandas.testing import assert_frame_equal

from sklearn_utilities.report_non_finite import ReportNonFinite


def test_fit(caplog: Any) -> None:
    caplog.set_level("DEBUG")
    X = DataFrame(
        {
            "clean": [1, 2, 3],
            "inf": [4, float("inf"), 6],
            "nan": [7, 8, float("nan")],
            "both": [float("inf"), float("nan"), 9],
        }
    )
    transformer = ReportNonFinite(on_fit=True, calc_corr=True)
    transformer.fit(X)
    # Add assertions to check the expected behavior of the fit method
    assert "Non-finite values in X during fit" in caplog.text


def test_transform(caplog: Any) -> None:
    caplog.set_level("DEBUG")
    X = DataFrame(
        {
            "clean": [1, 2, 3],
            "inf": [4, float("inf"), 6],
            "nan": [7, 8, float("nan")],
            "both": [float("inf"), float("nan"), 9],
        }
    )
    transformer = ReportNonFinite(on_transform=True, calc_corr=True)
    X_transformed = transformer.transform(X)
    # Add assertions to check the expected behavior of the transform method
    assert "Non-finite values in X during transform" in caplog.text
    assert_frame_equal(X_transformed, X)
