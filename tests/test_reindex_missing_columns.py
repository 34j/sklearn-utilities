import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from sklearn_utilities.reindex_missing_columns import ReindexMissingColumns


def test_reindex_missing_columns() -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df_missing = df.drop(columns=["a"], inplace=False)
    df_expected = pd.DataFrame({"a": [np.nan] * 3, "b": [4, 5, 6]})
    estimator = ReindexMissingColumns().fit(df)

    with pytest.warns(UserWarning):
        df_out = estimator.transform(df_missing)

    assert_frame_equal(df_out, df_expected)
