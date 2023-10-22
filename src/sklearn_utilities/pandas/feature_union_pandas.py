from functools import partial
from typing import Any
from unittest.mock import patch

from pandas import concat
from sklearn.pipeline import FeatureUnion

_horizontal_concat = partial(concat, axis=1)


class FeatureUnionPandas(FeatureUnion):
    def fit_transform(self, X: Any, y: Any = None, **fit_params: Any) -> Any:
        with patch("numpy.hstack", side_effect=_horizontal_concat):
            return super().fit_transform(X, y, **fit_params)

    def transform(self, X: Any) -> Any:
        with patch("numpy.hstack", side_effect=_horizontal_concat):
            return super().transform(X)
