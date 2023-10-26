from __future__ import annotations

from typing import Literal

import numpy as np
from numpy import ndarray
from numpy.typing import ArrayLike
from pandas import DataFrame, Series
from sklearn.dummy import DummyRegressor
from typing_extensions import Self

from ..utils import drop_X_y


class DummyRegressorVar(DummyRegressor):
    """DummyRegressor with 1.0 variance."""

    def __init__(
        self,
        *,
        strategy: Literal["mean", "median", "quantile", "constant", "mean"] = "mean",
        constant: float | None | ArrayLike | int = None,
        quantile: float | None = None,
        allow_nan: bool = True,
    ) -> None:
        super().__init__(strategy=strategy, constant=constant, quantile=quantile)
        self.allow_nan = allow_nan
        self._var_regressor = DummyRegressor(strategy="constant", constant=1.0)

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: ArrayLike | None = None,
    ) -> Self:
        if (
            self.allow_nan
            and isinstance(y, (Series, DataFrame))
            and isinstance(X, (Series, DataFrame))
        ):
            X, y = drop_X_y(X, y)
        self._var_regressor = DummyRegressor(strategy="constant", constant=np.var(y))
        self._var_regressor.fit(X, y, sample_weight)
        return super().fit(X, y, sample_weight)

    def predict(
        self, X: ArrayLike, return_std: bool = False
    ) -> ndarray | tuple[ndarray, ndarray]:
        return super().predict(X, False), self.predict_var(X)

    def predict_var(self, X: ArrayLike) -> ArrayLike:
        return self._var_regressor.predict(X)
