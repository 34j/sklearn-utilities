from __future__ import annotations

import inspect
import re
from typing import Any, Generic, TypeVar

from numpy.typing import NDArray
from pandas import DataFrame, Series

from ..estimator_wrapper import EstimatorWrapperBase, TEstimator

TArray = TypeVar("TArray", bound="DataFrame | Series | NDArray[Any]")


def to_frame_or_series(array: TArray, X: DataFrame) -> DataFrame | Series | TArray:
    if isinstance(array, (DataFrame, Series)):
        return array
    try:
        if array.ndim == 1:
            return Series(
                array, index=X.index if X.shape[0] == array.shape[0] else None
            )
        if array.ndim == 2:
            return DataFrame(
                array,
                index=X.index if X.shape[0] == array.shape[0] else None,
                columns=X.columns if X.shape[1] == array.shape[1] else None,
            )
    except Exception:
        return array
    return array


class DataFrameWrapper(EstimatorWrapperBase[TEstimator], Generic[TEstimator]):
    pattern: str

    def __init__(
        self,
        estimator: TEstimator,
        *,
        pattern: str = "^(:?fit|transform|fit_transform|predict|predict_var)$",
    ) -> None:
        """A wrapper for estimators that returns pandas DataFrame or Series
        instead of numpy arrays for the methods that have "X" as an argument
        and the name matches the given pattern.

        Parameters
        ----------
        estimator : Any
            The estimator to be wrapped.
        pattern : str, optional
            The regex pattern to match the method names,
            by default "^(:?fit|transform|fit_transform|predict|predict_var)$"
        """
        super().__init__(estimator)
        self.pattern = pattern

    def __getattribute__(self, __name: str) -> Any:
        try:
            return object.__getattribute__(self, __name)
        except AttributeError:
            attr = getattr(self.estimator, __name)
            # if has "X"
            if (
                callable(attr)
                and re.search(self.pattern, __name)
                and "X" in inspect.signature(attr).parameters
            ):

                def wrapper(*args: Any, **kwargs: Any) -> Any:
                    X = inspect.signature(attr).bind(*args, **kwargs).arguments["X"]
                    return to_frame_or_series(attr(*args, **kwargs), X)

                return wrapper
            return attr
