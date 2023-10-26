from __future__ import annotations

import inspect
import re
import warnings
from typing import Any, Generic, Hashable, TypeVar

from numpy.typing import NDArray
from pandas import DataFrame, Index, Series

from ..estimator_wrapper import EstimatorWrapperBase, TEstimator

TArray = TypeVar("TArray", bound="DataFrame | Series | NDArray[Any]")


def to_frame_or_series(
    array: TArray,
    base_index: Index[Any],
    base_columns_or_name: Index[Any] | Hashable | None,
) -> DataFrame | Series | TArray:
    if isinstance(array, (DataFrame, Series)):
        return array
    try:
        if array.ndim == 1:
            return Series(
                array,
                index=base_index if array.shape[0] == len(base_index) else None,
                name=base_columns_or_name
                if not isinstance(base_columns_or_name, Index)
                else None,
            )
        if array.ndim == 2:
            return DataFrame(
                array,
                index=base_index if array.shape[0] == len(base_index) else None,
                columns=base_columns_or_name
                if (
                    isinstance(base_columns_or_name, Index)
                    and array.shape[1] == len(base_columns_or_name)
                )
                else None,
            )
    except Exception as e:
        warnings.warn(f"Could not convert {array} to DataFrame or Series: {e}")
        return array
    return array


def to_frame_or_series_tuple(
    array: tuple[TArray, ...] | TArray,
    base_index: Index[Any],
    base_columns_or_name: Index[Any] | Hashable,
) -> tuple[DataFrame | Series | TArray, ...] | DataFrame | Series | TArray:
    if isinstance(array, tuple):
        return tuple(
            to_frame_or_series(a, base_index, base_columns_or_name) for a in array
        )
    return to_frame_or_series(array, base_index, base_columns_or_name)


class DataFrameWrapper(EstimatorWrapperBase[TEstimator], Generic[TEstimator]):
    pattern_x: str
    y_columns_or_name: Index[Any] | Hashable | None = None

    def __init__(
        self,
        estimator: TEstimator,
        *,
        pattern_x: str = "^(:?fit|transform|fit_transform)$",
        pattern_y: str = "^predict.*?$",
    ) -> None:
        """A wrapper for estimators that returns pandas DataFrame or Series
        instead of numpy arrays for the methods that have "X" as an argument
        and the name matches the given pattern.

        Parameters
        ----------
        estimator : Any
            The estimator to be wrapped.
        pattern_x : str, optional
            The regex pattern to match the method names,
            by default "^(:?transform|fit_transform)$"
        pattern_y : str, optional
            The regex pattern to match the method names,
            by default "^predict.*$"
        """
        super().__init__(estimator)
        self.pattern_x = pattern_x
        self.pattern_y = pattern_y

    def _save_y_columns_or_name(self, y: Any) -> None:
        if isinstance(y, Series):
            self.y_columns_or_name = y.name
        elif isinstance(y, DataFrame):
            self.y_columns_or_name = y.columns

    def __getattribute__(self, __name: str) -> Any:
        try:
            # do not call super().__getattribute__
            return object.__getattribute__(self, __name)
        except AttributeError:
            attr = getattr(self.estimator, __name)

            x_match = re.search(self.pattern_x, __name)
            y_match = re.search(self.pattern_y, __name)
            if (
                callable(attr)
                and (x_match or y_match)
                and "X" in inspect.signature(attr).parameters
            ):

                def wrapper(*args: Any, **kwargs: Any) -> Any:
                    # get X to get index and columns
                    X = inspect.signature(attr).bind(*args, **kwargs).arguments["X"]

                    # save y columns or name
                    if "y" in inspect.signature(attr).parameters:
                        y = inspect.signature(attr).bind(*args, **kwargs).arguments["y"]
                        self._save_y_columns_or_name(y)

                    # get result
                    result = attr(*args, **kwargs)

                    # avoid fit() not returning self but self.estimator
                    if result is self.estimator:
                        return self

                    # support tuple for return_std=True, etc.
                    return to_frame_or_series_tuple(
                        result,
                        X.index,
                        X.columns if x_match else self.y_columns_or_name,
                    )

                return wrapper

            # behaves like EstimatorWrapperBase
            return attr
