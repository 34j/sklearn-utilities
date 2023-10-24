from __future__ import annotations

from typing import Any, TypeVar

from numpy.typing import NDArray
from pandas import DataFrame, Series

TEstimator = TypeVar("TEstimator", bound=Any)
TEstimatorVar = TypeVar("TEstimatorVar", bound=Any)
TTransformer = TypeVar("TTransformer", bound=Any)

TX = TypeVar("TX", DataFrame, NDArray[Any])
TY = TypeVar("TY", DataFrame, Series, NDArray[Any])
TXPandas = TypeVar("TXPandas", bound=DataFrame)
TYPandas = TypeVar("TYPandas", DataFrame, Series)
