from typing import Any, TypeVar

from numpy.typing import NDArray
from pandas import DataFrame

TEstimator = TypeVar("TEstimator", bound=Any)
TX = TypeVar("TX", DataFrame, NDArray[Any])
