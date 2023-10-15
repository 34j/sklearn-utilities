from typing import Any, Generic

from pandas import DataFrame
from typing_extensions import Self

from .estimator_wrapper import EstimatorWrapperBase
from .types import TEstimator


class DropMissingRowsY(EstimatorWrapperBase[TEstimator], Generic[TEstimator]):
    """A wrapper for estimators that drops NaN values from y before fitting."""

    def __init__(self, estimator: TEstimator) -> None:
        """A wrapper for estimators that drops NaN values from y before fitting.

        Parameters
        ----------
        estimator : TEstimator
            The estimator to be wrapped.
        """
        super().__init__(estimator)

    def fit(self, X: DataFrame, y: Any = None, **fit_params: Any) -> Self:
        y = y.dropna()
        idx = X.index.intersection(y.index)
        self.estimator.fit(X.loc[idx], y.loc[idx], **fit_params)
        return self
