from typing import Any, Generic

from typing_extensions import Self

from .estimator_wrapper import EstimatorWrapperBase
from .types import TEstimator
from .utils import intersect_X_y


class IntersectXY(EstimatorWrapperBase[TEstimator], Generic[TEstimator]):
    """Estimator wrapper that intersects X and y indices before fitting."""

    def __init__(self, estimator: TEstimator) -> None:
        """Estimator wrapper that intersects X and y indices before fitting.
        Parameters
        ----------
        estimator : TEstimator
            The estimator to be wrapped.
        """
        super().__init__(estimator)

    def fit(self, X: Any, y: Any, **fit_params: Any) -> Self:
        X, y = intersect_X_y(X, y)
        self.estimator.fit(X, y, **fit_params)
        return self
