from __future__ import annotations

from typing import Any, Generic, Mapping, Sequence

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray
from typing_extensions import Self

from .estimator_wrapper import EstimatorWrapperBase
from .types import TEstimator


class RecursiveFitSubtractRegressor(
    Generic[TEstimator], EstimatorWrapperBase[TEstimator]
):
    """Regressor that fits the residual of the prediction of the previous model."""

    estimators: Sequence[TEstimator]

    def __init__(
        self, estimators: Sequence[TEstimator], *, n_jobs: int | None = None
    ) -> None:
        """Regressor that fits the residual of the prediction of the previous model.

        Parameters
        ----------
        estimators : Sequence[TEstimator]
            The estimators to be wrapped.
        n_jobs : int | None, optional
            The number of jobs to run in parallel when predicting, by default None
            Note that fitting cannot be parallelized.
        """
        self.estimators = estimators
        self.n_jobs = n_jobs

    def fit(
        self,
        X: Any,
        y: Any,
        predict_params: Mapping[str, Any] | None = None,
        **fit_params: Any,
    ) -> Self:
        if predict_params is None:
            predict_params = {}
        for i, model in enumerate(self.estimators):
            model.fit(X, y, **fit_params)
            if i < len(self.estimators) - 1:
                y -= model.predict(X, **predict_params)
        return self

    def predict(self, X: Any, **predict_params: Any) -> NDArray[Any]:
        parrarel_results = Parallel(n_jobs=self.n_jobs)(
            delayed(model.predict)(X, **predict_params) for model in self.estimators
        )
        if parrarel_results is None:
            raise ValueError("joblib.Parallel returned None")
        return np.sum(parrarel_results, axis=0)

    def predict_var(self, X: Any, **predict_params: Any) -> NDArray[Any]:
        return self.estimator.predict_var(X, **predict_params)

    @property
    def estimator(self) -> TEstimator:  # type: ignore
        return self.estimators[-1]
