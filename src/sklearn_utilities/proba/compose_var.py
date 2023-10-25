from __future__ import annotations

from typing import Any, Generic, Literal, overload

from typing_extensions import Self

from ..estimator_wrapper import EstimatorWrapperBase
from ..types import TX, TY, TEstimator, TEstimatorVar
from .dummy_regressor import DummyRegressorVar


class ComposeVarEstimator(
    EstimatorWrapperBase[TEstimator], Generic[TEstimator, TEstimatorVar]
):
    """Compose an estimator with a variance estimator."""

    def __init__(
        self, estimator: TEstimator, estimator_var: TEstimatorVar = DummyRegressorVar()
    ) -> None:
        """Compose an estimator with a variance estimator.

        Parameters
        ----------
        estimator : TEstimator
            The estimator to be wrapped.
        estimator_var : TEstimatorVar, optional
            The variance estimator to be wrapped, by default DummyRegressorVar()
        """
        super().__init__(estimator)
        self.estimator_var = estimator_var

    def fit(self, X: TX, y: TY, **fit_params: Any) -> Self:
        self.estimator.fit(X, y, **fit_params)
        self.estimator_var.fit(X, y, **fit_params)
        return self

    @overload
    def predict(
        self, X: TX, return_std: Literal[False] = ..., **predict_params: Any
    ) -> TY:
        ...

    @overload
    def predict(
        self, X: TX, return_std: Literal[True], **predict_params: Any
    ) -> tuple[TY, TY]:
        ...

    def predict(
        self, X: TX, return_std: bool = False, **predict_params: Any
    ) -> TY | tuple[TY, TY]:
        if return_std:
            return (
                self.estimator.predict(X, **predict_params),
                self.estimator_var.predict(X, return_std=True, **predict_params)[1],
            )
        return self.estimator.predict(X, **predict_params)

    def predict_var(self, X: TX, **predict_params: Any) -> TY:
        return self.estimator_var.predict_var(X, **predict_params)
