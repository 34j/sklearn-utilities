from __future__ import annotations

from typing import Any, Generic

from pandas import DataFrame, Series
from typing_extensions import Self

from ..estimator_wrapper import EstimatorWrapperBase
from ..id_transformer import IdTransformer
from ..types import TX, TY, TEstimator, TTransformer


class TransformedTargetEstimatorVar(
    EstimatorWrapperBase[TEstimator], Generic[TEstimator, TTransformer]
):
    """TransformTargetRegressor with std/var support."""

    def __init__(
        self, estimator: TEstimator, transformer: TTransformer = IdTransformer()
    ) -> None:
        super().__init__(estimator)
        self.transformer = transformer

    def fit(self, X: TX, y: TY, **fit_params: Any) -> Self:
        # expand y to 2d
        if isinstance(y, Series):
            y = y.to_frame()
        elif y.ndim == 1:
            y = y[:, None]
        y = self.transformer.fit_transform(y).squeeze(axis=1)
        self.estimator.fit(X, y, **fit_params)
        return self

    def predict(self, X: TX, **predict_params: Any) -> TY | tuple[TY, TY]:
        if predict_params.get("return_std", False):
            pred, pred_std = self.estimator.predict_var(X, **predict_params)
            pred, pred_std = DataFrame(pred), DataFrame(pred_std)
            return self.transformer.inverse_transform(pred).squeeze(
                axis=1
            ), self.transformer.inverse_transform(pred_std, return_std=True).squeeze(
                axis=1
            )
        return self.transformer.inverse_transform(
            DataFrame(self.estimator.predict(X, **predict_params))
        ).squeeze(axis=1)

    def predict_var(self, X: TX, **predict_params: Any) -> TY:
        return self.transformer.inverse_transform_var(
            DataFrame(self.estimator.predict_var(X, **predict_params))
        ).squeeze(axis=1)
