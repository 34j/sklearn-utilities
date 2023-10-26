from __future__ import annotations

import warnings
from typing import Any, Callable, Generic

from pandas import Series
from typing_extensions import Self

from ..estimator_wrapper import EstimatorWrapperBase
from ..id_transformer import IdTransformer
from ..types import TX, TY, TEstimator, TTransformer


def _parse_2d(func: Callable[..., TY]) -> Callable[..., TY]:
    def wrapper(X: TX, **kwargs: Any) -> TY:
        ndim = X.ndim
        if ndim == 1:
            if isinstance(X, Series):
                X = X.to_frame()
            else:
                X = X[:, None]
        res = func(X, **kwargs)
        if ndim == 1:
            try:
                return res.squeeze(axis=1)
            except ValueError as e:
                warnings.warn(f"Failed to squeeze: {e}")
        return res

    return wrapper


class TransformedTargetEstimatorVar(
    EstimatorWrapperBase[TEstimator], Generic[TEstimator, TTransformer]
):
    """TransformTargetRegressor with std/var support."""

    def __init__(
        self,
        estimator: TEstimator,
        *,
        transformer: TTransformer = IdTransformer(),
        inverse_transform_separately: bool = False,
    ) -> None:
        super().__init__(estimator)
        self.transformer = transformer
        self.inverse_transform_separately = inverse_transform_separately

    def fit(self, X: TX, y: TY, **fit_params: Any) -> Self:
        y = _parse_2d(self.transformer.fit_transform)(y)
        self.estimator.fit(X, y, **fit_params)
        return self

    def predict(self, X: TX, **predict_params: Any) -> TY | tuple[TY, TY]:
        if predict_params.get("return_std", False):
            pred, pred_std = self.estimator.predict(X, **predict_params)
            if self.inverse_transform_separately:
                return _parse_2d(self.transformer.inverse_transform)(pred), _parse_2d(
                    self.transformer.inverse_transform
                )(pred_std, return_std=True)
            else:
                return _parse_2d(self.transformer.inverse_transform)(
                    (pred, pred_std), return_std=True
                )
        pred = self.estimator.predict(X, **predict_params)
        return _parse_2d(self.transformer.inverse_transform)(pred)

    def predict_var(self, X: TX, **predict_params: Any) -> TY:
        pred_var = self.estimator.predict_var(X, **predict_params)
        return _parse_2d(self.transformer.inverse_transform_var)(pred_var)
