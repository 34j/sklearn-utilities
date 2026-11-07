from __future__ import annotations

from typing import Any, Self, TypeVar

from sklearn.base import BaseEstimator, TransformerMixin

T = TypeVar("T", bound=Any)


class IdTransformer(BaseEstimator, TransformerMixin):
    """A transformer that does nothing."""

    def fit(self, X: Any, y: Any = None, **fit_params: Any) -> Self:
        return self

    def transform(self, X: T, **transform_params: Any) -> T:
        return X

    def inverse_transform(self, X: T, **transform_params: Any) -> T:
        return X

    def inverse_transform_var(self, X: T, **transform_params: Any) -> T:
        return X
