import warnings
from typing import Any, Callable, Literal

from pandas import DataFrame, Index
from sklearn.base import BaseEstimator, TransformerMixin
from typing_extensions import Self

from .types import TXPandas


class ReindexMissingColumns(BaseEstimator, TransformerMixin):
    """Reindex X to match the columns of the training data to avoid errors."""

    def __init__(
        self,
        *,
        if_missing: Literal["warn", "raise"]
        | Callable[["Index[Any]", "Index[Any]"], None] = "warn",
        reindex_kwargs: dict[
            Literal["method", "copy", "level", "fill_value", "limit", "tolerance"], Any
        ] = {},
    ) -> None:
        """Reindex X to match the columns of the training data to avoid errors.

        Parameters
        ----------
        if_missing : Literal['warn', 'raise'] | Callable[[Index[Any], Index[Any]], None], optional
            If callable, the first argument is the expected columns and the
            second argument is the actual columns, by default 'warn'
        reindex_kwargs : dict[Literal['method', 'copy', 'level', 'fill_value',
        'limit', 'tolerance'], Any], optional
            Keyword arguments to pass to reindex, by default {}
        """
        self.if_missing = if_missing
        self.reindex_kwargs = reindex_kwargs

    def fit(self, X: DataFrame, y: Any = None, **fit_params: Any) -> Self:
        self.feature_names_in_ = X.columns
        return self

    def transform(self, X: TXPandas, y: Any = None, **fit_params: Any) -> TXPandas:
        expected_columns = self.feature_names_in_
        actual_columns = X.columns
        if not expected_columns.equals(actual_columns):
            missing_columns = expected_columns.difference(actual_columns)
            if self.if_missing == "warn":
                warnings.warn(f"Missing columns: {missing_columns}")
            elif self.if_missing == "raise":
                raise ValueError(f"Missing columns: {missing_columns}")
            elif isinstance(self.if_missing, Callable):  # type: ignore
                self.if_missing(expected_columns, actual_columns)
            else:
                raise ValueError(f"Invalid value for if_missing: {self.if_missing}")

        return X.reindex(columns=self.feature_names_in_, **self.reindex_kwargs)
