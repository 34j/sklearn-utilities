from typing import Any, Callable, Sequence

from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin, check_is_fitted
from typing_extensions import Self

from .id_transformer import IdTransformer


class ExcludedColumnTransformerPandas(BaseEstimator, TransformerMixin):
    """A transformer that excludes columns from the input data frame."""

    feature_names_in_: Sequence[str]
    feature_names_out_: Sequence[str]

    def __init__(
        self,
        estimator: Any = IdTransformer(),
        exclude_columns: Sequence[str] | Callable[[Sequence[str]], Sequence[bool]] = [],
    ) -> None:
        """A transformer that excludes columns from the input data frame.

        Parameters
        ----------
        estimator : Any, optional
            The estimator to be wrapped, by default IdTransformer()
        exclude_columns : Sequence[str] | Callable[[Sequence[str]], Sequence[bool]],
        optional
            The columns to be excluded, by default []
            If callable, a function that takes the column names as an argument.
        """
        self.exclude_columns = exclude_columns
        self.estimator = estimator

    def _fit(self, X: DataFrame) -> None:
        self.feature_names_in_ = list(X.columns)
        if isinstance(self.exclude_columns, Callable):  # type: ignore
            exclude_columns = X.columns[self.exclude_columns(X.columns)]  # type: ignore
        else:
            exclude_columns = self.exclude_columns
        self.feature_names_out_ = list(set(X.columns) - set(exclude_columns))

    def fit(self, X: DataFrame, **fit_params: Any) -> Self:
        self._fit(X)
        self.estimator.fit(X[self.feature_names_out_], **fit_params)
        return self

    def transform(
        self, X: DataFrame, y: Any = None, **transform_params: Any
    ) -> DataFrame:
        check_is_fitted(self)
        return self.estimator.transform(X[self.feature_names_out_], **transform_params)

    def fit_transform(
        self, X: DataFrame, y: Any = None, **fit_params: Any
    ) -> DataFrame:
        self._fit(X)
        return self.estimator.fit_transform(X[self.feature_names_out_], y, **fit_params)


class IncludedColumnTransformerPandas(BaseEstimator, TransformerMixin):
    """A transformer that includes columns from the input data frame."""

    feature_names_in_: Sequence[str]
    feature_names_out_: Sequence[str]

    def __init__(
        self,
        estimator: Any = IdTransformer(),
        include_columns: Sequence[str] | Callable[[Sequence[str]], Sequence[bool]] = [],
    ) -> None:
        """A transformer that includes columns from the input data frame.

        Parameters
        ----------
        estimator : Any, optional
            The estimator to be wrapped, by default IdTransformer()
        include_columns : Sequence[str] | Callable[[Sequence[str]], Sequence[bool]],
        optional
            The columns to be included, by default []
            If callable, a function that takes the column names as an argument.
        """
        self.include_columns = include_columns
        self.estimator = estimator

    def _fit(self, X: DataFrame) -> None:
        self.feature_names_in_ = list(X.columns)
        if isinstance(self.include_columns, Callable):  # type: ignore
            self.feature_names_out_ = X.columns[
                self.include_columns(X.columns)  # type: ignore
            ]
        else:
            self.feature_names_out_ = self.include_columns  # type: ignore

    def fit(self, X: DataFrame, **fit_params: Any) -> Self:
        self._fit(X)
        self.estimator.fit(X[self.feature_names_out_], **fit_params)
        return self

    def transform(
        self, X: DataFrame, y: Any = None, **transform_params: Any
    ) -> DataFrame:
        check_is_fitted(self)
        return self.estimator.transform(X[self.feature_names_out_], **transform_params)

    def fit_transform(
        self, X: DataFrame, y: Any = None, **fit_params: Any
    ) -> DataFrame:
        self._fit(X)
        return self.estimator.fit_transform(X[self.feature_names_out_], y, **fit_params)
