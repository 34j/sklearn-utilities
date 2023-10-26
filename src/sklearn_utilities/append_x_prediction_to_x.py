from __future__ import annotations

from typing import Any, Generic, Hashable, Sequence

from pandas import DataFrame, Series, concat
from sklearn.base import BaseEstimator, TransformerMixin, check_is_fitted
from typing_extensions import Self

from .pandas import DataFrameWrapper
from .types import TEstimator


class AppendXPredictionToX(BaseEstimator, TransformerMixin, Generic[TEstimator]):
    """Append the prediction of X by the estimator to X."""

    estimator: TEstimator

    def __init__(
        self,
        estimator: TEstimator,
        *,
        variables: Sequence[Hashable] | None = None,
        append: bool = True,
        append_pred_diff: bool = True,
        append_pred_real_diff: bool = True,
    ) -> None:
        """Append the prediction of X by the estimator to X.
        The new columns are suffixed with "_pred".

        Parameters
        ----------
        estimator : TEstimator
            The estimator to be wrapped.
        variables : Sequence[Hashable] | None, optional
            The variables to be used for prediction, by default None
        append : bool, optional
            Whether to append the original X, by default True
        append_diff : bool, optional
            Whether to append the difference between
            the previous prediction and the current prediction, by default True
            The column names are suffixed with "_pred_diff"
        append_real_diff : bool, optional
            Whether to append the difference between
            the current value and the current prediction, by default True
            The column names are suffixed with "_pred_real_diff"
        """
        self.estimator = estimator
        self.variables = variables
        self.append = append
        self.append_pred_diff = append_pred_diff
        self.append_pred_real_diff = append_pred_real_diff

    def fit(self, X: DataFrame, y: Series | None = None, **fit_params: Any) -> Self:
        if self.variables is not None:
            X = X.loc[:, self.variables]
        X_future = X.shift(-1)  # do not specify freq

        # drop first
        X = X.iloc[:-1]
        X_future = X_future.iloc[:-1]

        self.estimator_ = DataFrameWrapper(self.estimator)
        self.estimator_.fit(X, X_future, **fit_params)
        return self

    def transform(
        self, X: DataFrame, y: Series | None = None, **transform_params: Any
    ) -> DataFrame:
        check_is_fitted(self, "estimator_")

        to_concat = []

        # X
        if self.append:
            to_concat.append(X)

        # X_pred
        X_variables = X.loc[:, self.variables] if self.variables is not None else X
        X_pred = self.estimator_.predict(X_variables)
        to_concat.append(X_pred.add_suffix("_pred"))

        # X_pred_diff
        if self.append_pred_diff:
            X_pred_diff = X_pred.diff(1)
            to_concat.append(X_pred_diff.add_suffix("_pred_diff"))

        # X_pred_real_diff
        if self.append_pred_real_diff:
            X_pred_real_diff = X_pred - X_variables
            to_concat.append(X_pred_real_diff.add_suffix("_pred_real_diff"))

        # concat
        X_concat = concat(to_concat, axis=1)
        return X_concat
