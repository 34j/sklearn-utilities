from __future__ import annotations

from typing import Any, Generic, Hashable, Sequence

from pandas import DataFrame, Series, concat
from sklearn.base import BaseEstimator, TransformerMixin, check_is_fitted
from sklearn.multioutput import MultiOutputRegressor
from typing_extensions import Self

from .drop_missing_rows_y import DropMissingRowsY
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
        n_jobs: int | None = -1,
        verbose: int = 1,
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
        n_jobs : int | None, optional
            The number of jobs to run in parallel, by default -1
        verbose : int, optional
            The verbosity level, by default 1
        """
        self.estimator = estimator
        self.variables = variables
        self.append = append
        self.append_pred_diff = append_pred_diff
        self.append_pred_real_diff = append_pred_real_diff
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X: DataFrame, y: Series | None = None, **fit_params: Any) -> Self:
        if self.variables is not None:
            X = X.loc[:, self.variables]
        X_future = X.shift(-1)  # do not specify freq
        self.estimator_ = MultiOutputRegressor(
            DropMissingRowsY(self.estimator), n_jobs=self.n_jobs
        )
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
        X_pred = self.estimator_.predict(
            X.loc[:, self.variables] if self.variables is not None else X
        ).add_suffix("_pred")
        to_concat.append(X_pred)

        # X_pred_diff
        if self.append_pred_diff:
            X_pred_diff = X_pred.diff(1).add_suffix("_pred_diff")
            to_concat.append(X_pred_diff)

        # X_pred_real_diff
        if self.append_pred_real_diff:
            X_pred_real_diff = (X_pred - X.loc[X_pred.index, X.columns]).add_suffix(
                "_pred_real_diff"
            )
            to_concat.append(X_pred_real_diff)

        # concat
        X_concat = concat(to_concat, axis=1)
        return X_concat
