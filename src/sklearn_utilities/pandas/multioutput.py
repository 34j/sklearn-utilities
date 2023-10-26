from __future__ import annotations

from logging import getLogger
from typing import Any, Generic, TypeVar

import joblib
import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_is_fitted
from tqdm_joblib import tqdm_joblib
from typing_extensions import Self

from ..types import TEstimator
from ..utils import drop_X_y
from .dataframe_wrapper import to_frame_or_series_tuple

LOG = getLogger(__name__)

# Scikit learn compatible esitimator that wraps an estimator
# Creates a new estimator for each target variable
# X and Y are pandas dataframes
# X.index.intersection(y.index) is used to fit the estimator,
# calculate for each column
# not X.intersection(Y.index)
# use joblib Parallel to fit the estimators in parallel

TX = TypeVar("TX", bound="DataFrame | Series")
TY = TypeVar("TY", bound="DataFrame | Series")


def _fit_X_y(
    estimator: Any,
    X: DataFrame,
    y: Series,
    *,
    pass_numpy: bool = False,
    safe: bool = False,
    **fit_params: Any,
) -> BaseEstimator:
    X, y = drop_X_y(X, y)
    LOG.debug(f"Length of {y.name}: {len(y)}")
    estimator = clone(estimator, safe=safe)
    if pass_numpy:
        return estimator.fit(X.values, y.values, **fit_params)
    return estimator.fit(X, y, **fit_params)


class SmartMultioutputEstimator(BaseEstimator, RegressorMixin, Generic[TEstimator]):
    estimator: TEstimator
    estimators_: list[TEstimator]

    def __init__(
        self,
        estimator: TEstimator,
        *,
        n_jobs: int | None = -1,
        verbose: int = 1,
        pass_numpy: bool = False,
    ) -> None:
        """MultioutputEstimator that
        1. Supports tuples of arrays in `predict()` (for
        `return_std=True`)
        2. Returns list of scores in `score()`
        3. Supports pandas DataFrame and Series

        Parameters
        ----------
        estimator : TEstimator
            The estimator to be wrapped.
        n_jobs : int | None, optional
            The number of jobs to run in parallel, by default -1
        verbose : int, optional
            Whether to show progress bar, by default 1
        pass_numpy : bool, optional
            Whether to pass numpy arrays to the estimator, by default False
        """
        self.estimator = clone(estimator, safe=False)
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.pass_numpy = pass_numpy

    def fit(self, X: DataFrame, y: DataFrame, **fit_params: Any) -> Self:
        self.feature_names_in_ = X.columns
        self.y_names_in_ = y.columns
        with tqdm_joblib(
            desc="Fitting estimators",
            total=len(y.columns),
            disable=self.verbose == 0,
        ):
            estimators = joblib.Parallel(n_jobs=self.n_jobs)(
                joblib.delayed(_fit_X_y)(
                    self.estimator,
                    X,
                    item,
                    pass_numpy=self.pass_numpy,
                    **fit_params,
                )
                for col, item in y.items()
            )
            if estimators is None:
                raise ValueError("No estimators fitted")
            self.estimators_ = list(estimators)
        return self

    def predict(
        self, X: DataFrame, **predict_params: Any
    ) -> (
        DataFrame
        | Series
        | NDArray[Any]
        | tuple[DataFrame | Series | NDArray[Any], ...]
    ):
        check_is_fitted(self)
        X = X[self.feature_names_in_]
        preds = [est.predict(X, **predict_params) for est in self.estimators_]
        preds_: DataFrame | Series | NDArray[Any] | tuple[
            DataFrame | Series | NDArray[Any], ...
        ]
        if any(isinstance(pred, tuple) for pred in preds):
            # list of tuples of arrays to tuples of arrays
            preds_ = tuple(np.array(pred).T for pred in zip(*preds))
        else:
            preds_ = np.array(preds).T
        return to_frame_or_series_tuple(preds_, X.index, self.y_names_in_)

    def score(self, X: DataFrame, y: DataFrame, **score_params: Any) -> NDArray[Any]:
        check_is_fitted(self)
        X = X[self.feature_names_in_]
        y = y[self.y_names_in_]
        return np.array(
            [
                est.score(X, y[col], **score_params)
                for est, col in zip(self.estimators_, self.y_names_in_)
            ]
        )

    def predict_var(
        self, X: DataFrame, **predict_params: Any
    ) -> (
        DataFrame
        | Series
        | NDArray[Any]
        | tuple[DataFrame | Series | NDArray[Any], ...]
    ):
        check_is_fitted(self)
        X = X[self.feature_names_in_]
        preds = np.array(
            [est.predict_var(X, **predict_params) for est in self.estimators_]
        ).T
        return to_frame_or_series_tuple(preds, X.index, self.y_names_in_)

    def __iter__(self) -> Any:
        return iter(self.estimators_)

    def __len__(self) -> int:
        return len(self.estimators_)
