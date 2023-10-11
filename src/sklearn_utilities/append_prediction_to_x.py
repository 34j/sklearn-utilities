from typing import Any, Generic, Sequence, TypeVar

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray
from pandas import DataFrame, Series, concat
from sklearn.base import BaseEstimator, TransformerMixin, check_is_fitted, clone
from typing_extensions import Self

TEstimator = TypeVar("TEstimator", bound=Any)
TX = TypeVar("TX", DataFrame, NDArray[Any])


def generate_new_prefix(X: DataFrame, prefix: str = "y_pred_") -> str:
    """Generate a new column name for the prediction."""
    last_number = (
        X.columns.to_series()
        .str.extract(rf"^{prefix}(\d+)_.+$", expand=False)
        .fillna(-1)
        .astype(int)
        .max()
    )
    return f"{prefix}{last_number + 1}"


class AppendPredictionToXSingle(BaseEstimator, TransformerMixin, Generic[TEstimator]):
    """Append the prediction of the estimator to X.
    To use multiple estimators, use AppendPredictionToX instead."""

    estimator: TEstimator
    estimator_: TEstimator
    """The fitted estimator."""

    def __init__(self, estimator: TEstimator, *, concat: bool = True) -> None:
        """Append the prediction of the estimator to X.

        If pandas DataFrame is given, the prediction is added as a new column
        with the name "y_pred_{estimator.__class__.__name__}_{i}" if the prediction
        is 1D, or "y_pred_{estimator.__class__.__name__}_{i}_{column_name}" if the
        prediction is 2D.

        To use multiple estimators, use AppendPredictionToX instead.

        Parameters
        ----------
        estimator : Any
            The estimator to be wrapped.
        concat : bool, optional
            Whether to concatenate the prediction to X,
            by default True
        """
        self.estimator = estimator
        self.concat = concat

    def fit(self, X: Any, y: Any = None, **fit_params: Any) -> Self:
        """Fit the estimator."""
        self.estimator_ = clone(self.estimator).fit(X, y, **fit_params)
        return self

    def transform(self, X: TX, y: Any = None, **predict_params: Any) -> TX:
        """Append the prediction of the estimator to X.
        If pandas DataFrame is given, the prediction is added as a new column
        with the name "y_pred_{estimator.__class__.__name__}_{i}" if the prediction
        is 1D, or "y_pred_{estimator.__class__.__name__}_{i}_{column_name}" if the
        prediction is 2D."""
        check_is_fitted(self, "estimator_")
        y_pred = self.estimator_.predict(X, **predict_params)

        # concat the prediction X
        if isinstance(X, DataFrame):
            # pandas

            # add prefix
            prefix = f"y_pred_{self.estimator_.__class__.__name__}_"
            prefix = generate_new_prefix(X, prefix)
            if y_pred.ndim == 1:
                y_pred = Series(y_pred, index=X.index, name=prefix).to_frame()
            else:
                y_pred = DataFrame(y_pred, index=X.index)
                y_pred = y_pred.add_prefix(prefix + "_")
            y_pred.columns = y_pred.columns.str.replace(
                r"[^a-zA-Z0-9_]", "_", regex=True
            )

            # concatenate
            if self.concat:
                return concat([X, y_pred], axis=1)
            return y_pred
        else:
            # numpy
            if y_pred.ndim == 1:
                y_pred = y_pred.reshape(-1, 1)

            # concatenate
            if self.concat:
                return np.hstack([X, y_pred])
            return y_pred


class AppendPredictionToX(BaseEstimator, TransformerMixin, Generic[TEstimator]):
    """Append the prediction of the estimators to X."""

    estimators: Sequence[TEstimator]
    estimators_: Sequence[TEstimator]

    def __init__(
        self,
        estimators: Sequence[TEstimator] | TEstimator,
        *,
        concat: bool = True,
        n_jobs: int | None = -1,
    ) -> None:
        """Append the prediction of the estimators to X.

        If pandas DataFrame is given, the prediction is added as a new column
        with the name
        "y_pred_{estimator.__class__.__name__}_{i}_{estimator_index}"
        if the prediction is 1D, or
        "y_pred_{estimator.__class__.__name__}_{i}_{column_name}_{estimator_index}"
        if the prediction is 2D.

        Parameters
        ----------
        estimators : Sequence[TEstimator] | TEstimator
            The estimator(s) to be wrapped.
        concat : bool, optional
            Whether to concatenate the prediction to X,
            by default True
        n_jobs : int | None, optional
            The number of jobs to run in parallel, by default -1
        """
        if not isinstance(estimators, Sequence):
            estimators = [estimators]
        self.estimators = estimators
        self.concat = concat
        self.n_jobs = n_jobs

    def fit(self, X: Any, y: Any = None, **fit_params: Any) -> Self:
        """Fit the estimators."""
        estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(AppendPredictionToXSingle(clone(estimator), concat=False).fit)(
                X, y, **fit_params
            )
            for estimator in self.estimators
        )
        if estimators_ is None:
            raise RuntimeError("Failed to fit estimators")
        self.estimators_ = estimators_
        return self

    def transform(self, X: TX, y: Any = None, **predict_params: Any) -> TX:
        """Append the prediction of the estimators to X.

        If pandas DataFrame is given, the prediction is added as a new column
        with the name
        "y_pred_{estimator.__class__.__name__}_{i}_{estimator_index}"
        if the prediction is 1D, or
        "y_pred_{estimator.__class__.__name__}_{i}_{column_name}_{estimator_index}"
        if the prediction is 2D."""
        check_is_fitted(self, "estimators_")
        transformed: list[TX] = [
            estimator_.transform(X, y, **predict_params).add_suffix(f"_{i}")
            for i, estimator_ in enumerate(self.estimators_)
        ]
        if isinstance(X, DataFrame):
            if self.concat:
                return concat([X, *transformed], axis=1)
            return concat(transformed, axis=1)
        else:
            if self.concat:
                return np.hstack([X, *transformed])
            return np.hstack(transformed)
