from typing import Any

import numpy as np
from sklearn.feature_selection import GenericUnivariateSelect
from typing_extensions import Self


class LooseGenericUnivariateSelect(GenericUnivariateSelect):
    """A GenericUnivariateSelect that does not require y
    and accepts missing values in X."""

    def fit(self, X: Any, y: Any = None) -> Self:
        y = np.ones(X.shape[0])
        return super().fit(X, y)

    def _validate_data(self, X: Any, y: Any = None, *args: Any, **kwargs: Any) -> Any:
        kwargs["force_all_finite"] = False
        kwargs.pop("multi_output", None)
        return super()._validate_data(X, y, *args, **kwargs)

    def _more_tags(self) -> dict[str, Any]:
        return {"requires_y": False, "allow_nan": True}


def DropMissingColumns(
    threshold_not_missing: float = 0.9,
) -> LooseGenericUnivariateSelect:
    """Drop columns with missing values above a threshold.

    Parameters
    ----------
    threshold_not_missing : float, optional
        If the ratio of non-missing values is below or equals to this threshold,
        the column is dropped, by default 0.9
    """
    if threshold_not_missing > 1:
        return LooseGenericUnivariateSelect(
            lambda X, y: np.isfinite(X).mean(axis=0),
            mode="k_best",
            param=threshold_not_missing,
        )
    return LooseGenericUnivariateSelect(
        lambda X, y: np.isfinite(X).mean(axis=0),
        mode="percentile",
        param=threshold_not_missing * 100,
    )


# class DropMissingColumns(BaseEstimator, SelectorMixin):
#     """Drop columns with missing values above a threshold."""
#     def __init__(self, threshold_not_missing: float = 0.5) -> None:
#         """Drop columns with missing values above a threshold.

#         Parameters
#         ----------
#         threshold_not_missing : float, optional
#             If the ratio of non-missing values is below this threshold,
#             the column is dropped, by default 0.5
#         """
#         self.threshold_not_missing = threshold_not_missing

#     def fit(self, X: Any, y: Any = None, **fit_params: Any) -> Any:
#         self.selector = ColumnTransformer(
#             [
#                 (
#                     "drop_missing_columns",
#                     "passthrough",
#                     lambda X: X.notna().mean(axis=0) >= self.threshold_not_missing,
#                 )
#             ]
#         )
#         self.selector.fit(X, y, **fit_params)
#         return self

#     def transform(self, X: Any, **transform_params: Any) -> Any:
#         return self.selector.transform(X, **transform_params)
