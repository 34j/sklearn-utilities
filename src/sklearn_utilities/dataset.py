from __future__ import annotations

import numpy as np
from numpy.random import RandomState
from pandas import DataFrame, Series

from .types import TX, TY


def add_missing_values(
    dataset: tuple[TX, TY],
    *,
    missing_rate_x: float = 0.5,
    missing_rate_y: float = 0.5,
    random_state: RandomState | int | None = None,
) -> tuple[TX, TY]:
    """Add missing values to a dataset.

    Parameters
    ----------
    dataset : tuple[TX, TY]
        The dataset to add missing values to.
    missing_rate_x : float, optional
        The rate of missing values to add to X, by default 0.5
    missing_rate_y : float, optional
        The rate of missing values to add to y, by default 0.5
    random_state : RandomState | int | None, optional
        The random state to use, by default None

    Returns
    -------
    tuple[TX, TY]
        The dataset with missing values added.
    """
    random_state = RandomState(random_state)
    X_full_, y_full_ = dataset
    if isinstance(X_full_, DataFrame):
        X_full = X_full_.to_numpy()
    else:
        X_full = X_full_
    if isinstance(y_full_, (DataFrame, Series)):
        y_full = y_full_.to_numpy()
    else:
        y_full = y_full_

    X_mask = random_state.rand(*X_full.shape) < missing_rate_x
    y_mask = random_state.rand(*y_full.shape) < missing_rate_y
    X_missing = X_full.copy()
    X_missing[X_mask] = np.nan
    y_missing = y_full.copy()
    y_missing[y_mask] = np.nan

    if isinstance(X_full_, DataFrame):
        X_missing = DataFrame(X_missing, columns=X_full_.columns, index=X_full_.index)
    if isinstance(y_full_, Series):
        y_missing = Series(y_missing, index=y_full_.index, name=y_full_.name)
    elif isinstance(y_full_, DataFrame):
        y_missing = DataFrame(y_missing, columns=y_full_.columns, index=y_full_.index)
    return X_missing, y_missing
