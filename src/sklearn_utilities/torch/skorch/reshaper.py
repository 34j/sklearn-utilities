from __future__ import annotations

import warnings
from typing import Any, Generic, TypeVar

import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view
from skorch import NeuralNet
from typing_extensions import Self

from sklearn_utilities.estimator_wrapper import EstimatorWrapperBase
from sklearn_utilities.types import TX, TY

TEstimator = TypeVar("TEstimator", bound=NeuralNet)


class SkorchReshaper(EstimatorWrapperBase[TEstimator], Generic[TEstimator]):
    """skorch wrapper that reshapes tabular data for NNs."""

    def __init__(self, estimator: TEstimator) -> None:
        """skorch wrapper that reshapes tabular data for NNs.
        X: [B, F] -> [B, F]
        y: [B] -> [B, 1] or [B, NY] -> [B, NY] where
        B: batch, F: features, NY: number of outputs

        Parameters
        ----------
        estimator : TEstimator
            The estimator to wrap.
        """
        super().__init__(estimator)
        if self.estimator.device == "cpu":
            warnings.warn(
                "You are using a CPU for training, which may be very slow. "
                f"Consider using a GPU. CUDA Availability: {torch.cuda.is_available()}",
                UserWarning,
            )

    def fit(self, X: TX, y: TY, **fit_params: Any) -> Self:
        # allow multioutput
        X_, y_ = self._validate_data(
            X,
            y,
            validate_separately=(
                {"force_all_finite": False, "allow_nd": True, "ensure_2d": False},
                {"force_all_finite": False, "allow_nd": True, "ensure_2d": False},
            ),
        )
        X_ = X_.astype(np.float32)
        y_ = y_.astype(np.float32)
        if y_.ndim == 1:
            y_ = np.expand_dims(y_, axis=1)
        self.y_ndim_ = y_.ndim
        self.estimator.fit(X_, y_, **fit_params)
        return self

    def predict(self, X: TX, **predict_params: Any) -> TY:
        X_: np.ndarray = self._validate_data(
            X, force_all_finite=False, allow_nd=True, ensure_2d=False
        )
        X_ = X_.astype(np.float32)
        y = self.estimator.predict(X_, **predict_params)
        if self.y_ndim_ == 1 and y.shape[1] == 1:
            if y.shape[1] == 1:
                y = y.squeeze(axis=1)
        return y


class SkorchCNNReshaper(EstimatorWrapperBase[TEstimator], Generic[TEstimator]):
    """skorch wrapper that reshapes tabular data for CNNs using sliding windows."""

    def __init__(self, estimator: TEstimator, *, window_size: int | None) -> None:
        """skorch wrapper that reshapes tabular data for CNNs using sliding windows.
        X: [B, F] -> [B - H + 1, 1, H, F] if window_size is not None (for Conv2d)
           [B, F] -> [B, 1, F] if window_size is None (for Conv1d)
        y: [B] -> [B - H + 1, 1] or [B, NY] -> [B - H + 1, NY] where
        C = 1: channels, B: batch, H: window, F: features, NY: number of outputs

        Parameters
        ----------
        estimator : TEstimator
            The estimator to wrap.
        window_size : int | None
            The size of the sliding window.
            Make sure that CNN kernel size is equal or larger than this.
            If None, no sliding window is applied.
        """
        self.estimator = estimator
        self.window_size = window_size
        if self.estimator.device == "cpu":
            warnings.warn(
                "You are using a CPU for training, which may be very slow. "
                f"Consider using a GPU. CUDA availability: {torch.cuda.is_available()}",
                UserWarning,
            )

    def fit(self, X: TX, y: TY, **fit_params: Any) -> Self:
        X_, y_ = self._validate_data(
            X,
            y,
            validate_separately=(
                {"force_all_finite": False, "allow_nd": True, "ensure_2d": False},
                {"force_all_finite": False, "allow_nd": True, "ensure_2d": False},
            ),
        )
        X_ = X_.astype(np.float32)
        y_ = y_.astype(np.float32)
        if self.window_size is not None:
            X_ = sliding_window_view(X_, self.window_size, axis=0)
            if X_.shape[0] != y_.shape[0] - self.window_size + 1:
                raise AssertionError(
                    f"X.shape[0] = {X_.shape[0]} "
                    "!= y.shape[0] - self.window_size + 1 = "
                    f"{y_.shape[0] - self.window_size + 1}"
                )
            y_ = y_[self.window_size - 1 :]
        X_ = np.expand_dims(X_, axis=1)
        self.y_ndim_ = y_.ndim
        if y_.ndim == 1:
            y_ = np.expand_dims(y_, axis=1)
        self.estimator.fit(X_, y_, **fit_params)
        return self

    def predict(self, X: TX, **predict_params: Any) -> TY:
        X_ = self._validate_data(
            X, force_all_finite=False, allow_nd=True, ensure_2d=False
        )
        X_ = X_.astype(np.float32)
        if self.window_size is not None:
            X_ = sliding_window_view(X_, self.window_size, axis=0)
        X_ = np.expand_dims(X_, axis=1)
        y = self.estimator.predict(X_, **predict_params)
        if self.y_ndim_ == 1:
            if y.shape[1] == 1:
                y = y.squeeze(axis=1)
        return y
