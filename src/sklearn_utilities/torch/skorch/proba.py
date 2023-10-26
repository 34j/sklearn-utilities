from __future__ import annotations

from typing import Any, Generic, Literal, Sequence

import numpy as np
import torch
import torch.nn as nn

from ...types import TX, TY, TEstimator
from .reshaper import SkorchCNNReshaper, SkorchReshaper


class LNErrors(nn.Module):
    def __init__(self, n: int = 2) -> None:
        """Returns L^n errors (not mean of L^n errors).

        Parameters
        ----------
        n : int, optional
            The exponent, by default 2
        """
        super().__init__()
        self.n = n

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return torch.abs(y_true - y_pred).pow(self.n)


class AsymmetricLoss(nn.Module):
    def __init__(
        self,
        *,
        t: float,
        loss_pred_grater: nn.Module = LNErrors(1),
        loss_pred_less: nn.Module = LNErrors(1),
    ) -> None:
        super().__init__()
        self.loss_pred_grater = loss_pred_grater
        self.loss_pred_less = loss_pred_less
        self.t = t

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return torch.mean(
            torch.max(
                self.t * self.loss_pred_grater(y_pred, y_true),
                (1 - self.t) * self.loss_pred_less(y_pred, y_true),
            )
        )


def _reshape_y_pred(
    y_pred: torch.Tensor, y_true_shape: torch.Size, n_ts: int
) -> torch.Tensor:
    if y_pred.ndim == 2 and len(y_true_shape) == 2:
        # [B, Ts * Ny] -> [B, Ts, Ny]
        if y_pred.shape[1] == n_ts * y_true_shape[1]:
            y_pred = y_pred.reshape(y_pred.shape[0], n_ts, -1)
        else:
            raise ValueError(
                "Got y_pred.ndim == 2 and y_true.ndim == 2, but "
                f"y_pred.shape[1] = {y_pred.shape[1]} != "
                f"n_ts * y_true.shape[1] = {n_ts} * {y_true_shape[1]}"
            )
    return y_pred


def _get_ts_axis(y_pred_shape: torch.Size, y_true_shape: torch.Size) -> int:
    # ignore first axis (batch)
    y_pred_shape = y_pred_shape[1:]
    y_true_shape = y_true_shape[1:]

    # find the shape difference between y_pred and y_true
    # if Ts = Ny (y_true_shape[1] = y_true_shape[2]),
    # then ts_axis = -1
    ts_axis = -1
    for i, s in enumerate(y_pred_shape):
        if s not in y_true_shape:
            ts_axis = i + 1
            break
    return ts_axis


class AsymmetricLosses(nn.Module):
    """Asymmetric loss with multioutput support.

    Parameters
    ----------
    y_pred : torch.Tensor
        The predicted values.
        [B, Ts * Ny] or [B, Ts, Ny] or [B, Ny, Ts]
    y_true : torch.Tensor
        The true values.
        [B, Ny] or [B] (if Ny == 1)

    Returns
    -------
    torch.Tensor
        0-dim tensor with the loss."""

    def __init__(
        self, *, ts: Sequence[float] | int, loss: nn.Module = LNErrors(1)
    ) -> None:
        """Asymmetric loss with multioutput support.

        Parameters
        ----------
        ts : Sequence[float] | int
            The list of `t` to use for fitting the data or the number of `t` to use.
            If `ts` is an integer,
            `np.linspace(1 / (ts * 2), 1 - 1 / (ts * 2), ts)` is used.
        loss : nn.Module, optional
            The loss function to use, by default LNErrors(1)
        """
        super().__init__()
        if isinstance(ts, int):
            ts = list(np.linspace(1 / (ts * 2), 1 - 1 / (ts * 2), ts))
        self.ts = ts
        self.loss = loss
        self.losses = nn.ModuleList(
            [
                AsymmetricLoss(t=t, loss_pred_grater=loss, loss_pred_less=loss)
                for t in ts
            ]
        )

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred = _reshape_y_pred(y_pred, y_true.shape, len(self.ts))
        self.y_true_shape_ = y_true.shape
        self.ts_axis_ = _get_ts_axis(y_pred.shape, y_true.shape)

        # calculate losses
        losses = []
        for y_pred_q, loss in zip(
            torch.moveaxis(y_pred, self.ts_axis_, 0), self.losses
        ):
            losses.append(loss(y_pred_q, y_true))
        return torch.stack(losses).mean()


class SkorchReshaperProbaMixin(Generic[TEstimator]):
    """skorch wrapper mixin that converts quantile predictions to mean and std."""

    estimator: TEstimator
    m_type: Literal["mean", "median", "nanmean", "nanmedian"]
    """M-statistics type to return from `predict` by default"""
    var_type: Literal["var", "std", "ptp", "nanvar", "nanstd"]
    """Variance type to return from `predict` by default"""

    def __init__(
        self,
        estimator: TEstimator,
        *args: Any,
        m_type: Literal["mean", "median", "nanmean", "nanmedian"] = "mean",
        var_type: Literal["var", "std", "ptp", "nanvar", "nanstd"] = "std",
        **kwargs: Any,
    ) -> None:
        super().__init__(estimator, *args, **kwargs)  # type: ignore
        self.m_type = m_type
        self.var_type = var_type

    @staticmethod
    def _get(
        y: TY,
        type_: Literal[
            "mean",
            "median",
            "var",
            "std",
            "ptp",
            "nanmean",
            "nanmedian",
            "nanvar",
            "nanstd",
        ],
        *,
        axis: int,
    ) -> TY:
        if hasattr(np, type_):
            return getattr(np, type_)(y, axis=axis)
        else:
            raise ValueError(f"Unknown type: {type_}")

    def predict(
        self,
        X: TX,
        *,
        return_std: bool = False,
        type_: Literal[
            "mean",
            "median",
            "nanmean",
            "nanmedian",
            "var",
            "std",
            "ptp",
            "nanvar",
            "nanstd",
        ]
        | tuple[
            Literal["mean", "median", "nanmean", "nanmedian"],
            Literal["var", "std", "ptp", "nanvar", "nanstd"],
        ]
        | None = None,
        **predict_params: Any,
    ) -> TY | tuple[TY, TY]:
        ts_axis_ = self.estimator.criterion.ts_axis_
        y = super().predict(X, **predict_params)  # type: ignore
        y = _reshape_y_pred(
            y, self.estimator.criterion.y_true_shape_, len(self.estimator.criterion.ts)
        )

        if return_std or isinstance(type_, tuple):
            if isinstance(type_, str):
                type_tuple = (type_, self.var_type)
            elif type_ is None:
                type_tuple = (self.m_type, self.var_type)
            else:
                type_tuple = type_
            return self._get(y, type_tuple[0], axis=ts_axis_), self._get(
                y, type_tuple[1], axis=ts_axis_
            )
        return self._get(y, type_ or self.m_type, axis=ts_axis_)


class SkorchReshaperProba(  # type: ignore
    SkorchReshaperProbaMixin[TEstimator],
    SkorchReshaper[TEstimator],
    Generic[TEstimator],
):
    def __init__(
        self,
        estimator: TEstimator,
        m_type: Literal["mean", "median", "nanmean", "nanmedian"] = "mean",
        var_type: Literal["var", "std", "ptp", "nanvar", "nanstd"] = "std",
    ) -> None:
        """skorch wrapper that reshapes tabular data for NNs
        and supports quantile predictions.
        X: [B, F] -> [B, F]
        y: [B] -> [B, 1] or [B, NY] -> [B, NY] where
        B: batch, F: features, NY: number of outputs

        Use `AsymmetricLosses` for NeuralNet.criterion.

        `estimator.module` may return [B, Ts * NY] or [B, Ts, NY] or [B, NY, Ts].
        If NY = Ts, assumes that [B, NY, Ts] is returned.

        Parameters
        ----------
        estimator : TEstimator
            The estimator to wrap.
        m_type : Literal['mean', 'median', 'nanmean', 'nanmedian'], optional
            M-statistics type to return from `predict` by default, by default "mean"
        var_type : Literal['var', 'std', 'ptp', 'nanvar', 'nanstd'], optional
            Variance type to return from `predict` by default, by default "std"

        Examples
        --------
        >>> from sklearn_utilities.torch.skorch import SkorchReshaperProba, AsymmetricLosses
        >>> from skorch import NeuralNet
        >>> import torch.nn as nn
        >>> net = nn.Sequential(nn.LazyLinear(10), nn.GELU(), nn.LazyLinear(5))
        >>> est = SkorchReshaperProba(NeuralNet(module=net, criterion=AsymmetricLosses(ts=5)))
        >>> est.fit(X_train, Y_train)
        >>> y_pred, y_std = est.predict(X_test, return_std=True)
        """
        super().__init__(estimator, m_type=m_type, var_type=var_type)


class SkorchCNNReshaperProba(  # type: ignore
    SkorchReshaperProbaMixin[TEstimator],
    SkorchCNNReshaper[TEstimator],
    Generic[TEstimator],
):
    def __init__(
        self,
        estimator: TEstimator,
        *,
        window_size: int | None,
        m_type: Literal["mean", "median", "nanmean", "nanmedian"] = "mean",
        var_type: Literal["var", "std", "ptp", "nanvar", "nanstd"] = "std",
    ) -> None:
        """skorch wrapper that reshapes tabular data for CNNs using sliding windows
        and supports quantile predictions.
        X: [B, F] -> [B - H + 1, 1, H, F] if window_size is not None (for Conv2d)
           [B, F] -> [B, 1, F] if window_size is None (for Conv1d)
        y: [B] -> [B - H + 1, 1] or [B, NY] -> [B - H + 1, NY] where
        C = 1: channels, B: batch, H: window, F: features, NY: number of outputs

        Use `AsymmetricLosses` for NeuralNet.criterion.

        `estimator.module` may return [B, Ts * NY] or [B, Ts, NY] or [B, NY, Ts].
        If NY = Ts, assumes that [B, NY, Ts] is returned.

        Parameters
        ----------
        estimator : TEstimator
            The estimator to wrap.
        window_size : int | None
            The size of the sliding window.
            Make sure that CNN kernel size is equal or larger than this.
            If None, no sliding window is applied.
        m_type : Literal['mean', 'median', 'nanmean', 'nanmedian'], optional
            M-statistics type to return from `predict` by default, by default "mean"
        var_type : Literal['var', 'std', 'ptp', 'nanvar', 'nanstd'], optional
            Variance type to return from `predict` by default, by default "std"

        Examples
        --------
        >>> from sklearn_utilities.torch.skorch import SkorchCNNReshaperProba, AsymmetricLosses
        >>> from skorch import NeuralNet
        >>> import torch.nn as nn
        >>> net = nn.Sequential(nn.Conv1d(1, 10, 3), nn.GELU(), nn.Flatten(), nn.LazyLinear(5))
        >>> est = SkorchCNNReshaperProba(NeuralNet(module=net, criterion=AsymmetricLosses(ts=5)))
        >>> est.fit(X_train, Y_train)
        >>> y_pred, y_std = est.predict(X_test, return_std=True)
        """
        super().__init__(
            estimator, m_type=m_type, var_type=var_type, window_size=window_size
        )
