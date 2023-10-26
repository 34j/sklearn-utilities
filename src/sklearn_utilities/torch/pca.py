from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, TransformerMixin, check_is_fitted
from typing_extensions import Self


def svd_flip(
    u: torch.Tensor, v: torch.Tensor, u_based_decision: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sign correction to ensure deterministic output from SVD.

    Adjusts the columns of u and the rows of v such that the loadings in the
    columns in u that are largest in absolute value are always positive.

    Parameters
    ----------
    u : torch.Tensor
        Parameters u and v are the output of `linalg.svd` or
        :func:`~sklearn.utils.extmath.randomized_svd`, with matching inner
        dimensions so one can compute `np.dot(u * s, v)`.

    v : torch.Tensor
        Parameters u and v are the output of `linalg.svd` or
        :func:`~sklearn.utils.extmath.randomized_svd`, with matching inner
        dimensions so one can compute `np.dot(u * s, v)`.
        The input v should really be called vt to be consistent with scipy's
        output.

    u_based_decision : bool, default=True
        If True, use the columns of u as the basis for sign flipping.
        Otherwise, use the rows of v. The choice of which variable to base the
        decision on is generally algorithm dependent.

    Returns
    -------
    u_adjusted : torch.Tensor
        Array u with adjusted columns and the same dimensions as u.

    v_adjusted : torch.Tensor
        Array v with adjusted rows and the same dimensions as v.
    """
    if u_based_decision:
        max_abs_cols = torch.argmax(torch.abs(u), dim=0)
        signs = torch.sign(u[max_abs_cols, torch.arange(u.shape[1])])
    else:
        max_abs_rows = torch.argmax(torch.abs(v), dim=1)
        signs = torch.sign(v[torch.arange(v.shape[0]), max_abs_rows])
    u *= signs
    v *= signs[:, None]
    return u, v


def wrap_torch(func: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a function to convert all non-torch.Tensor arguments to torch.Tensor
    and convert the result to numpy.ndarray.

    Parameters
    ----------
    func : Callable[..., Any]
        The function to wrap.

    Returns
    -------
    Callable[..., Any]
        The wrapped function.
    """

    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        args_ = list(args)
        kwargs_ = dict(kwargs)
        for i, arg in enumerate(args_):
            if isinstance(arg, torch.Tensor):
                continue
            try:
                args_[i] = torch.as_tensor(arg, device=self.device, dtype=self.dtype)
            except TypeError:
                pass
        for k, v in kwargs_.items():
            if isinstance(v, torch.Tensor):
                continue
            try:
                kwargs_[k] = torch.as_tensor(v, device=self.device, dtype=self.dtype)
            except TypeError:
                pass

        result = func(self, *args_, **kwargs_)
        if isinstance(result, torch.Tensor):
            return result.detach().cpu().numpy()
        return result

    return wrapper


class PCATorch(nn.Module, BaseEstimator, TransformerMixin):
    """PCA using torch.linalg.svd.
    Might be faster than sklearn.decomposition.PCA
    if using GPU, but the algorithm is less efficient.

    If we could easily replace `np` with `torch` in sklearn...

    Attributes
    ----------
    mean_ : torch.Tensor
        The mean vector.
    components_ : torch.Tensor
        `Vt` where
        `X = U D Vt`
        `Y = X Vh`"""

    def __init__(
        self,
        n_components: int | None = None,
        *,
        qr: bool = False,
        device: torch.device | int | str = "cuda"
        if torch.cuda.is_available()
        else "cpu",
        dtype: torch.dtype = torch.float32,
        **kwargs: Any,
    ) -> None:
        """PCA using torch.linalg.svd.
        Might be faster than sklearn.decomposition.PCA
        if using GPU, but the algorithm is less efficient.

        If we could easily replace `np` with `torch` in sklearn...

        Parameters
        ----------
        n_components : int | None, optional
            Number of components to keep, by default None
        qr : bool, optional
            Whether to use QR decomposition, by default False
            If True, the result will be different from
            sklearn.decomposition.PCA
        device : torch.device | int | str, optional
            The device to use, by default
            `"cuda" if torch.cuda.is_available() else "cpu"`
        dtype : torch.dtype, optional
            The dtype to use, by default torch.float32

        Attributes
        ----------
        mean_ : torch.Tensor
            The mean vector.
        components_ : torch.Tensor
            `Vt` where
            `X = U D Vt`
            `Y = X Vh`"""
        super().__init__()
        self.n_components = n_components
        self.qr = qr
        self.device = device
        self.dtype = dtype
        self.kwargs = kwargs

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.transform(X)

    @wrap_torch
    @torch.no_grad()
    def fit(self, X: torch.Tensor) -> Self:
        _, d = X.size()
        if self.n_components is not None:
            d = min(self.n_components, d)
        self.register_buffer("mean_", X.mean(0, keepdim=True))
        Xc = X - self.mean_
        if self.qr:
            _, R = torch.linalg.qr(Xc)
            U, _, Vt = torch.linalg.svd(R, full_matrices=False)
        else:
            U, _, Vt = torch.linalg.svd(Xc, full_matrices=False)
        U, Vt = svd_flip(U, Vt)  # to be deterministic and consistent with sklearn
        self.register_buffer("components_", Vt[:d])
        return self

    @wrap_torch
    def transform(self, X: torch.Tensor) -> torch.Tensor:
        check_is_fitted(self, ["mean_", "components_"])
        Xc = X - self.mean_
        return torch.matmul(Xc, self.components_.t())  # V.T == inverse(V)

    @wrap_torch
    def inverse_transform(self, X: torch.Tensor) -> torch.Tensor:
        check_is_fitted(self, ["mean_", "components_"])
        return torch.matmul(X, self.components_) + self.mean_
