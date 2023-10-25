import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from skorch import NeuralNet

from sklearn_utilities.torch.skorch.proba import (
    AsymmetricLoss,
    AsymmetricLosses,
    LNErrors,
    SkorchCNNReshaperProba,
    SkorchReshaperProba,
)

from .test_reshaper import X_test, X_train, Y_test, Y_train, y_test, y_train

batch = 4
multioutput = 2

n_ts = 5
ts = torch.linspace(0.1, 1, n_ts)


def test_ln_errors():
    errors = LNErrors(1)
    errors2 = nn.L1Loss(reduction="none")
    y_pred = torch.randn(batch)
    y_true = torch.randn(batch)
    loss_ = errors(y_pred, y_true)
    l2 = errors2(y_pred, y_true)
    assert torch.allclose(loss_, l2)


def test_asymmetric_loss():
    loss = AsymmetricLoss(t=0.5)
    y_pred = torch.randn(batch)
    y_true = torch.randn(batch)
    loss_ = loss(y_pred, y_true)
    assert loss_.shape == torch.Size([])


def test_asymmetric_losses():
    ts = torch.arange(0.1, 1, 0.1)
    loss = AsymmetricLosses(ts=ts)
    y_pred = torch.randn(batch, len(ts))
    y_true = torch.randn(batch)
    loss_ = loss(y_pred, y_true)
    assert loss_.shape == torch.Size([])


def test_asymmetric_losses2():
    ts = torch.arange(0.1, 1, 0.1)
    loss = AsymmetricLosses(ts=ts)
    y_pred = torch.randn(batch, len(ts))
    y_true = torch.randn(batch, 1)
    loss_ = loss(y_pred, y_true)
    assert loss_.shape == torch.Size([])


def test_asymmetric_losses_multioutput():
    ts = torch.arange(0.1, 1, 0.1)
    loss = AsymmetricLosses(ts=ts)
    y_pred = torch.randn(batch, multioutput, len(ts))
    y_true = torch.randn(batch, multioutput)
    loss_ = loss(y_pred, y_true)
    assert loss_.shape == torch.Size([])


def test_asymmetric_losses_multioutput_flat():
    loss = AsymmetricLosses(ts=ts)
    y_pred = torch.randn(batch, len(ts) * multioutput)
    y_true = torch.randn(batch, multioutput)
    loss_ = loss(y_pred, y_true)
    assert loss_.shape == torch.Size([])


def test_reshaper_multioutput():
    net = nn.Sequential(nn.LazyLinear(10), nn.GELU(), nn.LazyLinear(n_ts * 2))
    est = SkorchReshaperProba(NeuralNet(module=net, criterion=AsymmetricLosses(ts=ts)))
    est.fit(X_train, Y_train)
    y_pred, y_std = est.predict(X_test, return_std=True)
    mean_squared_error(Y_test, y_pred)


def test_cnn_reshaper():
    net = nn.Sequential(
        nn.LazyConv2d(2, 3), nn.GELU(), nn.Flatten(), nn.LazyLinear(n_ts)
    )
    est = SkorchCNNReshaperProba(
        NeuralNet(module=net, criterion=AsymmetricLosses(ts=n_ts)), window_size=5
    )
    est.fit(X_train, y_train)
    y_pred, y_std = est.predict(X_test, return_std=True)
    mean_squared_error(y_test[5 - 1 :], y_pred)
