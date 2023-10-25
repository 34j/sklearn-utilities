import numpy as np
import torch.nn as nn
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from skorch import NeuralNet

from sklearn_utilities.torch.skorch.reshaper import SkorchCNNReshaper, SkorchReshaper

X, y = load_diabetes(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)
Y_train, Y_test = np.array([y_train, y_train]).T, np.array([y_test, y_test]).T


def test_reshaper():
    net = nn.Sequential(nn.LazyLinear(10), nn.GELU(), nn.LazyLinear(1))
    est = SkorchReshaper(NeuralNet(module=net, criterion=nn.MSELoss()))
    est.fit(X_train, y_train)
    y_pred = est.predict(X_test)
    mean_squared_error(y_test, y_pred)


def test_reshaper_multioutput():
    net = nn.Sequential(nn.LazyLinear(10), nn.GELU(), nn.LazyLinear(2))
    est = SkorchReshaper(NeuralNet(module=net, criterion=nn.MSELoss()))
    est.fit(X_train, Y_train)
    y_pred = est.predict(X_test)
    mean_squared_error(Y_test, y_pred)


def test_cnn_reshaper():
    net = nn.Sequential(nn.LazyConv2d(2, 3), nn.GELU(), nn.Flatten(), nn.LazyLinear(1))
    est = SkorchCNNReshaper(
        NeuralNet(module=net, criterion=nn.MSELoss()), window_size=5
    )
    est.fit(X_train, y_train)
    y_pred = est.predict(X_test)
    mean_squared_error(y_test[5 - 1 :], y_pred)
