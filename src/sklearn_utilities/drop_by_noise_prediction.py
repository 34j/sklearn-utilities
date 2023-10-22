from __future__ import annotations

from numbers import Real
from typing import Any, Literal

import numpy as np
from lightgbm import LGBMRegressor
from numpy.random import RandomState
from numpy.typing import NDArray
from sklearn.base import clone
from sklearn.feature_selection import SelectFromModel
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval, StrOptions
from typing_extensions import Self


class DropByNoisePrediction(SelectFromModel):
    """Remove features based on their importance to a model's prediction of noise.

    "Unsupervised Learning by Predicting Noise"
    https://arxiv.org/pdf/1704.05310.pdf
    https://ar5iv.labs.arxiv.org/html/1704.05310

    "Neural Architecture Search with Random Labels"
    https://arxiv.org/abs/2101.11834
    https://ar5iv.labs.arxiv.org/html/2101.11834

    Original Implementation:
    https://gist.github.com/richmanbtc/075178cd0e6d15c4a251128068991d47
    """

    def __init__(
        self,
        estimator: Any | None = None,
        *,
        drop_rate: float = 0.1,
        distribution: Literal["uniform", "normal", "arange"] = "uniform",
        random_state: RandomState | int | None = None,
    ) -> None:
        """Remove features based on their importance to a model's prediction of noise.

        "Unsupervised Learning by Predicting Noise"
        https://arxiv.org/pdf/1704.05310.pdf
        https://ar5iv.labs.arxiv.org/html/1704.05310

        "Neural Architecture Search with Random Labels"
        https://arxiv.org/abs/2101.11834
        https://ar5iv.labs.arxiv.org/html/2101.11834

        Original Implementation:
        https://gist.github.com/richmanbtc/075178cd0e6d15c4a251128068991d47

        Parameters
        ----------
        estimator : Any, optional
            Estimator to use for feature importance.
            If None, uses a default LGBMestimator,
            by default None
        percentile : float, optional
            Percent of features to keep, by default 10
        distribution : Literal['uniform', 'normal', 'arange'], optional
            Distribution to use for the target, by default &quot;uniform&quot;
        """
        self._parameter_constraints.update(
            {
                "drop_rate": [Interval(Real, 0, 1, closed="both")],
                "distribution": [StrOptions({"uniform", "normal", "arange"})],
            }
        )
        estimator = (
            LGBMRegressor(n_jobs=-1, random_state=random_state)
            if estimator is None
            else clone(estimator)
        )
        self.drop_rate = drop_rate
        self.distribution = distribution
        self.random_state = random_state
        check_random_state(self.random_state)
        super().__init__(
            estimator=estimator,
            threshold=-np.inf,
            prefit=False,
            importance_getter=lambda x: -x.feature_importances_,
            max_features=lambda x: int(x.shape[1] * (1 - drop_rate)),
        )

    def _generate_y(self, X: NDArray[Any]) -> NDArray[Any]:
        self.random_state_ = check_random_state(self.random_state)
        if self.distribution == "arange":
            y = np.arange(X.shape[0])
        elif hasattr(self.random_state_, self.distribution):
            y = getattr(self.random_state_, self.distribution)(0, 1, size=X.shape[0])
        else:
            raise ValueError(f"Invalid distribution: {self.distribution}")
        return y

    def fit(self, X: Any, y: Any = None, **fit_params: Any) -> Self:
        y = self._generate_y(X)
        return super().fit(X, y, **fit_params)

    def fit_transform(self, X: Any, y: Any = None, **fit_params: Any) -> Any:
        y = self._generate_y(X)
        return super().fit_transform(X, y, **fit_params)
