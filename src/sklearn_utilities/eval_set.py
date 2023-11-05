from __future__ import annotations

import importlib.util
import warnings
from typing import Any, Generic, Literal

from numpy.random import RandomState
from sklearn.model_selection import train_test_split
from typing_extensions import Self

from .estimator_wrapper import EstimatorWrapperBase
from .types import TEstimator


class EvalSetWrapper(EstimatorWrapperBase[TEstimator], Generic[TEstimator]):
    """A wrapper that splits the data into train and validation sets and
    passes the validation set to `eval_set` parameter of the estimator."""

    def __init__(
        self,
        estimator: TEstimator,
        *,
        test_size: float | int | None = None,
        train_size: float | int | None = None,
        random_state: int | RandomState | None = None,
        shuffle: bool = True,
        stratify: bool = False,
        **kwargs: Any,
    ) -> None:
        """A wrapper that splits the data into train and validation sets and
        passes the validation set to `eval_set` parameter of the estimator.

        Parameters
        ----------
        estimator : Any
            The estimator to wrap.
        test_size : float or int, default=None
            If float, should be between 0.0 and 1.0 and represent the proportion
            of the dataset to include in the test split. If int, represents the
            absolute number of test samples. If None, the value is set to the
            complement of the train size. If ``train_size`` is also None, it will
            be set to 0.25.
            Alias: ``validation_fraction``
        train_size : float or int, default=None
            If float, should be between 0.0 and 1.0 and represent the
            proportion of the dataset to include in the train split. If
            int, represents the absolute number of train samples. If None,
            the value is automatically set to the complement of the test size.
            Alias: ``train_fraction``
        random_state : int, RandomState instance or None, default=None
            Controls the shuffling applied to the data before applying the split.
            Pass an int for reproducible output across multiple function calls.
            See :term:`Glossary <random_state>`.
        shuffle : bool, optional
            Whether or not to shuffle the data before splitting. If shuffle=False
            then stratify must be None. by default True
        stratify : bool, optional
            Whether or not stratify the data before splitting. If stratify=True,
            y must be categorical. by default False
        **kwargs : Any
            ``validation_fraction`` : alias for ``test_size`` (``sklearn`` style)
            ``train_fraction`` : alias for ``train_size`` (``sklearn`` style)
        """
        super().__init__(estimator)
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state
        self.shuffle = shuffle
        self.stratify = stratify
        self.kwargs = kwargs
        for key, value in kwargs.items():
            if key not in [
                "validation_fraction",
                "train_fraction",
            ]:
                warnings.warn(f"Unknown parameter: {key}: {value}")

    def fit(self, X: Any, y: Any, **fit_params: Any) -> Self:
        """Fit the estimator with `eval_set` set to the validation set.

        Parameters
        ----------
        X : Any
            The training input samples.
        y : Any
            The target values.

        Returns
        -------
        Self
            The fitted estimator.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size or self.kwargs.get("validation_fraction", None),
            train_size=self.train_size or self.kwargs.get("train_fraction", None),
            random_state=self.random_state,
            shuffle=self.shuffle,
            stratify=y if self.stratify else None,
        )
        fit_params = fit_params | {
            "eval_set": [(X_test, y_test)],
        }
        self.estimator.fit(X_train, y_train, **fit_params)
        return self


if importlib.util.find_spec("catboost") is not None:
    import re
    from typing import TypeVar

    import tqdm
    from catboost import CatBoost

    TCatBoost = TypeVar("TCatBoost", bound=CatBoost)

    class CatBoostEvalSetWrapper(EvalSetWrapper[TCatBoost], Generic[TCatBoost]):
        """A wrapper that splits the data into train and validation sets and
        passes the validation set to `eval_set` parameter of the estimator and
        shows the progress bar using `tqdm`.

        It is recommended to set `iterations` in `CatBoost.__init__` to show the
        progress bar.
        It is recommended to set `early_stopping_rounds` in `CatBoost.__init__`
        to enable early stopping."""

        def __init__(
            self,
            estimator: TCatBoost,
            *,
            test_size: float | int | None = None,
            train_size: float | int | None = None,
            random_state: int | RandomState | None = None,
            shuffle: bool = True,
            stratify: bool = False,
            tqdm_cls: Literal[
                "auto",
                "autonotebook",
                "std",
                "notebook",
                "asyncio",
                "keras",
                "dask",
                "tk",
                "gui",
                "rich",
                "contrib.slack",
                "contrib.discord",
                "contrib.telegram",
                "contrib.bells",
            ]
            | type[tqdm.std.tqdm] = "auto",
            tqdm_kwargs: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> None:
            """A wrapper that splits the data into train and validation sets and
            passes the validation set to `eval_set` parameter of the estimator and
            shows the progress bar using `tqdm`.

            It is recommended to set `iterations` in `CatBoost.__init__` to show the
            progress bar.
            It is recommended to set `early_stopping_rounds` in `CatBoost.__init__`
            to enable early stopping.

            Parameters
            ----------
            estimator : Any
                The estimator to wrap.
            test_size : float or int, default=None
                If float, should be between 0.0 and 1.0 and represent the proportion
                of the dataset to include in the test split. If int, represents the
                absolute number of test samples. If None, the value is set to the
                complement of the train size. If ``train_size`` is also None, it will
                be set to 0.25.
                Alias: ``validation_fraction``
            train_size : float or int, default=None
                If float, should be between 0.0 and 1.0 and represent the
                proportion of the dataset to include in the train split. If
                int, represents the absolute number of train samples. If None,
                the value is automatically set to the complement of the test size.
                Alias: ``train_fraction``
            random_state : int, RandomState instance or None, default=None
                Controls the shuffling applied to the data before applying the split.
                Pass an int for reproducible output across multiple function calls.
                See :term:`Glossary <random_state>`.
            shuffle : bool, optional
                Whether or not to shuffle the data before splitting. If shuffle=False
                then stratify must be None. by default True
            stratify : bool, optional
                Whether or not stratify the data before splitting. If stratify=True,
                y must be categorical. by default False
            tqdm_cls : Literal['auto', 'autonotebook', 'std', 'notebook', 'asyncio',
                'keras', 'dask', 'tk', 'gui', 'rich', 'contrib.slack', 'contrib.discord',
                'contrib.telegram', 'contrib.bells'] or type[tqdm.std.tqdm] or None, optional
                The tqdm class or module name, by default 'auto'
            tqdm_kwargs : dict[str, Any] or None, optional
                The keyword arguments passed to the tqdm class initializer
            **kwargs : Any
                ``validation_fraction`` : alias for ``test_size`` (``sklearn`` style)
                ``train_fraction`` : alias for ``train_size`` (``sklearn`` style)

            Examples
            --------
            >>> from catboost import CatBoostRegressor
            >>> from sklearn.datasets import make_regression
            >>> from sklearn_utilities import CatBoostEvalSetWrapper
            >>> X, y = make_regression()
            >>> # `iterations` is recommended to be set to show the progress bar.
            >>> # `early_stopping_rounds` should be set to enable early stopping.
            >>> estimator = CatBoostRegressor(iterations=1000, early_stopping_rounds=10)
            >>> estimator = CatBoostEvalSetWrapper(estimator)
            >>> estimator.fit(X, y)
            """
            super().__init__(
                estimator,
                test_size=test_size,
                train_size=train_size,
                random_state=random_state,
                shuffle=shuffle,
                stratify=stratify,
                **kwargs,
            )
            self.tqdm_cls = tqdm_cls
            self.tqdm_kwargs = tqdm_kwargs

            if isinstance(tqdm_cls, str):
                tqdm_module = importlib.import_module(f"tqdm.{tqdm_cls}")
                self.tqdm_cls_ = getattr(tqdm_module, "tqdm")
            else:
                self.tqdm_cls_ = tqdm_cls
            self.tqdm_kwargs_ = tqdm_kwargs or {}
            if "total" in self.tqdm_kwargs_:
                warnings.warn("'total' in tqdm_kwargs is ignored.", UserWarning)

        def fit(self, X: Any, y: Any, **fit_params: Any) -> Self:
            class ProgressBarPrint:
                def __init__(self_child) -> None:
                    self_child.pbar: tqdm.std.tqdm | None = None

                def write(self_child, text: str) -> None:
                    if self_child.pbar is None:
                        self_child.pbar = self.tqdm_cls_(
                            **(
                                self.tqdm_kwargs_
                                | {
                                    "total": self.estimator._get_params().get(
                                        "iterations", None
                                    ),
                                }
                            ),
                        )
                    try:
                        # 0:      learn: 221.7751345      test: 210.0125818       test1: 210.0125818
                        # best: 210.0125818 (0)   total: 196ms    remaining: 19.4s
                        # numbers = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", text)]
                        # do not start with test
                        text = re.sub(r"test\d*: ", "", text)
                        numbers = [
                            float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", text)
                        ]
                        n_iter = int(numbers[0]) + 1
                        scores = numbers[1:-4]
                        best_score = numbers[-4]
                        best_iter = int(numbers[-3])

                        self_child.pbar.set_postfix_str(
                            f"{', '.join(f'{score:g}' for score in scores)}"
                            f"{'=' if best_score == scores[1] else '<' if best_score > scores[1] else '>'}"  # noqa
                            f"{best_score:g}@{best_iter:g}it",
                        )
                        self_child.pbar.update(n_iter - self_child.pbar.n)
                    except Exception:
                        self_child.pbar.write(text)

            fit_params = fit_params | {
                "log_cout": ProgressBarPrint(),
            }
            return super().fit(X, y, **fit_params)

    if __name__ == "__main__":
        from catboost import CatBoostRegressor
        from sklearn.datasets import make_regression
        from sklearn.metrics import mean_squared_error

        X, y = make_regression(n_targets=2, random_state=0)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        estimator = CatBoostEvalSetWrapper(
            CatBoostRegressor(
                iterations=100,
                learning_rate=0.4,
                early_stopping_rounds=10,
                objective="MultiRMSE",
            ),
        )
        estimator.fit(X_train, y_train)
        mean_squared_error(y_test, estimator.predict(X_test))
