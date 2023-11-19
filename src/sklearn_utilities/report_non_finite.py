from logging import getLogger
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from pandas import DataFrame, Series, Timestamp
from sklearn.base import BaseEstimator, TransformerMixin
from typing_extensions import Self

from .types import TXPandas

LOG = getLogger(__name__)


class ReportNonFinite(BaseEstimator, TransformerMixin):
    """Report non-finite values in X or y."""

    def __init__(
        self,
        *,
        on_fit: bool = False,
        on_fit_y: bool = False,
        on_transform: bool = True,
        plot: bool = True,
        calc_corr: bool = False,
        callback: Callable[[dict[str, DataFrame | Series]], None] | None = None,
        callback_figure: Callable[[Figure], None]
        | None = lambda fig: Path("sklearn_utilities_info/ReportNonFinite").mkdir(  # type: ignore
            parents=True, exist_ok=True
        )
        or fig.savefig(
            Path("sklearn_utilities_info/ReportNonFinite")
            / f"{Timestamp.now().isoformat().replace(':', '-')}.png"
        ),
    ) -> None:
        """Report non-finite values in X or y.

        Parameters
        ----------
        on_fit : bool, optional
            Whether to report non-finite values in X during fit, by default False
        on_fit_y : bool, optional
            Whether to report non-finite values in y during fit, by default False
        on_transform : bool, optional
            Whether to report non-finite values in X during transform, by default True
        plot : bool, optional
            Whether to plot the report result, by default True
        calc_corr : bool, optional
            Whether to calculate the correlation of non-finite values, by default False
        callback : Callable[[dict[str, DataFrame  |  Series]], None] | None, optional
            The callback function, by default None
        callback_figure : _type_, optional
            The callback function for figure, by default
            `lambda fig:
            Path("sklearn-utilities/ReportNonFinite").mkdir(parents=True, exist_ok=True)
            or fig.savefig(Path("sklearn-utilities/ReportNonFinite") /
            f"{Timestamp.now().isoformat().replace(':', '-')}.png")`
        """
        self.on_fit = on_fit
        self.on_fit_y = on_fit_y
        self.on_transform = on_transform
        self.plot = plot
        self.calc_corr = calc_corr
        self.callback = callback
        self.callback_figure = callback_figure

    def fit(self, X: DataFrame, y: Any = None, **fit_params: Any) -> Self:
        if self.on_fit:
            try:
                self._report(X, "fit")
            except Exception as e:
                LOG.warning(f"Failed to report non-finite values in X during fit: {e}")
                LOG.exception(e)

        if self.on_fit_y:
            try:
                DataFrame(y)
            except Exception as e:
                LOG.warning(f"Failed to convert y to DataFrame during fit: {e}")
                LOG.exception(e)

            try:
                self._report(DataFrame(y), "fit_y")
            except Exception as e:
                LOG.warning(f"Failed to report non-finite values in y during fit: {e}")
                LOG.exception(e)
        return self

    def transform(self, X: TXPandas, y: Any = None, **fit_params: Any) -> TXPandas:
        if self.on_transform:
            try:
                self._report(X, "transform")
            except Exception as e:
                LOG.warning(
                    f"Failed to report non-finite values in X during transform: {e}"
                )
                LOG.exception(e)
        return X

    def _report(self, X: TXPandas, caller: str = "") -> TXPandas:
        """Report non-finite values in X.

        Parameters
        ----------
        X : TXPandas
            Input data.
        caller : str, optional
            The caller name used in the log message, by default "".

        Returns
        -------
        TXPandas
            Input data.
        """
        is_na = X.isna()
        is_inf = X.isin([np.inf, -np.inf])
        is_non_finite = is_na | is_inf

        d: dict[str, DataFrame | Series] = {
            "nan_rate_by_column": is_na.mean(),
            "inf_rate_by_column": is_inf.mean(),
            "nan_rate_by_row": is_na.mean(axis=1),
            "inf_rate_by_row": is_inf.mean(axis=1),
        }
        d = d | {
            "non_finite_rate_by_column": d["nan_rate_by_column"]
            + d["inf_rate_by_column"],
            "non_finite_rate_by_row": d["nan_rate_by_row"] + d["inf_rate_by_row"],
        }

        if self.calc_corr:
            d["nan_rate_corr_by_column"] = is_na.corr()
            d["inf_rate_corr_by_column"] = is_inf.corr()
            d["non_finite_rate_corr_by_column"] = is_non_finite.corr()

        LOG.info(f"Non-finite values in X during {caller}: {d}")

        if self.plot:
            import seaborn as sns

            fig, axes = plt.subplots(3, 3 if self.calc_corr else 2, figsize=(20, 10))
            fig.suptitle(f"Non-finite values in X during {caller}")
            d["nan_rate_by_column"].plot(
                ax=axes[0, 0],
                kind="bar",
                title="NaN rate By column",
                xlabel="column name",
                ylabel="NaN rate",
            )
            d["inf_rate_by_column"].plot(
                ax=axes[1, 0],
                kind="bar",
                title="Inf rate By column",
                xlabel="column name",
                ylabel="Inf rate",
            )
            d["non_finite_rate_by_column"].plot(
                ax=axes[2, 0],
                kind="bar",
                title="Non-finite rate By column",
                xlabel="column name",
                ylabel="Non-finite rate",
            )
            d["nan_rate_by_row"].plot(
                ax=axes[0, 1],
                kind="line",
                title="NaN rate By row",
                xlabel="row index",
                ylabel="NaN rate",
            )
            d["inf_rate_by_row"].plot(
                ax=axes[1, 1],
                kind="line",
                title="Inf rate By row",
                xlabel="row index",
                ylabel="Inf rate",
            )
            d["non_finite_rate_by_row"].plot(
                ax=axes[2, 1],
                kind="line",
                title="Non-finite rate By row",
                xlabel="row index",
                ylabel="Non-finite rate",
            )
            if self.calc_corr:
                sns.heatmap(
                    d["nan_rate_corr_by_column"], ax=axes[0, 2], vmin=-1, vmax=1
                )
                axes[0, 2].set_title("NaN rate Corr By column")
                sns.heatmap(
                    d["inf_rate_corr_by_column"], ax=axes[1, 2], vmin=-1, vmax=1
                )
                axes[1, 2].set_title("Inf rate Corr By column")
                sns.heatmap(
                    d["non_finite_rate_corr_by_column"], ax=axes[2, 2], vmin=-1, vmax=1
                )
                axes[2, 2].set_title("Non-finite rate Corr By column")

            # tight layout
            plt.tight_layout()

            # callback
            if self.callback_figure is not None:
                self.callback_figure(fig)

        if self.callback is not None:
            self.callback(d)
        return X
