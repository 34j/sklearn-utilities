from typing import Any, Literal

from sklearn.discriminant_analysis import StandardScaler


class StandardScalerVar(StandardScaler):
    with_mean: bool

    def __init__(
        self,
        *,
        copy: bool = True,
        with_mean: bool = True,
        with_std: bool = True,
        var_type: Literal["std", "var"] = "var"
    ) -> None:
        super().__init__(copy=copy, with_mean=with_mean, with_std=with_std)
        self.var_type = var_type

    def inverse_transform(
        self, X: Any, copy: bool | None = None, return_std: bool = False
    ) -> Any:
        if return_std:
            if isinstance(X, tuple):
                return self.inverse_transform(
                    X[0], copy, False
                ), self.inverse_transform(X[1], copy, True)
            prev_with_mean = self.with_mean
            self.with_mean = False
            for i in range(1 if self.var_type == "std" else 2):
                X_scaled = super().inverse_transform(X, copy if i == 0 else None)
            self.with_mean = prev_with_mean
            return X_scaled
        return super().inverse_transform(X, copy)
