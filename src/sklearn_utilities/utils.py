from .types import TXPandas, TYPandas


def intersect_X_y(X: TXPandas, y: TYPandas) -> tuple[TXPandas, TYPandas]:
    idx = X.index.intersection(y.index)
    return X.loc[idx], y.loc[idx]


def drop_X_y(X: TXPandas, y: TYPandas) -> tuple[TXPandas, TYPandas]:
    return intersect_X_y(X, y.dropna())
