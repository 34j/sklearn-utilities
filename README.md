# Sklearn Utilities

<p align="center">
  <a href="https://github.com/34j/sklearn-utilities/actions/workflows/ci.yml?query=branch%3Amain">
    <img src="https://img.shields.io/github/actions/workflow/status/34j/sklearn-utilities/ci.yml?branch=main&label=CI&logo=github&style=flat-square" alt="CI Status" >
  </a>
  <a href="https://sklearn-utilities.readthedocs.io">
    <img src="https://img.shields.io/readthedocs/sklearn-utilities.svg?logo=read-the-docs&logoColor=fff&style=flat-square" alt="Documentation Status">
  </a>
  <a href="https://codecov.io/gh/34j/sklearn-utilities">
    <img src="https://img.shields.io/codecov/c/github/34j/sklearn-utilities.svg?logo=codecov&logoColor=fff&style=flat-square" alt="Test coverage percentage">
  </a>
</p>
<p align="center">
  <a href="https://python-poetry.org/">
    <img src="https://img.shields.io/badge/packaging-poetry-299bd7?style=flat-square&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAASCAYAAABrXO8xAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAJJSURBVHgBfZLPa1NBEMe/s7tNXoxW1KJQKaUHkXhQvHgW6UHQQ09CBS/6V3hKc/AP8CqCrUcpmop3Cx48eDB4yEECjVQrlZb80CRN8t6OM/teagVxYZi38+Yz853dJbzoMV3MM8cJUcLMSUKIE8AzQ2PieZzFxEJOHMOgMQQ+dUgSAckNXhapU/NMhDSWLs1B24A8sO1xrN4NECkcAC9ASkiIJc6k5TRiUDPhnyMMdhKc+Zx19l6SgyeW76BEONY9exVQMzKExGKwwPsCzza7KGSSWRWEQhyEaDXp6ZHEr416ygbiKYOd7TEWvvcQIeusHYMJGhTwF9y7sGnSwaWyFAiyoxzqW0PM/RjghPxF2pWReAowTEXnDh0xgcLs8l2YQmOrj3N7ByiqEoH0cARs4u78WgAVkoEDIDoOi3AkcLOHU60RIg5wC4ZuTC7FaHKQm8Hq1fQuSOBvX/sodmNJSB5geaF5CPIkUeecdMxieoRO5jz9bheL6/tXjrwCyX/UYBUcjCaWHljx1xiX6z9xEjkYAzbGVnB8pvLmyXm9ep+W8CmsSHQQY77Zx1zboxAV0w7ybMhQmfqdmmw3nEp1I0Z+FGO6M8LZdoyZnuzzBdjISicKRnpxzI9fPb+0oYXsNdyi+d3h9bm9MWYHFtPeIZfLwzmFDKy1ai3p+PDls1Llz4yyFpferxjnyjJDSEy9CaCx5m2cJPerq6Xm34eTrZt3PqxYO1XOwDYZrFlH1fWnpU38Y9HRze3lj0vOujZcXKuuXm3jP+s3KbZVra7y2EAAAAAASUVORK5CYII=" alt="Poetry">
  </a>
  <a href="https://github.com/ambv/black">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square" alt="black">
  </a>
  <a href="https://github.com/pre-commit/pre-commit">
    <img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white&style=flat-square" alt="pre-commit">
  </a>
</p>
<p align="center">
  <a href="https://pypi.org/project/sklearn-utilities/">
    <img src="https://img.shields.io/pypi/v/sklearn-utilities.svg?logo=python&logoColor=fff&style=flat-square" alt="PyPI Version">
  </a>
  <img src="https://img.shields.io/pypi/pyversions/sklearn-utilities.svg?style=flat-square&logo=python&amp;logoColor=fff" alt="Supported Python versions">
  <img src="https://img.shields.io/pypi/l/sklearn-utilities.svg?style=flat-square" alt="License">
</p>

Utilities for scikit-learn.

## Installation

Install this via pip (or your favourite package manager):

```shell
pip install sklearn-utilities
```

## API

See [Docs](https://sklearn-utilities.readthedocs.io/en/latest/sklearn_utilities.html) for more information.

- `EstimatorWrapperBase`: base class for wrappers. Redirects all attributes which are not in the wrapper to the wrapped estimator.
- `DataFrameWrapper`: tries to convert every estimator output to a pandas DataFrame or Series.
- `FeatureUnionPandas`: a `FeatureUnion` that works with pandas DataFrames.
- `IncludedColumnTransformerPandas`, `ExcludedColumnTransformerPandas`: select columns by name.
- `AppendPredictionToX`: appends the prediction of y to X.
- `AppendXPredictionToX`: appends the prediction of X to X.
- `DropByNoisePrediction`: drops columns which has high importance in predicting noise.
- `DropMissingColumns`: drops columns with missing values above a threshold.
- `DropMissingRowsY`: drops rows with missing values in y. Use `feature_engine.DropMissingData` for X.
- `IntersectXY`: drops rows where the index of X and y do not intersect. Use with `feature_engine.DropMissingData`.
- `IdTransformer`: a transformer that does nothing.
- `RecursiveFitSubtractRegressor`: a regressor that recursively fits a regressor and subtracts the prediction from the target.
- `SmartMultioutputEstimator`: a `MultiOutputEstimator` that supports tuple of arrays in `predict()` and supports pandas `Series` and `DataFrame`.
- `until_event()`, `since_event()`: calculates the time since or until events (`Series[bool]`)
- `ComposeVarEstimator`: composes mean and std/var estimators.
- `DummyRegressorVar`: `DummyRegressor` that returns 1.0 for std/var.
- `TransformedTargetRegressorVar`: `TransformedTargetRegressor` with std/var support.
- `StandardScalerVar`: `StandardScaler` with std/var support.
- `EvalSetWrapper`, `CatBoostEvalSetWrapper`: wrapper that passes `eval_set` to `fit()` using `train_test_split()`. The latter shows progress bar (using tqdm) as well. Useful for early stopping. For LightGBM, see [lightgbm-callbacks](https://github.com/34j/lightgbm-callbacks).

### `sklearn_utilities.dataset`

- `add_missing_values()`: adds missing values to a dataset.

### `sklearn_utilities.torch`

- `PCATorch`: faster PCA using PyTorch with GPU support.

#### `sklearn_utilities.torch.skorch`

- `SkorchReshaper`, `SkorchCNNReshaper`: reshapes X and y for `nn.Linear` and `nn.Conv1d/2d` respectively. (For `nn.Conv2d`, uses `np.sliding_window_view()`.)
- `AllowNaN`: wraps a loss module and assign 0 to y and y_hat for indices where y contains NaN in `forward()`..

## See also

- [ml-tooling/best-of-ml-python](https://github.com/ml-tooling/best-of-ml-python)

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- prettier-ignore-start -->
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- markdownlint-disable -->
<!-- markdownlint-enable -->
<!-- ALL-CONTRIBUTORS-LIST:END -->
<!-- prettier-ignore-end -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
