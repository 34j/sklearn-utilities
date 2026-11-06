# Sklearn Utilities

<p align="center">
  <a href="https://github.com/34j/sklearn-utilities/actions/workflows/ci.yml?query=branch%3Amain">
    <img src="https://img.shields.io/github/actions/workflow/status/34j/sklearn-utilities/ci.yml?branch=main&label=CI&logo=github&style=flat-square" alt="CI Status" >
  </a>
  <a href="https://sklearn-utilities.readthedocs.io">
    <img src="https://img.shields.io/readthedocs/sklearn-utilities.svg?logo=read-the-docs&logoColor=fff&style=flat-square" alt="Documentation Status">
  </a>
</p>
<p align="center">
  <a href="https://codecov.io/gh/34j/sklearn-utilities">
    <img src="https://img.shields.io/codecov/c/github/34j/sklearn-utilities.svg?logo=codecov&logoColor=fff&style=flat-square" alt="Test coverage percentage">
  </a>
  <a href="https://github.com/astral-sh/uv">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json" alt="uv">
  </a>
  <a href="https://github.com/astral-sh/ruff">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff">
  </a>
  <a href="https://github.com/j178/prek">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/j178/prek/master/docs/assets/badge-v0.json" alt="prek">
  </a>
</p>
<p align="center">
  <a href="https://pypi.org/project/sklearn-utilities/">
    <img src="https://img.shields.io/pypi/v/sklearn-utilities.svg?logo=python&logoColor=fff&style=flat-square" alt="PyPI Version">
  </a>
  <img src="https://img.shields.io/pypi/pyversions/sklearn-utilities.svg?style=flat-square&logo=python&amp;logoColor=fff" alt="Supported Python versions">
  <img src="https://img.shields.io/pypi/l/sklearn-utilities.svg?style=flat-square" alt="License">
</p>

---

**Documentation**: <a href="https://sklearn-utilities.readthedocs.io" target="_blank">https://sklearn-utilities.readthedocs.io </a>

**Source Code**: <a href="https://github.com/34j/sklearn-utilities" target="_blank">https://github.com/34j/sklearn-utilities </a>

---

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
- `AppendPredictionToX`, `AppendXPredictionToX`: append predictions to X.
- `DropByNoisePrediction`: drops columns which have high importance in predicting noise.
- `DropMissingColumns`, `DropMissingRowsY`: drop columns or rows containing missing values.
- `IntersectXY`: drops rows where the index of X and y do not intersect.
- `ReindexMissingColumns`: reindexes columns of X in `transform()` to match the columns of X in `fit()`.
- `ReportNonFinite`: reports non-finite values in X and/or y.
- `IdTransformer`: a transformer that does nothing.
- `RecursiveFitSubtractRegressor`: recursively fits a regressor and subtracts its prediction from the target.
- `SmartMultioutputEstimator`: a multi-output estimator that supports tuple predictions and pandas objects.
- `until_event()`, `since_event()`: calculate the time since or until events.
- `ComposeVarEstimator`, `DummyRegressorVar`, `TransformedTargetRegressorVar`, `PipelineVar`, `StandardScalerVar`: estimators with variance support.
- `EvalSetWrapper`: passes `eval_set` to `fit()`, mainly for CatBoost.

### `sklearn_utilities.dataset`

- `add_missing_values()`: adds missing values to a dataset.

### `sklearn_utilities.torch`

- `PCATorch`: faster PCA using PyTorch with GPU support.

#### `sklearn_utilities.torch.skorch`

- `SkorchReshaper`, `SkorchCNNReshaper`: reshape X and y for common neural network layers.
- `AllowNaN`: wraps a loss module and handles NaN values in y.
