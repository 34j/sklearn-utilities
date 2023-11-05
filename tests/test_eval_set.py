from catboost import CatBoostRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from sklearn_utilities.eval_set import CatBoostEvalSetWrapper


def test_eval_set():
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
