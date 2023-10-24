from typing import Any

from sklearn.pipeline import Pipeline


class PipelineVar(Pipeline):
    """Pipeline that supports predict_var method"""

    def predict_var(self, X: Any, **predict_params: Any) -> Any:
        Xt = X
        for _, name, transform in self._iter(with_final=False):
            Xt = transform.transform(Xt)
        return self.steps[-1][-1].predict_var(Xt, **predict_params)
