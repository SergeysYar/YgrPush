from __future__ import annotations

import joblib
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.ml.base import QualityModel


class BayesianRidgeModel:
    def __init__(self) -> None:
        self.pipeline: Pipeline | None = None

    def fit(self, X, y, sample_weight=None) -> "BayesianRidgeModel":
        self.pipeline = Pipeline(
            [("scaler", StandardScaler()), ("bayesian", BayesianRidge())]
        )
        try:
            if sample_weight is not None:
                self.pipeline.fit(X, y, bayesian__sample_weight=sample_weight)
            else:
                self.pipeline.fit(X, y)
        except TypeError:
            self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        if self.pipeline is None:
            raise ValueError("BayesianRidgeModel is not trained")
        return self.pipeline.predict(X)

    def save(self, path: str) -> None:
        if self.pipeline is None:
            raise ValueError("BayesianRidgeModel is not trained")
        joblib.dump(self.pipeline, path)

    def load(self, path: str) -> "BayesianRidgeModel":
        self.pipeline = joblib.load(path)
        return self

    def explain(self, X):
        return {"model": "bayesian_ridge"}
