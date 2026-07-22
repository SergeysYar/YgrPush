from __future__ import annotations

import joblib
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.ml.base import QualityModel


class PLSModel:
    def __init__(self) -> None:
        self.pipeline: Pipeline | None = None
        self.n_components: int | None = None

    def fit(self, X, y) -> "PLSModel":
        n_components = min(len(X) - 1, X.shape[1]) if len(X) > 1 else 0
        if n_components < 1:
            raise ValueError("Not enough batches to train PLS")
        self.n_components = n_components
        self.pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("pls", PLSRegression(n_components=n_components)),
            ]
        )
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        if self.pipeline is None:
            raise ValueError("PLSModel is not trained")
        return self.pipeline.predict(X)

    def save(self, path: str) -> None:
        if self.pipeline is None:
            raise ValueError("PLSModel is not trained")
        joblib.dump(self.pipeline, path)

    def load(self, path: str) -> "PLSModel":
        self.pipeline = joblib.load(path)
        return self

    def explain(self, X):
        return {"model": "pls", "n_components": self.n_components or 0}
