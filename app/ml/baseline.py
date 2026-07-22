from __future__ import annotations

import joblib
from typing import Any

from app.ml.base import QualityModel


class BaselineModel:
    def __init__(self) -> None:
        self.global_mean: float | None = None
        self.category_means: dict[str, float] = {}

    def fit(
        self,
        X: Any,
        y: Any,
        category: list[str] | None = None,
        sample_weight: list[float] | None = None,
    ) -> "BaselineModel":
        if sample_weight is None:
            self.global_mean = float(y.mean())
        else:
            total_weight = float(sum(sample_weight))
            self.global_mean = (
                float(sum(float(value) * float(weight) for value, weight in zip(y, sample_weight)) / total_weight)
                if total_weight > 0
                else float(y.mean())
            )
        if category is not None:
            grouped = {}
            for index, (cat, value) in enumerate(zip(category, y)):
                if cat is None:
                    continue
                weight = float(sample_weight[index]) if sample_weight is not None else 1.0
                grouped.setdefault(cat, []).append((float(value), weight))
            self.category_means = {
                cat: (
                    sum(value * weight for value, weight in values) / sum(weight for _, weight in values)
                    if sum(weight for _, weight in values) > 0
                    else self.global_mean
                )
                for cat, values in grouped.items()
            }
        return self

    def predict(self, X: Any, category: list[str] | None = None) -> list[float]:
        if self.global_mean is None:
            raise ValueError("BaselineModel must be fit before predict")
        if category is None:
            return [self.global_mean] * len(X)
        return [self.category_means.get(cat, self.global_mean) for cat in category]

    def save(self, path: str) -> None:
        joblib.dump({"global_mean": self.global_mean, "category_means": self.category_means}, path)

    def load(self, path: str) -> "BaselineModel":
        state = joblib.load(path)
        self.global_mean = state["global_mean"]
        self.category_means = state["category_means"]
        return self

    def explain(self, X: Any) -> dict[str, float]:
        return {"baseline_mean": float(self.global_mean) if self.global_mean is not None else 0.0}
