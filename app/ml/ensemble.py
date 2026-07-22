from __future__ import annotations

from typing import Any


class EnsembleModel:
    def __init__(self) -> None:
        self.models: list[Any] = []

    def add_model(self, model: Any, weight: float = 1.0) -> None:
        self.models.append((model, weight))

    def predict(self, X: Any) -> list[float]:
        if not self.models:
            raise ValueError("No models in ensemble")
        predictions = [model.predict(X) for model, _ in self.models]
        weights = [weight for _, weight in self.models]
        result = None
        for preds, weight in zip(predictions, weights):
            if result is None:
                result = [float(x) * weight for x in preds]
            else:
                result = [sum(x) for x in zip(result, [float(val) * weight for val in preds])]
        total_weight = sum(weights)
        return [value / total_weight for value in result]
