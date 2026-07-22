from __future__ import annotations

from typing import Any, Protocol


class QualityModel(Protocol):
    def fit(self, X: Any, y: Any) -> Any:
        ...

    def predict(self, X: Any) -> Any:
        ...

    def save(self, path: str) -> None:
        ...

    def load(self, path: str) -> None:
        ...

    def explain(self, X: Any) -> dict[str, float]:
        ...
