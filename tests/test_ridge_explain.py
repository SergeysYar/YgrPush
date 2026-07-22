import pandas as pd

from app.ml.ridge import RidgeModel


def test_ridge_explain_returns_top_factors():
    X = pd.DataFrame(
        {
            "num_steps": [1.0, 2.0, 3.0, 4.0],
            "average_ph": [5.4, 5.6, 5.8, 6.0],
            "salt_steps": [0.0, 1.0, 1.0, 2.0],
        }
    )
    y = pd.Series([100.0, 120.0, 135.0, 160.0])

    model = RidgeModel(alpha=1.0)
    model.fit(X, y)

    explanation = model.explain(X.iloc[[0]])

    assert "top_factors" in explanation
    assert len(explanation["top_factors"]) > 0
    assert "feature" in explanation["top_factors"][0]
    assert "contribution" in explanation["top_factors"][0]

