import pandas as pd

from app.features.batch_features import BatchFeatureBuilder


def test_batch_feature_builder_returns_extended_features():
    batch_df = pd.DataFrame(
        [
            {
                "id": 1,
                "loading_step_type": "water",
                "duration_minutes": 10.0,
                "startpH": 5.5,
                "endpH": 5.7,
                "startTemp": 25.0,
                "endTemp": 30.0,
                "startfreq": 100.0,
                "endfreq": 110.0,
                "component_1": 501,
                "mass_1": 12.0,
            },
            {
                "id": 2,
                "loading_step_type": "salt",
                "duration_minutes": 20.0,
                "startpH": 5.8,
                "endpH": 6.0,
                "startTemp": 31.0,
                "endTemp": 33.0,
                "startfreq": 120.0,
                "endfreq": 125.0,
                "component_1": 601,
                "mass_1": 4.0,
            },
        ]
    )

    features = BatchFeatureBuilder().build_full_batch_features(batch_df)

    assert features["num_steps"] == 2.0
    assert features["duration_sum"] == 30.0
    assert features["component_mass_total"] == 16.0
    assert features["water_steps"] == 1.0
    assert features["salt_steps"] == 1.0
    assert features["unique_components_count"] == 2.0
