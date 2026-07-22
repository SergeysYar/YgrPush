from __future__ import annotations

from enum import Enum
from typing import Final


class FeatureType(str, Enum):
    """Feature category types."""
    PROCESS_STEP = "process_step"
    TEMPERATURE = "temperature"
    PH = "ph"
    PE = "pe"
    FREQUENCY = "frequency"
    DURATION = "duration"
    COMPONENT = "component"
    PRODUCT = "product"
    DATA_QUALITY = "data_quality"


# List of numeric feature names by type
NUMERIC_FEATURES: Final[dict[str, list[str]]] = {
    FeatureType.PROCESS_STEP: [
        "num_steps",
        "num_loading_steps",
        "num_measurement_steps",
        "num_correction_steps",
    ],
    FeatureType.TEMPERATURE: [
        "first_temp",
        "last_temp",
        "avg_temp",
        "min_temp",
        "max_temp",
    ],
    FeatureType.PH: [
        "first_valid_ph",
        "last_valid_ph",
        "avg_ph",
        "min_ph",
        "max_ph",
        "max_ph_change",
    ],
    FeatureType.PE: [
        "avg_pe",
        "max_pe",
    ],
    FeatureType.FREQUENCY: [
        "avg_freq",
        "min_freq",
        "max_freq",
        "stirrer_cycles",
    ],
    FeatureType.DURATION: [
        "total_duration",
        "avg_step_duration",
        "max_step_duration",
        "invalid_durations_count",
    ],
    FeatureType.DATA_QUALITY: [
        "missing_sensor_readings",
        "suspicious_values_count",
    ],
}

# Categorical features
CATEGORICAL_FEATURES: Final[list[str]] = [
    "product_category",
    "product_base",
]

def get_all_feature_names() -> list[str]:
    """Get list of all numeric feature names."""
    names = []
    for feature_list in NUMERIC_FEATURES.values():
        names.extend(feature_list)
    return names


def get_features_by_type(feature_type: FeatureType) -> list[str]:
    """Get feature names for a specific type."""
    return NUMERIC_FEATURES.get(feature_type, [])
