import pandas as pd
from app.data.component_normalizer import ComponentNormalizer, normalize_step_type


def test_normalize_step_type_water():
    assert normalize_step_type("вода") == "water"
    assert normalize_step_type("water") == "water"
    assert normalize_step_type("H2O") == "water"


def test_normalize_step_type_salt():
    assert normalize_step_type("соль") == "salt"
    assert normalize_step_type("nacl") == "salt"
    assert normalize_step_type("хлорид") == "salt"


def test_normalize_step_type_other():
    assert normalize_step_type(None) == "other"
    assert normalize_step_type("unknown_step") == "other"


def test_component_normalizer_empty_batch():
    normalizer = ComponentNormalizer()
    df = pd.DataFrame({
        "id": [1],
        "component_1": [None],
        "mass_1": [None]
    })
    result = normalizer.normalize_batch_components(df)
    assert result.empty


def test_component_normalizer_single_component():
    normalizer = ComponentNormalizer()
    df = pd.DataFrame({
        "id": [1, 2],
        "loading_step_type": ["water", "salt"],
        "component_1": [10, 20],
        "mass_1": [50.0, 75.0],
        "component_2": [None, None],
        "mass_2": [None, None]
    })
    result = normalizer.normalize_batch_components(
        df,
        component_lookup={
            10: {"name": "Water", "function_1": "water", "function_2": "", "function_3": ""},
            20: {"name": "Sodium Chloride", "function_1": "salt", "function_2": "", "function_3": ""},
        },
    )
    assert len(result) == 2
    assert result["component_id"].unique().tolist() == [10, 20]
    assert result["component_name"].tolist() == ["Water", "Sodium Chloride"]
    assert result["component_group"].tolist() == ["water", "salt"]
