from app.data.validator import DataValidator, ValidationRule


def test_validator_ph_valid():
    validator = DataValidator()
    is_valid, error = validator.validate_value("startpH", "7.0")
    assert is_valid
    assert error is None


def test_validator_ph_out_of_range_high():
    validator = DataValidator()
    is_valid, error = validator.validate_value("startpH", "15.0")
    assert not is_valid
    assert error is not None
    assert "above maximum" in error


def test_validator_ph_out_of_range_low():
    validator = DataValidator()
    is_valid, error = validator.validate_value("startpH", "-1.0")
    assert not is_valid
    assert error is not None
    assert "below minimum" in error


def test_validator_duration_negative():
    validator = DataValidator()
    is_valid, error = validator.validate_value("duration_minutes", "-5.0")
    assert not is_valid


def test_validator_missing_field():
    validator = DataValidator()
    is_valid, error = validator.validate_value("unknown_field", "10.0")
    assert is_valid  # Unknown fields are valid by default


def test_validator_row():
    validator = DataValidator()
    row = {"startpH": "7.0", "endpH": "15.0", "duration_minutes": "10"}
    errors = validator.validate_row(row)
    assert len(errors) > 0
    assert any("endpH" in str(e) for e in errors)
