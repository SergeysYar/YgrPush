from app.data.cleaner import clean_duration, is_valid_ph


def test_clean_duration_negative():
    assert clean_duration(-5) is None


def test_clean_duration_string():
    assert clean_duration("15") == 15.0


def test_is_valid_ph_inside_range():
    assert is_valid_ph("7.0") is True


def test_is_valid_ph_out_of_range():
    assert is_valid_ph("15") is False
