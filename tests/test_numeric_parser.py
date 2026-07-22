from app.data.numeric_parser import parse_numeric


def test_parse_numeric_with_dot():
    assert parse_numeric("1.23") == 1.23


def test_parse_numeric_with_comma():
    assert parse_numeric("1,23") == 1.23


def test_parse_numeric_none():
    assert parse_numeric(None) is None


def test_parse_numeric_non_numeric():
    assert parse_numeric("abc") is None
