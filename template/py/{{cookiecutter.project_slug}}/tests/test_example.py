import pytest
from {{cookiecutter.package_name}} import add, divide, multiply, subtract


@pytest.mark.parametrize("a, b, expected", [
    (1, 2, 3),
    (0, 0, 0),
    (-1, -1, -2),
])
def test_add(a, b, expected):
    assert add(a, b) == expected


@pytest.mark.parametrize("a, b, expected", [
    (1, 2, -1),
    (0, 0, 0),
    (-1, -1, 0),
])
def test_subtract(a, b, expected):
    assert subtract(a, b) == expected


@pytest.mark.parametrize("a, b, expected", [
    (1, 2, 2),
    (0, 0, 0),
    (-1, -1, 1),
])
def test_multiply(a, b, expected):
    assert multiply(a, b) == expected


@pytest.mark.parametrize("a, b, expected", [
    (1, 2, 0.5),
    (-1, -1, 1),
])
def test_divide(a, b, expected):
    assert divide(a, b) == expected


@pytest.mark.parametrize("a, b", [
    (0, 0),
    (10, 0),
    (-10, 0),
])
def test_divide_by_zero(a, b):
    with pytest.raises(ValueError, match="Division by zero is not allowed."):
        divide(a, b)
