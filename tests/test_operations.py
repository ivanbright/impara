import pytest
from impara import add, multiply, average

def test_add():
    assert add(10, 25) == 35

def test_multiply():
    assert multiply(4, 5) == 20

def test_average():
    assert average([10, 20, 30, 40]) == 25.0

def test_average_empty():
    with pytest.raises(ValueError):
        average([])
