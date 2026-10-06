"""Core arithmetic and statistical helper functions for impara."""
from typing import Sequence


def add(a: float, b: float) -> float:
    """Return the sum of two numbers."""
    return a + b


def multiply(a: float, b: float) -> float:
    """Return the product of two numbers."""
    return a * b


def average(numbers: Sequence[float]) -> float:
    """Return the arithmetic mean of a non-empty sequence of numbers."""
    if not numbers:
        raise ValueError("average() requires at least one number")
    return sum(numbers) / len(numbers)