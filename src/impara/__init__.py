"""impara: discover real, evidence-backed problems worth building.

Public API::

    from impara import discover, investigate

The arithmetic helpers from 0.2.0 are retained for backward compatibility.
"""
__version__ = "0.5.2"

from .operations import add, average, multiply
from .pipeline import discover, investigate

__all__ = [
    "add",
    "multiply",
    "average",
    "discover",
    "investigate",
    "__version__",
]
