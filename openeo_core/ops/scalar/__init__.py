"""Scalar openEO process helpers."""

from openeo_core.ops.scalar.arrays import array_element
from openeo_core.ops.scalar.comparison import eq
from openeo_core.ops.scalar.logic import or_
from openeo_core.ops.scalar.statistics import median

__all__ = ["array_element", "eq", "median", "or_"]
