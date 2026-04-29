"""Tests for scalar openEO helpers (array_element, eq, median, or_)."""

import numpy as np
import pytest

from openeo_core.exceptions import (
    ArrayElementNotAvailable,
    ArrayElementParameterConflict,
    ArrayElementParameterMissing,
    ArrayNotLabeled,
)
from openeo_core.ops.scalar import array_element, eq, median, or_


@pytest.mark.parametrize(
    "x,y,kw,out",
    [
        (1, None, {}, None),
        (None, None, {}, None),
        (1, 1, {}, True),
        (1, "1", {}, False),
        (0, False, {}, False),
        (1.02, 1, {"delta": 0.01}, False),
        (-1, -1.001, {"delta": 0.01}, True),
        (115, 110, {"delta": 10}, True),
        ("Test", "test", {}, False),
        ("Test", "test", {"case_sensitive": False}, True),
        ("Ä", "ä", {"case_sensitive": False}, True),
        (
            "2018-01-01T00:00:00Z",
            "2018-01-01T00:00:00+00:00",
            {},
            False,
        ),
    ],
)
def test_eq_spec_examples(x: object, y: object, kw: dict, out: object) -> None:
    assert eq(x, y, **kw) is out


def test_eq_numbers_delta_none_float_int() -> None:
    assert eq(1, 1.0, delta=None) is True


def test_eq_numpy_scalar() -> None:
    assert eq(np.float64(115), np.int32(110), delta=10) is True


def test_eq_ndarray_rejected() -> None:
    with pytest.raises(TypeError, match="0-dimensional"):
        eq(np.ones(2), np.ones(2))


def test_eq_unsupported_type_raises() -> None:
    with pytest.raises(TypeError, match="eq operands"):
        eq([], [])


@pytest.mark.parametrize(
    "x,y,out",
    [
        (True, True, True),
        (False, False, False),
        (True, None, True),
        (None, True, True),
        (False, None, None),
        (None, False, None),
        (None, None, None),
    ],
)
def test_or_spec_examples(x: object, y: object, out: object) -> None:
    assert or_(x, y) is out


def test_or_numpy_bool() -> None:
    assert or_(np.bool_(False), np.bool_(True)) is True


def test_or_invalid_type_raises() -> None:
    with pytest.raises(TypeError, match="`or` operands"):
        or_(1, True)


@pytest.mark.parametrize(
    "data,ignore_nodata,out",
    [
        ([1, 3, 3, 6, 7, 8, 9], True, 6),
        ([1, 2, 3, 4, 5, 6, 8, 9], True, 4.5),
        ([-1, -0.5, None, 1], True, -0.5),
        ([-1, 0, None, 1], False, None),
        ([], True, None),
        ([None, None], True, None),
    ],
)
def test_median_spec_examples(data: list[object], ignore_nodata: bool, out: object) -> None:
    assert median(data, ignore_nodata=ignore_nodata) == out


def test_median_numpy_array() -> None:
    assert median(np.array([1, 2, 3])) == 2


def test_median_invalid_item_type_raises() -> None:
    with pytest.raises(TypeError, match="`median` data items"):
        median([1, "x", 3])


def test_median_invalid_bool_item_raises() -> None:
    with pytest.raises(TypeError, match="not boolean"):
        median([True, False])


@pytest.mark.parametrize(
    "data,index,out",
    [
        ([9, 8, 7, 6, 5], 2, 7),
        (["A", "B", "C"], 0, "A"),
    ],
)
def test_array_element_spec_examples(data: list[object], index: int, out: object) -> None:
    assert array_element(data, index=index) == out


def test_array_element_empty_return_nodata_true() -> None:
    assert array_element([], index=0, return_nodata=True) is None


def test_array_element_parameter_missing_raises() -> None:
    with pytest.raises(ArrayElementParameterMissing, match="requires either"):
        array_element([1, 2, 3])


def test_array_element_parameter_conflict_raises() -> None:
    with pytest.raises(ArrayElementParameterConflict, match="only allows"):
        array_element([1, 2, 3], index=1, label="a")


def test_array_element_label_not_supported_raises() -> None:
    with pytest.raises(ArrayNotLabeled, match="not a labeled array"):
        array_element([1, 2, 3], label="a")


def test_array_element_label_raises_even_when_return_nodata() -> None:
    with pytest.raises(ArrayNotLabeled, match="not a labeled array"):
        array_element([1, 2, 3], label="a", return_nodata=True)


def test_array_element_index_out_of_bounds_raises() -> None:
    with pytest.raises(ArrayElementNotAvailable, match="no element with the specified"):
        array_element([1, 2, 3], index=99, return_nodata=False)
