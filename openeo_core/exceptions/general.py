"""General openEO process exceptions."""


class DimensionNotAvailable(Exception):
    """A dimension with the specified name does not exist."""


class DimensionLabelCountMismatch(Exception):
    """A dimension cannot be dropped because it still has zero or multiple labels."""


class ArrayElementNotAvailable(Exception):
    """The array has no element with the specified index or label."""


class ArrayElementParameterMissing(Exception):
    """The array_element process requires either index or label parameter."""


class ArrayElementParameterConflict(Exception):
    """The array_element process only allows either index or label, not both."""


class ArrayNotLabeled(Exception):
    """A label was requested from an array that is not labeled."""


class UnitMismatch(Exception):
    """The unit of the spatial reference system does not match the expected unit."""


class IncompatibleDataCubes(Exception):
    """The data cube and the mask are incompatible, e.g. because of different dimensions or labels."""


class KernelDimensionsUneven(Exception):
    """Each dimension of the kernel must have an uneven number of elements."""
