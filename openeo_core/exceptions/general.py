"""General openEO process exceptions."""


class DimensionNotAvailable(Exception):
    """A dimension with the specified name does not exist."""


class UnitMismatch(Exception):
    """The unit of the spatial reference system does not match the expected unit."""


class IncompatibleDataCubes(Exception):
    """The data cube and the mask are incompatible, e.g. because of different dimensions or labels."""


class KernelDimensionsUneven(Exception):
    """Each dimension of the kernel must have an uneven number of elements."""
