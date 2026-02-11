"""Exceptions for the NDVI process (openEO spec)."""


class NirBandAmbiguous(Exception):
    """The NIR band can't be resolved, please specify the specific NIR band name."""


class RedBandAmbiguous(Exception):
    """The red band can't be resolved, please specify the specific red band name."""


class DimensionAmbiguous(Exception):
    """Dimension of type ``bands`` is not available or is ambiguous."""


class BandExists(Exception):
    """A band with the specified target name exists."""
