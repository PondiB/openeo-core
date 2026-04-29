"""openeo-core – standalone Python library for openEO processes, data cubes, and ML models."""

from openeo_core.datacube import DataCube
from openeo_core.exceptions import (
    ArrayElementNotAvailable,
    ArrayElementParameterConflict,
    ArrayElementParameterMissing,
    ArrayNotLabeled,
    BandExists,
    DimensionAmbiguous,
    DimensionLabelCountMismatch,
    DimensionNotAvailable,
    IncompatibleDataCubes,
    KernelDimensionsUneven,
    NirBandAmbiguous,
    RedBandAmbiguous,
    UnitMismatch,
)
from openeo_core.types import Cube, RasterCube, VectorCube

__all__ = [
    "DataCube",
    "Cube",
    "RasterCube",
    "VectorCube",
    # openEO spec exceptions
    "ArrayElementNotAvailable",
    "ArrayElementParameterConflict",
    "ArrayElementParameterMissing",
    "ArrayNotLabeled",
    "BandExists",
    "DimensionAmbiguous",
    "DimensionLabelCountMismatch",
    "DimensionNotAvailable",
    "IncompatibleDataCubes",
    "KernelDimensionsUneven",
    "NirBandAmbiguous",
    "RedBandAmbiguous",
    "UnitMismatch",
]

__version__ = "0.3.0"
