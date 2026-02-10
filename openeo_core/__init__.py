"""openeo-core – standalone Python library for openEO processes, data cubes, and ML models."""

from openeo_core.datacube import DataCube
from openeo_core.types import Cube, RasterCube, VectorCube

__all__ = [
    "DataCube",
    "Cube",
    "RasterCube",
    "VectorCube",
]

__version__ = "0.1.0"
