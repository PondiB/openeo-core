"""PyTorch model architectures for satellite image time series classification."""

from openeo_core.model.torch_models.tempcnn import TempCNN
from openeo_core.model.torch_models.lighttae import LightTAE

__all__ = ["TempCNN", "LightTAE"]
