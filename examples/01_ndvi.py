from openeo_core import DataCube

# Load Sentinel-2 from AWS Earth Search over Albuquerque, NM
# Note: Earth Search uses common band names (red, nir), not Sentinel codes (B04, B08)
cube = DataCube.load_collection(
    "sentinel-2-l2a",
    spatial_extent={"west": -106.5, "south": 35.0, "east": -106.4, "north": 35.1},
    temporal_extent=("2023-06-01", "2023-06-30"),
    bands=["red", "nir"],
    properties={"eo:cloud_cover": {"lt": 50}},
)

print(f"Loaded cube: {cube.data.dims}, shape={cube.data.shape}")

# Fluent chaining: aggregate to monthly mean, then compute NDVI
result = (
    cube
    .aggregate_temporal(period="month", reducer="mean")
    .ndvi(nir="nir", red="red")
)

print(f"Monthly NDVI result: {result.data.dims}, shape={result.data.shape}")
print("Done! (data is dask-lazy, call .compute() to materialise)")
