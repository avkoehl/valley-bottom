import matplotlib.pyplot as plt
import geopandas as gpd
import rioxarray as rxr

from remaster.core import extract_valleyfloors
from remaster.config import Config

config = Config()

dem = rxr.open_rasterio(
    "./data/1802015301_dem.tif",
    masked=True,
).squeeze()
flowlines = gpd.read_file(
    "./data/1802015301_flowlines.gpkg",
)


floors = extract_valleyfloors(dem, flowlines, config)
floors.rio.to_raster("floors.tif")
