import matplotlib.pyplot as plt
import geopandas as gpd
import rioxarray as rxr

from remaster.core import extract_valleyfloors
from remaster.config import Config

config = Config()

dem = rxr.open_rasterio(
    "/Users/arthurkoehl/programs/pasternack/valleyx/data/test_sites_10m/1805000203-dem.tif",
    masked=True,
).squeeze()
flowlines = gpd.read_file(
    "/Users/arthurkoehl/programs/pasternack/valleyx/data/test_sites_10m/1805000203-flowlines.shp"
)


floors = extract_valleyfloors(dem, flowlines, config)
floors.rio.to_raster("floors.tif")
