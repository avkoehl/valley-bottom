import matplotlib.pyplot as plt
import geopandas as gpd
import rioxarray as rxr

from remaster.core import extract_valleyfloors
from remaster.config import Config

config = Config()

dem = rxr.open_rasterio(
    # "../eval/data/topo/1804001001_dem.tif",
    "../eval/data/topo/1807020302_dem.tif",
    masked=True,
).squeeze()
flowlines = gpd.read_file(
    ###    "../eval/data/topo/1804001001_flowlines.gpkg",
    "../eval/data/topo/1807020302_flowlines.gpkg",
)


floors = extract_valleyfloors(dem, flowlines, config)
floors.rio.to_raster("floors.tif")
