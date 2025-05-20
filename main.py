import matplotlib.pyplot as plt
import geopandas as gpd
import rioxarray as rxr

from remaster.core import extract_valleyfloors
from remaster.config import Config

config = Config()

dem = rxr.open_rasterio(
    # "../eval/data/topo/1804001001_dem.tif",
    "./data/huc10/1710010205-dem10m.tif",
    masked=True,
).squeeze()
flowlines = gpd.read_file(
    ###    "../eval/data/topo/1804001001_flowlines.gpkg",
    "./data/huc10/1710010205-flowlinesmr.gpkg"
)


floors_bin, floors_labeled, basins = extract_valleyfloors(dem, flowlines, config)
