import matplotlib.pyplot as plt
import geopandas as gpd
import rioxarray as rxr

from remaster.core import extract_valleyfloors
from remaster.config import Config

config = Config()

# dem = rxr.open_rasterio("./data/dem_180102080204.tif", masked=True).squeeze()
# flowlines = gpd.read_file("./data/flowlines.gpkg")

dem = rxr.open_rasterio("./data/dem_160502010301.tif", masked=True).squeeze()
flowlines = gpd.read_file("./data/nhd_160502010301.gpkg")

floors = extract_valleyfloors(dem, flowlines, config)
