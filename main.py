import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import rioxarray as rxr
from whitebox_workflows import WbEnvironment

from remaster.hydro import align_flowlines
from remaster.hydro import hand_and_basins

dem = rxr.open_rasterio("./data/dem_180102080204.tif", masked=True).squeeze()
flowlines = gpd.read_file("./data/flowlines.gpkg")
wbe = WbEnvironment()

aligned_flowlines = align_flowlines(dem, flowlines, wbe)
hand, basins = hand_and_basins(dem, aligned_flowlines, wbe)
