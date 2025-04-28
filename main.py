import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import rioxarray as rxr
from xrspatial import slope as calc_slope
from whitebox_workflows import WbEnvironment

from remaster.hydro import align_flowlines
from remaster.hydro import hand_and_basins
from remaster.reach import network_reaches

dem = rxr.open_rasterio("./data/dem_180102080204.tif", masked=True).squeeze()
slope = calc_slope(dem)
flowlines = gpd.read_file("./data/flowlines.gpkg")
wbe = WbEnvironment()

# SETUP
# align flowlines, segment reaches, hand and basins, cost graph
aligned_flowlines = align_flowlines(dem, flowlines, wbe)
reaches = network_reaches(
    aligned_flowlines, slope, sample_distance=10, penalty=10, minsize=100
)
hand, basins = hand_and_basins(dem, reaches, wbe)
