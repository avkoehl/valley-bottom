import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import rioxarray as rxr
from xrspatial import slope as calc_slope
from whitebox_workflows import WbEnvironment

from remaster.hydro import align_flowlines
from remaster.hydro import hand_and_basins
from remaster.reach import network_reaches
from remaster.extent import dem_to_graph
from remaster.extent import define_valley_extent
from remaster.contour_analysis import analyze_rem_contours
from remaster.rem import compute_rem

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
graph = dem_to_graph(dem.data, walls=np.isnan(dem.data))


# ITERATE THROUGH REACHES
floors = dem.copy()
floors.data = np.zeros_like(dem.data)
floors = floors.rio.write_nodata(0)
floors = floors.astype(np.uint8)

for _, reach in reaches.iterrows():
    # REM APPROACH ON FLAT GRADIENT REACHES
    # INTERVAL, SLOPE THRESHOLD, EXTENT, HOLE CLOSING
    if reach["mean_slope"] < 3:
        extent = define_valley_extent(
            reach["geometry"],
            graph,
            dem,
            cost_threshold=30,
            min_hole_to_keep_fraction=0.01,
        )
        masked_dem = dem.where(extent)
        reach_rem = compute_rem(
            reach.geometry, masked_dem, sample_distance=10, max_val=40, min_val=-10
        )
        df = analyze_rem_contours(reach_rem, interval=3)
        threshold = df[df["mean_slope"] > 7]["max"].min()
        if threshold == 0:
            threshold = 3
        if not np.isfinite(threshold):
            threshold = 20
        print(f"Low Gradient Reach {reach['streamID']} threshold: {threshold}")
        floors.data[reach_rem <= threshold] = 1
    else:
        # INTERVAL, SLOPE THRESHOLD, EXTENT, HOLE CLOSING
        # SLOPE OF DEM OR HAND?  -> SLOPE OF DEM
        # USE MEDIAN INSTEAD OF MEAN
        masked_hand = hand.where(basins == reach["streamID"])
        df = analyze_rem_contours(masked_hand, interval=3)
        threshold = df[df["mean_slope"] > 10]["max"].min()
        if threshold == 0:
            threshold = 3
        if not np.isfinite(threshold):
            threshold = 20
        print(f"High Gradient Reach {reach['streamID']} threshold: {threshold}")
        floors.data[masked_hand <= threshold] = 1
