import time

import numpy as np
from xrspatial import slope as calc_slope
from whitebox_workflows import WbEnvironment
from skimage.morphology import remove_small_holes
from shapely.geometry import mapping
from loguru import logger

from remaster.hydro import align_flowlines
from remaster.hydro import hand_and_basins
from remaster.reach import network_reaches
from remaster.extent import dem_to_graph
from remaster.extent import define_valley_extent
from remaster.contour_analysis import analyze_rem_contours
from remaster.rem import compute_rem
from remaster.config import Config
from remaster.utils.smooth import smooth_raster
from remaster.utils.time import format_time_duration


def extract_valleyfloors(dem, flowlines, config=Config()):
    """
    Extract valley floors from a DEM using REM approach.

    Parameters
    ----------
    dem : xarray.DataArray
        The digital elevation model (DEM) data.
    flowlines : geopandas.GeoDataFrame
        The flowlines data.
    config : Config
        Configuration object containing parameters for the extraction process.

    Returns
    -------
    xarray.DataArray
        The extracted valley floors.
    """
    start_time = time.time()
    logger.info(
        f"Starting valley floor extraction with DEM shape: {dem.shape}, resolution: {dem.rio.resolution()}, and {len(flowlines)} flowlines"
    )
    logger.debug(f"Configuration parameters: {config.__dict__}")

    wbe = WbEnvironment()

    logger.info("Starting preprocessing steps (slope, flowlines, reaches, HAND, graph)")
    slope = calc_slope(smooth_raster(dem, config.spatial_radius, config.sigma))
    aligned_flowlines = align_flowlines(dem, flowlines, wbe)
    reaches = network_reaches(
        aligned_flowlines,
        slope,
        config.sample_distance,
        config.pelt_penalty,
        config.minsize,
    )
    hand, basins = hand_and_basins(dem, reaches, wbe)
    graph = dem_to_graph(dem.data, walls=np.isnan(dem.data))
    floors = dem.copy()
    floors.data = np.zeros_like(dem.data)
    floors = floors.rio.write_nodata(0)
    floors = floors.astype(np.uint8)

    # iterate through reaches
    counter = 1
    total = len(reaches)
    logger.info(f"Starting valley delineation for {total} reaches")
    for _, reach in reaches.iterrows():
        current_progress = int((counter / total) * 100)
        logger.debug(
            f"Processing reach {reach['streamID']} {counter}/{total} ({current_progress}%)"
        )
        if reach["mean_slope"] < config.gradient_threshold:
            floor_mask = process_low_gradient_reach(
                reach.geometry,
                dem,
                slope,
                graph,
                config.cost_threshold,
                config.lg_interval,
                config.lg_slope_threshold,
                config.lg_default_threshold,
            )
        else:
            reach_hand = hand.where(basins == reach["streamID"])
            floor_mask = process_high_gradient_reach(
                reach_hand,
                slope,
                config.hg_slope_threshold,
                config.hg_interval,
                config.hg_default_threshold,
            )

        floors.data[floor_mask] = 1
        counter += 1

    logger.info("Post-processing the valley floor mask")
    floors = post_process_floor_mask(
        floors,
        slope,
        reaches.geometry,
        config.floor_max_slope,
        config.min_hole_to_keep_area,
    )
    end_time = time.time()
    elapsed_time = format_time_duration(end_time - start_time)
    logger.info(f"Valley floor extraction completed, execution time: {elapsed_time}")
    return floors


def post_process_floor_mask(floors, slope, reaches, max_slope, min_hole_area):
    # apply slope threshold
    floors.data[slope >= max_slope] = 0

    # attach reach
    geom = mapping(reaches.geometry.union_all())
    copy = floors.copy().astype(np.float32)
    copy.data = np.ones_like(floors.data)
    reach_mask = copy.rio.clip([geom], all_touched=True, drop=False)
    floors.data[reach_mask > 0] = 1

    num_cells = min_hole_area / (slope.rio.resolution()[0] ** 2)
    floors.data = remove_small_holes(floors.data > 0, int(num_cells))
    return floors.astype(np.uint8)


def process_low_gradient_reach(
    reach,
    dem,
    slope,
    graph,
    cost_threshold,
    lg_interval,
    lg_slope_threshold,
    lg_default_threshold,
):
    extent = define_valley_extent(
        reach,
        graph,
        dem,
        cost_threshold=cost_threshold,
        min_hole_to_keep_fraction=0.01,
    )
    masked_dem = dem.where(extent)
    reach_rem = compute_rem(reach, masked_dem, sample_distance=10)
    df = analyze_rem_contours(reach_rem, slope, lg_interval)
    threshold = df[df["median_slope"] > lg_slope_threshold]["max"].min()
    if not np.isfinite(threshold):
        threshold = lg_default_threshold
    threshold = max(threshold, lg_interval)
    return reach_rem <= threshold


def process_high_gradient_reach(
    hand, slope, hg_slope_threshold, hg_interval, hg_default_threshold
):
    df = analyze_rem_contours(hand, slope, hg_interval)
    threshold = df[df["median_slope"] > hg_slope_threshold]["max"].min()
    if not np.isfinite(threshold):
        threshold = hg_default_threshold
    threshold = max(threshold, hg_interval)
    return hand <= threshold
