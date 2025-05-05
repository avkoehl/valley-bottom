import time

import numpy as np
from whitebox_workflows import WbEnvironment
from skimage.morphology import remove_small_holes
from skimage.morphology import label
from shapely.geometry import mapping
from loguru import logger

from remaster.hydro import calc_slope
from remaster.hydro import align_flowlines
from remaster.hydro import hand_and_basins
from remaster.reach import network_reaches
from remaster.extent import dem_to_graph
from remaster.extent import define_valley_extent
from remaster.contour_analysis import identify_rem_threshold
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
    logger.debug("Compute slope")
    slope = calc_slope(smooth_raster(dem, config.spatial_radius, config.sigma), wbe)
    logger.debug("Align Flowlines to dem")
    aligned_flowlines = align_flowlines(
        dem, flowlines, wbe, min_length=config.min_reach_length
    )

    logger.debug("Split flowlines into reaches by shifts in slope")
    reaches = network_reaches(
        aligned_flowlines,
        dem,
        config.sample_distance,
        config.pelt_penalty,
        config.minsize,
    )
    logger.debug("Compute HAND and reach catchments")
    hand, basins = hand_and_basins(dem, reaches, wbe)
    logger.debug("Create cost distance graph")
    graph = dem_to_graph(dem.data, walls=np.isnan(dem.data))
    floors = dem.copy()
    floors.data = np.zeros_like(dem.data)
    floors = floors.rio.write_nodata(0)
    floors = floors.astype(np.uint8)

    counter = 1
    total = len(reaches)
    logger.info(f"Starting valley delineation for {total} reaches")

    for _, reach in reaches.iterrows():
        current_progress = int((counter / total) * 100)
        logger.info(
            f"Processing {reach['streamID']} {counter}/{total} ({current_progress}%)"
        )

        if reach["length"] < config.min_reach_length:
            continue

        if reach["mean_slope"] < config.low_gradient_threshold:  # low gradient
            extent = define_valley_extent(
                reach.geometry,
                graph,
                dem,
                cost_threshold=config.lg_max_extent,
                min_hole_to_keep_fraction=0.01,
            )
            reach_rem = compute_rem(
                reach.geometry,
                dem.where(extent),
                sample_distance=config.rem_sample_distance,
            )
            threshold = identify_rem_threshold(
                reach_rem.where(reach_rem > -5),
                slope.where(extent),
                config.lg_interval,
                config.lg_slope_threshold,
            )
            if threshold is None:
                threshold = config.lg_default_threshold
                logger.debug(f"Using default rem threshold: {threshold}")
            else:
                logger.debug(f"Identified rem threshold: {threshold}")
            floor_mask = reach_rem < threshold
        elif (  # medium gradient
            config.low_gradient_threshold
            <= reach["mean_slope"]
            <= config.medium_gradient_threshold
        ):
            hand_r = hand.where(basins == reach["streamID"])
            extent = hand_r < config.mg_max_extent
            threshold = identify_rem_threshold(
                hand.where(extent),
                slope.where(extent),
                config.mg_interval,
                config.mg_slope_threshold,
            )
            if threshold is None:
                threshold = config.mg_default_threshold
                logger.debug(f"Using default hand threshold: {threshold}")
            else:
                logger.debug(f"Identified hand threshold: {threshold}")
            floor_mask = hand_r < threshold
        else:  # high gradient
            hand_r = hand.where(basins == reach["streamID"])
            extent = hand_r < config.hg_max_extent
            threshold = identify_rem_threshold(
                hand.where(extent),
                slope.where(extent),
                config.hg_interval,
                config.hg_slope_threshold,
            )
            if threshold is None:
                threshold = config.hg_default_threshold
                logger.debug(f"Using default hand threshold: {threshold}")
            else:
                logger.debug(f"Identified hand threshold: {threshold}")
            floor_mask = hand_r < threshold

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
    processed = floors.copy()
    processed.data[slope >= max_slope] = 0

    # remove small holes
    num_cells = min_hole_area / (slope.rio.resolution()[0] ** 2)
    processed.data = remove_small_holes(processed.data > 0, int(num_cells))

    # burnin reach
    geom = mapping(reaches.geometry.union_all())
    copy = processed.copy().astype(np.float32)
    copy.data = np.ones_like(processed.data)
    reach_mask = copy.rio.clip([geom], all_touched=True, drop=False)
    reach_mask = reach_mask > 0
    processed.data[reach_mask] = 1

    # remove any areas that are not connected to the reach_mask
    processed.data = remove_disconnected_areas(processed.data, reach_mask)

    return processed.astype(np.uint8)


def remove_disconnected_areas(floor_mask_arr, reach_mask_arr):
    labeled = label(floor_mask_arr, connectivity=2)

    values = labeled[reach_mask_arr > 0]
    values = np.unique(values)
    values = values[np.isfinite(values)]

    labeled = np.where(
        np.isin(labeled, values, invert=True), 0, labeled
    )  # remove all other areas
    floor_mask_arr = np.where(labeled > 0, 1, 0)  # convert to binary mask
    floor_mask_arr = floor_mask_arr.astype(np.uint8)
    return floor_mask_arr
