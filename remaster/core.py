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
from remaster.strahler import label_streams
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
    aligned_flowlines = align_flowlines(dem, flowlines, wbe)
    aligned_flowlines = label_streams(aligned_flowlines)
    aligned_flowlines = aligned_flowlines.loc[
        aligned_flowlines.length > config.min_reach_length
    ]

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

    floors = dem.copy()
    floors.data = np.zeros_like(dem.data)
    floors = floors.rio.write_nodata(0)
    floors = floors.astype(np.uint16)

    counter = 1
    total = len(reaches)
    logger.info(f"Starting valley delineation for {total} reaches")

    for _, reach in reaches.iterrows():
        current_progress = int((counter / total) * 100)
        reach_hand = hand.where(basins == reach["streamID"])
        threshold = select_threshold(reach["mean_slope"], reach["strahler"], config)

        floor_mask = reach_hand < threshold
        floors.data[floor_mask] = 1
        logger.debug(
            f"{reach['streamID']} (slope: {reach['mean_slope']:.2f}, order: {reach['strahler']}, hand: {threshold}) {current_progress}%"
        )
        counter += 1

    # post process
    floors = post_process_floor_mask(
        floors,
        slope,
        stream_mask(reaches.geometry.union_all(), dem),
        config.floor_max_slope,
        config.min_hole_to_keep_area,
    )

    # label with the basin id where basin id is remapped to label?
    # get the mapping for the streamID to the segment id
    mapping = {}
    for i, value in enumerate(reaches["streamID"]):
        mapping[value] = reaches["segmentID"].iloc[i]
    remap_func = np.vectorize(lambda x: mapping.get(x, x))
    remapped_basins = basins.copy()
    remapped_basins.data = remap_func(basins.data)
    floors_labeled = floors * remapped_basins
    floors_labeled = floors_labeled.where(floors_labeled > 0)
    floors_labeled = floors_labeled.rio.write_nodata(0)
    floors_labeled = floors_labeled.astype(np.uint16)

    end_time = time.time()
    elapsed_time = format_time_duration(end_time - start_time)
    logger.info(f"Valley floor extraction completed, execution time: {elapsed_time}")
    return floors, floors_labeled, remapped_basins


def select_threshold(mean_slope, strahler, config):
    """
    Select the threshold for valley floor extraction based on slope and Strahler order.

    Parameters
    ----------
    mean_slope : float
        The mean slope of the reach.
    strahler : int
        The Strahler order of the reach.
    config : Config
        Configuration object containing parameters for the extraction process.

    Returns
    -------
    int
        The selected threshold for valley floor extraction.
    """
    if mean_slope < config.low_gradient_threshold:  # low gradient
        if strahler >= 3:
            return config.hand_lg_strahler_threeplus
        elif strahler == 2:
            return config.hand_lg_strahler_two
        else:
            return config.hand_lg_strahler_one
    elif (  # medium gradient
        config.low_gradient_threshold <= mean_slope <= config.medium_gradient_threshold
    ):
        return config.hand_mg
    else:  # high gradient
        return config.hand_hg


def stream_mask(linestring, base):
    geom = mapping(linestring)
    stream_mask = base.rio.clip([geom], all_touched=True, drop=False)
    stream_mask = stream_mask > 0
    return stream_mask


def post_process_floor_mask(
    floors, slope, stream_mask, max_slope, min_hole_area
):  # apply slope threshold
    processed = floors.copy()
    processed.data[slope >= max_slope] = 0

    # remove small holes
    num_cells = min_hole_area / (slope.rio.resolution()[0] ** 2)
    processed.data = remove_small_holes(processed.data > 0, int(num_cells))

    # burnin stream_mask
    processed.data[stream_mask > 0] = 1

    # remove any areas that are not connected to the reach_mask
    processed.data = remove_disconnected_areas(processed.data, stream_mask)

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
