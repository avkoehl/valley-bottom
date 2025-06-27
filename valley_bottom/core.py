import time

import numpy as np
from whitebox_workflows import WbEnvironment
from skimage.morphology import remove_small_holes
from skimage.morphology import label
from shapely.geometry import mapping
from loguru import logger

from valley_bottom.hydro import calc_slope
from valley_bottom.hydro import align_flowlines
from valley_bottom.hydro import hand_and_basins
from valley_bottom.reach import network_reaches
from valley_bottom.strahler import label_streams
from valley_bottom.config import Config
from valley_bottom.utils.smooth import smooth_raster
from valley_bottom.utils.time import format_time_duration


def extract_valley_bottom(dem, flowlines, config=Config(), return_basins=False):
    """
    Extract valley bottoms from a DEM using height above nearest drainage (HAND) method.

    Parameters
    ----------
    dem : xarray.DataArray
        The digital elevation model (DEM) data.
    flowlines : geopandas.GeoDataFrame
        The flowlines data.
    config : Config
        Configuration object containing parameters for the extraction process.
    return_basins : bool
        Whether to return the basin labels.

    Returns
    -------
    xarray.DataArray
        The extracted valley bottom.
    """
    start_time = time.time()
    logger.info(
        f"Starting valley bottom extraction with DEM shape: {dem.shape}, resolution: {dem.rio.resolution()}, and {len(flowlines)} flowlines"
    )
    logger.debug(f"Configuration parameters: {config.__dict__}")

    wbe = WbEnvironment()

    logger.info("Starting preprocessing steps (slope, flowlines, reaches, HAND, graph)")
    logger.debug("Compute slope")
    slope = calc_slope(smooth_raster(dem, config.spatial_radius, config.sigma), wbe)
    logger.debug("Align Flowlines to dem")
    aligned_flowlines = align_flowlines(dem, flowlines, wbe)
    aligned_flowlines = label_streams(aligned_flowlines)

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

    bottoms = dem.copy()
    bottoms.data = np.zeros_like(dem.data)
    bottoms = bottoms.rio.write_nodata(0)
    bottoms = bottoms.astype(np.uint16)

    counter = 1
    total = len(reaches)
    logger.info(f"Starting valley delineation for {total} reaches")

    for _, reach in reaches.iterrows():
        current_progress = int((counter / total) * 100)
        reach_hand = hand.where(basins == reach["streamID"])
        threshold = select_threshold(reach["mean_slope"], reach["strahler"], config)

        bottom_mask = reach_hand < threshold
        bottoms.data[bottom_mask] = 1
        logger.debug(
            f"{reach['streamID']} (slope: {reach['mean_slope']:.2f}, order: {reach['strahler']}, hand: {threshold}) {current_progress}%"
        )
        counter += 1

    # post process
    bottoms = post_process_bottom_mask(
        bottoms,
        slope,
        stream_mask(reaches.geometry.union_all(), dem),
        config.bottom_max_slope,
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
    bottoms_labeled = bottoms * remapped_basins
    bottoms_labeled = bottoms_labeled.where(bottoms_labeled > 0)
    bottoms_labeled = bottoms_labeled.rio.write_nodata(0)
    bottoms_labeled = bottoms_labeled.astype(np.uint16)

    end_time = time.time()
    elapsed_time = format_time_duration(end_time - start_time)
    logger.info(f"Valley bottom extraction completed, execution time: {elapsed_time}")
    if return_basins:
        return bottoms_labeled, remapped_basins
    return bottoms_labeled


def select_threshold(mean_slope, strahler, config):
    """
    Select the threshold for valley bottom extraction based on slope and Strahler order.

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
        The selected threshold for valley bottom extraction.
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


def post_process_bottom_mask(
    bottoms, slope, stream_mask, max_slope, min_hole_area
):  # apply slope threshold
    processed = bottoms.copy()
    processed.data[slope >= max_slope] = 0

    # remove small holes
    num_cells = min_hole_area / (slope.rio.resolution()[0] ** 2)
    processed.data = remove_small_holes(processed.data > 0, int(num_cells))

    # burnin stream_mask
    processed.data[stream_mask > 0] = 1

    # remove any areas that are not connected to the reach_mask
    processed.data = remove_disconnected_areas(processed.data, stream_mask)

    return processed.astype(np.uint8)


def remove_disconnected_areas(bottom_mask_arr, reach_mask_arr):
    labeled = label(bottom_mask_arr, connectivity=2)

    values = labeled[reach_mask_arr > 0]
    values = np.unique(values)
    values = values[np.isfinite(values)]

    labeled = np.where(
        np.isin(labeled, values, invert=True), 0, labeled
    )  # remove all other areas
    bottom_mask_arr = np.where(labeled > 0, 1, 0)  # convert to binary mask
    bottom_mask_arr = bottom_mask_arr.astype(np.uint8)
    return bottom_mask_arr
