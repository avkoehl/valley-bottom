"""
align_flowlines
hand_and_basins
"""

import geopandas as gpd
import numpy as np
from shapely import Point
import networkx as nx

from remaster.streams_to_vector import vectorize_flowpaths
from remaster.utils.wbw_helper import (
    gpd_to_wbeVector,
    wbeRaster_to_rxr,
    rxr_to_wbeRaster,
)
from remaster.utils.geom import binary_raster_to_polygon
from remaster.streams_to_vector import vectorize_flowpaths


def calc_slope(dem, wbe):
    """
    Calculate slope using WhiteboxTools.
    """
    dem = rxr_to_wbeRaster(dem, wbe)
    slope = wbe.slope(dem)
    slope = wbeRaster_to_rxr(slope, wbe)
    return slope


def fill_depressions_with_retry(
    dem, wbe, fix_flats=True, flat_increment=None, max_depth=None, retry_counter=3
):
    attempt = 0
    last_exception = None

    while attempt < retry_counter:
        try:
            conditioned = wbe.fill_depressions(
                dem,
                fix_flats=fix_flats,
                flat_increment=flat_increment,
                max_depth=max_depth,
            )
            return conditioned
        except Exception as e:
            last_exception = e
            attempt += 1
            print(f"Attempt {attempt}/{retry_counter} failed. Retrying...")

    raise last_exception or Exception(
        "Failed to execute fill_depressions after multiple attempts"
    )


def align_flowlines(dem, flowlines, wbe, min_length):
    dem = rxr_to_wbeRaster(dem, wbe)
    conditioned = fill_depressions_with_retry(
        dem, wbe, fix_flats=True, flat_increment=None, max_depth=None
    )
    pointer = wbe.d8_pointer(conditioned)
    seed_points = _identify_source_nodes(flowlines, wbe)
    aligned_stream = wbe.trace_downslope_flowpaths(seed_points, pointer)
    aligned_flowlines = vectorize_flowpaths(aligned_stream, pointer, wbe, min_length)
    return aligned_flowlines


def hand_and_basins(dem, flowlines, wbe):
    dem = rxr_to_wbeRaster(dem, wbe)
    flowlines_wbe = gpd_to_wbeVector(flowlines, wbe)
    conditioned = fill_depressions_with_retry(
        dem, wbe, fix_flats=True, flat_increment=None, max_depth=None
    )
    pointer = wbe.d8_pointer(conditioned)
    flow_acc = wbe.d8_flow_accum(pointer, out_type="cells", input_is_pointer=True)
    stream = wbe.rasterize_streams(flowlines_wbe, dem, use_feature_id=True)
    hand = wbe.elevation_above_stream(conditioned, stream)
    pour_points = _identify_pour_points(stream, flow_acc, wbe)
    basins = wbe.watershed(pointer, pour_points)
    basins = label_basins(basins, flowlines, wbe)
    hand, basins = wbeRaster_to_rxr(hand, wbe), wbeRaster_to_rxr(basins, wbe)
    return hand, basins


def label_basins(basins, flowlines, wbe):
    # Label basins with the stream ID
    basins = wbeRaster_to_rxr(basins, wbe)
    new_basins = basins.copy()

    for basin_id in np.unique(basins.data):
        if basin_id == 0:
            continue
        if not np.isfinite(basin_id):
            continue

        mask = basins == basin_id
        basindf = binary_raster_to_polygon(mask, return_df=True)
        clipped = flowlines.clip(basindf)
        if clipped.empty:
            continue
        # get the stream ID of the clipped flowline with the biggest length
        clipped["olap"] = clipped.geometry.length
        clipped = clipped.sort_values("olap", ascending=False)
        stream_id = clipped.iloc[0].streamID

        # assign the stream ID to the basin
        new_basins.data[mask] = stream_id

    # convert back to wbeRaster
    new_basins = rxr_to_wbeRaster(new_basins, wbe)
    return new_basins


def _identify_source_nodes(flowlines, wbe):
    G = nx.DiGraph()
    for flowline in flowlines.geometry:
        start = flowline.coords[0]
        end = flowline.coords[-1]
        G.add_edge(start, end)

    source_nodes = []
    for node in G.nodes():
        if G.in_degree(node) == 0:
            source_nodes.append(Point(node))
    source_nodes = gpd.GeoSeries(source_nodes, crs=flowlines.crs)
    source_nodes = gpd_to_wbeVector(source_nodes, wbe)
    return source_nodes


def _identify_pour_points(stream, flow_acc, wbe):
    # simply return the stream cell with the highest flow accumulation
    stream_rxr = wbeRaster_to_rxr(stream, wbe)
    flow_acc_rxr = wbeRaster_to_rxr(flow_acc, wbe)

    pour_points = []
    for stream_val in np.unique(stream_rxr.data):
        if stream_val == 0:
            continue
        if np.isnan(stream_val):
            continue
        rows, cols = np.where(stream_rxr.data == stream_val)
        x_coords = stream_rxr.x.values[cols]
        y_coords = stream_rxr.y.values[rows]
        flow_acc_values = flow_acc_rxr.data[rows, cols]
        max_index = np.argmax(flow_acc_values)
        pour_points.append(Point(x_coords[max_index], y_coords[max_index]))

    pour_points = gpd.GeoSeries(pour_points, crs=stream_rxr.rio.crs)
    pour_points = gpd_to_wbeVector(pour_points, wbe)
    return pour_points
