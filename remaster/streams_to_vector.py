import numba
from numba.typed import Dict
import numpy as np
import geopandas as gpd
from shapely import LineString
from rasterio.transform import xy


from remaster.utils.wbw_helper import (
    wbeRaster_to_rxr,
)


def vectorize_flowpaths(stream, pointer, wbe, min_length):
    labeled_stream = wbe.stream_link_identifier(pointer, stream)
    link_class = wbe.stream_link_class(pointer, stream)
    link_class_rxr = wbeRaster_to_rxr(link_class, wbe)
    pointer_rxr = wbeRaster_to_rxr(pointer, wbe)
    labeled_stream_rxr = wbeRaster_to_rxr(labeled_stream, wbe)
    dirmap = generate_numba_friendly_dirmap()

    unique_streams = np.unique(labeled_stream_rxr)
    unique_streams = unique_streams[unique_streams > 0]

    flowlines = []
    for stream_id in unique_streams:
        pointer_arr = pointer_rxr.where(labeled_stream_rxr == stream_id).data
        link_class_arr = link_class_rxr.where(labeled_stream_rxr == stream_id).data
        path = trace_stream_flowpath(link_class_arr, pointer_arr, dirmap)
        if path is None:  # stream is too short
            continue
        xs, ys = xy(link_class_rxr.rio.transform(), path[:, 0], path[:, 1])
        linestring = LineString(zip(xs, ys))
        flowlines.append({"geometry": linestring, "streamID": stream_id})
    flowlines_gdf = gpd.GeoDataFrame(flowlines, crs=link_class_rxr.rio.crs)
    flowlines_gdf = flowlines_gdf[flowlines_gdf.geometry.length > min_length]
    return flowlines_gdf


def trace_stream_flowpath(link_class_arr, pointer_arr, dirmap):
    """
    Trace the flowpath from a start cell (source or junction) to an end cell (junction or sink).

    Args:
        link_class_arr: Array with labels for stream features (3=source, 4=junction, 5=sink) masked to a single stream
        pointer_arr: Array with flow direction pointer values masked to a single stream
        dirmap: Dictionary mapping pointer values to directional offsets

    Returns:
        np array of [rows,cols] coordinates of the flowpath
    """
    start_points = np.argwhere(np.logical_or(link_class_arr == 3, link_class_arr == 4))
    break_points_arr = np.logical_or(link_class_arr == 4, link_class_arr == 5).astype(
        np.int64
    )

    for point in start_points:
        current_row, current_col = point
        path = _trace_flowpath_numba(
            current_row, current_col, pointer_arr, dirmap, break_points_arr
        )
        if len(path) >= 1:
            return path


def generate_numba_friendly_dirmap():
    dirmap = {
        64: (-1, -1),  # up left
        128: (-1, 0),  # up
        1: (-1, 1),  # up right
        32: (0, -1),  # left
        0: (0, 0),  # stay (terminal cell)
        2: (0, 1),  # right
        16: (1, -1),  # down left
        8: (1, 0),  # down
        4: (1, 1),  # down right
    }
    dirmap_numba = Dict()  # numba typed dict
    for k, v in dirmap.items():
        dirmap_numba[np.float32(k)] = np.int64(v)
    return dirmap_numba


@numba.njit
def _trace_flowpath_numba(
    current_row, current_col, flow_dir_values, dirmap, break_points_arr
):
    nrows, ncols = flow_dir_values.shape
    path = [(current_row, current_col)]
    while True:
        current_direction = flow_dir_values[current_row, current_col]
        if current_direction == 0:
            break
        if np.isnan(current_direction):
            break

        drow, dcol = dirmap[current_direction]
        next_row = current_row + drow
        next_col = current_col + dcol

        if not (0 <= next_row < nrows and 0 <= next_col < ncols):
            break

        current_row, current_col = next_row, next_col
        path.append((next_row, next_col))

        if break_points_arr[next_row, next_col] == 1:
            break

    return np.array(path)
