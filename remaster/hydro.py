"""
align_flowlines
hand_and_basins
"""

import geopandas as gpd
import numpy as np
from shapely import LineString
from shapely import Point

from remaster.utils.wbw_helper import (
    gpd_to_wbeVector,
    wbeRaster_to_rxr,
    rxr_to_wbeRaster,
)
from remaster.utils.geom import binary_raster_to_polygon


def vectorize_streams(stream, flow_acc, pointer, wbe, min_length):
    junctions = _identify_junction_nodes(pointer, stream, wbe)
    stream = wbeRaster_to_rxr(stream, wbe)
    flow_acc = wbeRaster_to_rxr(flow_acc, wbe)

    # get linestring for each unique stream value
    unique_streams = np.unique(stream.data)
    unique_streams = unique_streams[np.isfinite(unique_streams)]
    flowlines = []
    for stream_id in unique_streams:
        if stream_id == 0:
            continue
        rows, cols = np.where(stream.data == stream_id)
        x_coords = stream.x.values[cols]
        y_coords = stream.y.values[rows]
        flow_acc_values = flow_acc.data[rows, cols]
        sorted_indices = np.argsort(flow_acc_values)
        x_coords = x_coords[sorted_indices]
        y_coords = y_coords[sorted_indices]

        if len(x_coords) < 2:
            continue

        # add junction point if the last point is not a junction (or outlet)
        last_point = Point(x_coords[-1], y_coords[-1])
        closest_junction = junctions.distance(last_point).idxmin()
        closest_point = junctions.geometry.iloc[closest_junction]
        dist = closest_point.distance(last_point)

        if dist > 0 and dist < flow_acc.rio.resolution()[0] * 3:
            x_coords = np.append(x_coords, closest_point.x)
            y_coords = np.append(y_coords, closest_point.y)

        # add junction point if the first point is not a junction (or outlet)
        first_point = Point(x_coords[0], y_coords[0])
        closest_junction = junctions.distance(first_point).idxmin()
        closest_point = junctions.geometry.iloc[closest_junction]
        dist = closest_point.distance(first_point)

        if dist > 0 and dist < flow_acc.rio.resolution()[0] * 3:
            x_coords = np.insert(x_coords, 0, closest_point.x)
            y_coords = np.insert(y_coords, 0, closest_point.y)

        linestring = LineString(zip(x_coords, y_coords))

        flowlines.append({"streamID": stream_id, "geometry": linestring})

    flowlines = gpd.GeoDataFrame(flowlines, crs=stream.rio.crs)
    flowlines = flowlines[flowlines.length > min_length]
    return flowlines


def align_flowlines(dem, flowlines, wbe, min_length=15):
    # returns flowlines gpd.GeoSeries
    dem = rxr_to_wbeRaster(dem, wbe)
    flowlines = gpd_to_wbeVector(flowlines, wbe)

    conditioned = wbe.fill_depressions(
        dem, fix_flats=True, flat_increment=None, max_depth=None
    )
    pointer = wbe.d8_pointer(conditioned)
    flow_acc = wbe.d8_flow_accum(pointer, out_type="cells", input_is_pointer=True)

    stream = wbe.rasterize_streams(flowlines, dem, use_feature_id=True)
    seed_points = _identify_source_nodes(pointer, stream, wbe)
    # ^^ THIS DOESNT WORK IF NOT ALREADY ALIGNED!
    aligned_stream = wbe.trace_downslope_flowpaths(seed_points, pointer)
    labeled_stream = wbe.stream_link_identifier(pointer, aligned_stream)
    aligned_flowlines = vectorize_streams(
        labeled_stream, flow_acc, pointer, wbe, min_length=min_length
    )
    return aligned_flowlines


def hand_and_basins(dem, flowlines, wbe):
    dem = rxr_to_wbeRaster(dem, wbe)
    flowlines_wbe = gpd_to_wbeVector(flowlines, wbe)
    conditioned = wbe.fill_depressions(
        dem, fix_flats=True, flat_increment=None, max_depth=None
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


def _identify_source_nodes(pointer, stream, wbe):
    # Find source nodes (link class 3) in the stream raster
    link_class = wbe.stream_link_class(pointer, stream)
    link_class_rxr = wbeRaster_to_rxr(link_class, wbe)

    rows, cols = np.where(link_class_rxr.data == 3)
    x_coords = link_class_rxr.x.values[cols]
    y_coords = link_class_rxr.y.values[rows]
    geo_series = gpd.GeoSeries(
        gpd.points_from_xy(x_coords, y_coords), crs=link_class_rxr.rio.crs
    )
    sources = gpd_to_wbeVector(geo_series, wbe)
    return sources


def _identify_junction_nodes(pointer, stream, wbe, return_type="gpd"):
    # Find junction nodes (link class 4) in the stream raster
    link_class = wbe.stream_link_class(pointer, stream)
    link_class_rxr = wbeRaster_to_rxr(link_class, wbe)

    rows, cols = np.where(link_class_rxr.data == 4)
    x_coords = link_class_rxr.x.values[cols]
    y_coords = link_class_rxr.y.values[rows]
    geo_series = gpd.GeoSeries(
        gpd.points_from_xy(x_coords, y_coords), crs=link_class_rxr.rio.crs
    )
    if return_type == "gpd":
        return geo_series

    junctions = gpd_to_wbeVector(geo_series, wbe)
    return junctions


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
