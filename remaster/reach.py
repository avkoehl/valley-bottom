"""
Code for segmenting lines into reaches based on slope
"""

import geopandas as gpd
import ruptures as rpt
from shapely.geometry import Point
from shapely.ops import substring
import numpy as np
import pandas as pd


from remaster.utils.geom import coords_along_linestring


def network_reaches(stream_network, slope_raster, sample_distance, penalty, minsize):
    # sample_distance = 10, # penalty = 10, minsize = 50
    reach_df_list = []
    for _, row in stream_network.iterrows():
        reaches = segment_reaches(
            row.geometry, slope_raster, sample_distance, penalty, minsize
        )
        reaches["streamID"] = row["streamID"] * 100 + reaches["reach_id"]
        reaches["length"] = reaches.geometry.length
        reach_df_list.append(reaches)

    reaches = pd.concat(reach_df_list, ignore_index=True)
    return reaches


def segment_reaches(linestring, slope_raster, sample_distance, penalty, minsize):
    xs, ys = coords_along_linestring(linestring, sample_distance)
    slope_values = slope_raster.sel(x=xs, y=ys, method="nearest").values

    if linestring.length < (sample_distance * minsize):
        return gpd.GeoDataFrame(
            {
                "geometry": [linestring],
                "mean_slope": [np.nanmean(slope_values)],
                "reach_id": [1],
            },
            geometry="geometry",
            crs=slope_raster.rio.crs,
        )

    signal = slope_values.reshape(-1, 1)
    model = rpt.Pelt(model="l2", min_size=minsize).fit(signal)
    cps = model.predict(pen=penalty)

    if len(cps) <= 1:
        return gpd.GeoDataFrame(
            {
                "geometry": [linestring],
                "mean_slope": [np.nanmean(slope_values)],
                "reach_id": [1],
            },
            geometry="geometry",
            crs=slope_raster.rio.crs,
        )

    interpolated_points = [Point(x, y) for x, y in zip(xs, ys)]

    # Create LineStrings based on the changepoints in the interpolated points
    linestrings = []
    linestring_slopes = []
    start_idx = 0
    for cp in cps[:-1]:
        start_point = interpolated_points[start_idx]
        end_point = interpolated_points[cp]
        start_distance = linestring.project(start_point)
        end_distance = linestring.project(end_point)
        segment = substring(linestring, start_distance, end_distance)
        segment_slopes = slope_values[start_idx:cp]
        mean_slope = np.nanmean(segment_slopes)
        linestring_slopes.append(mean_slope)
        linestrings.append(segment)
        start_idx = cp
    # Add the last segment
    start_point = interpolated_points[start_idx]
    end_point = interpolated_points[-1]
    start_distance = linestring.project(start_point)
    segment = substring(linestring, start_distance, linestring.length)
    segment_slopes = slope_values[start_idx:-1]
    mean_slope = np.nanmean(segment_slopes)
    linestrings.append(segment)
    linestring_slopes.append(mean_slope)

    gdf = gpd.GeoDataFrame(
        {
            "geometry": linestrings,
            "mean_slope": linestring_slopes,
            "reach_id": np.arange(1, len(linestrings) + 1),
        },
        geometry="geometry",
        crs=slope_raster.rio.crs,
    )

    return gdf
