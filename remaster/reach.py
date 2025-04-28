"""
Code for segmenting lines into reaches based on slope
"""

import geopandas as gpd
import ruptures as rpt
from shapely.geometry import Point
from shapely.ops import substring


from remaster.utils.geom import coords_along_linestring


def segment_reaches(
    linestring, slope_raster, sample_distance=10, penalty=10, minsize=5
):
    if linestring.length < (minsize * sample_distance):
        return gpd.GeoSeries([linestring], crs=slope_raster.rio.crs)

    xs, ys = coords_along_linestring(linestring, sample_distance)
    slope = slope_raster.sel(x=xs, y=ys, method="nearest").values

    signal = slope.reshape(-1, 1)
    model = rpt.Pelt(model="l2", min_size=minsize).fit(signal)
    cps = model.predict(pen=penalty)

    if len(cps) <= 1:
        return gpd.GeoSeries([linestring], crs=slope_raster.rio.crs)

    interpolated_points = [Point(x, y) for x, y in zip(xs, ys)]

    # Create LineStrings based on the changepoints in the interpolated points
    linestrings = []
    start_idx = 0
    for cp in cps[:-1]:
        start_point = interpolated_points[start_idx]
        end_point = interpolated_points[cp]
        start_distance = linestring.project(start_point)
        end_distance = linestring.project(end_point)
        segment = substring(linestring, start_distance, end_distance)
        linestrings.append(segment)
        start_idx = cp
    # Add the last segment
    start_point = interpolated_points[start_idx]
    end_point = interpolated_points[-1]
    start_distance = linestring.project(start_point)
    segment = substring(linestring, start_distance, linestring.length)
    linestrings.append(segment)

    return gpd.GeoSeries(linestrings, crs=slope_raster.rio.crs)
