import geopandas as gpd
import numpy as np
from shapely.geometry import LineString


def create_cross_sections(linestring, interval_distance, width, crs=None):
    # returns gpd.GeoSeries[gpd.LineString] for cross sections
    # index is the distance from the start of the linestring

    distances = np.arange(0, linestring.length + interval_distance, interval_distance)
    distances = distances[distances <= linestring.length]

    cross_sections = [
        _perpendicular_line(linestring, dist, width) for dist in distances
    ]
    series = gpd.GeoSeries(cross_sections, index=distances)
    if crs:
        series.crs = crs
    return series


def _points_on_either_side(linestring, distance, delta=1):
    left_delta = delta
    right_delta = delta

    if distance + delta > linestring.length:
        right_delta = linestring.length
    if distance - delta < 0:
        left_delta = 0

    left_point = linestring.interpolate(distance - left_delta)
    right_point = linestring.interpolate(distance + right_delta)
    return left_point, right_point


def _perpendicular_line(linestring, distance, width):
    point = linestring.interpolate(distance)
    left_point, right_point = _points_on_either_side(linestring, distance)

    angle = np.arctan2(right_point.y - left_point.y, right_point.x - left_point.x)
    angle = angle + np.pi / 2  # rotate 90 degrees

    start_x = point.x - width / 2 * np.cos(angle)
    start_y = point.y - width / 2 * np.sin(angle)
    end_x = point.x + width / 2 * np.cos(angle)
    end_y = point.y + width / 2 * np.sin(angle)

    return LineString([(start_x, start_y), (end_x, end_y)])
