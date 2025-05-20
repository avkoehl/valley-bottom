from warnings import warn as warning

import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import Polygon
from shapely.geometry import shape
import xarray as xr


def coords_along_linestring(linestring, sample_distance):
    dists = [i for i in range(0, int(linestring.length), sample_distance)]
    dists.append(int(linestring.length))  # add the end point
    xs, ys = np.zeros(len(dists)), np.zeros(len(dists))
    for i, dist in enumerate(dists):
        point = linestring.interpolate(dist)
        xs[i] = point.x
        ys[i] = point.y

    return xr.DataArray(xs, dims="z"), xr.DataArray(ys, dims="z")


def vectorize_raster(raster):
    transform = raster.rio.transform()
    geoms = []
    values = []
    for geom, value in rasterio.features.shapes(raster.data, transform=transform):
        if np.isnan(value):
            continue
        if value <= 0:
            continue
        geoms.append(shape(geom))
        values.append(value)

    df = gpd.GeoDataFrame({"value": values, "geometry": geoms}, crs=raster.rio.crs)
    return df


def binary_raster_to_polygon(raster, return_df=False):
    # assume anywhere value is not zero or nan is a polygon
    image = np.where(raster.data != 0, 1, 0)
    image = image.astype(np.uint8)

    transform = raster.rio.transform()
    polygons = []
    for shape, value in rasterio.features.shapes(image, transform=transform):
        if value == 1:
            exterior = shape["coordinates"][0]
            if len(shape["coordinates"]) > 1:
                interior = shape["coordinates"][1:]
            else:
                interior = []

            polygons.append(Polygon(exterior, interior))

    df = gpd.GeoDataFrame(polygons, columns=["geometry"], crs=raster.rio.crs)
    if return_df:
        return df

    polygon = df.union_all().buffer(0.01)
    if isinstance(polygon, Polygon):
        return polygon
    else:
        # if returns multipolygon, explode it and return list of polygons
        if polygon.geom_type == "MultiPolygon":
            polygons = [poly for poly in polygon.geoms if poly.is_valid]
            if len(polygons) == 1:
                return polygons[0]
            else:
                warning("More than one polygon found, returning list of polygons")
                return polygons


def remove_holes(polygon, maxarea=None):
    if maxarea is None:
        return Polygon(polygon.exterior.coords)

    holes = [hole for hole in polygon.interiors if hole.area <= maxarea]
    return Polygon(polygon.exterior.coords, holes=holes)
