"""
Code for finding detrending the down valley trend of a DEM using a linestring and IDW
modified from: https://github.com/DahnJ/REM-xarray
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import xarray as xr
from scipy.spatial import cKDTree as KDTree

from remaster.utils.geom import coords_along_linestring


def compute_rem(linestring, dem, sample_distance):
    points = _trend_line(linestring, dem, sample_distance)
    coords = np.array([points.geometry.x, points.geometry.y]).T
    values = points["fit"].values

    c_x, c_y = [dem.coords[c].values for c in ("x", "y")]
    c_interpolated = np.dstack(np.meshgrid(c_x, c_y)).reshape(-1, 2)

    tree = KDTree(coords)
    distances, indices = tree.query(c_interpolated, k=5)

    weights = 1 / (distances + 1e-10)  # avoid division by zero
    weights = weights / weights.sum(axis=1).reshape(-1, 1)
    interpolated_values = (weights * values[indices]).sum(axis=1)

    trend_raster = xr.DataArray(
        interpolated_values.reshape((len(c_y), len(c_x))).T,
        dims=("x", "y"),
        coords={"x": c_x, "y": c_y},
    )

    rem = dem - trend_raster
    return rem


def _trend_line(line, dem, sample_distance):
    xs, ys = coords_along_linestring(line, sample_distance)
    elevation_series = dem.sel(x=xs, y=ys, method="nearest").values
    fit = _fit_elevations(elevation_series)
    df = pd.DataFrame(
        {
            "x": xs.values,
            "y": ys.values,
            "elevation": elevation_series,
            "fit": fit,
        }
    )
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.x, df.y),
        crs=dem.rio.crs,
    )
    return gdf


def _fit_elevations(elevation_series):
    x = np.arange(len(elevation_series))
    coeffs = np.polyfit(x, elevation_series, 2)  # 2nd order polynomial
    return np.polyval(coeffs, x)
