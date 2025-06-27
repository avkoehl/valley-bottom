import copy

import numpy as np
from scipy.ndimage import gaussian_filter


def smooth_raster(raster, spatial_radius, sigma):
    result = copy.deepcopy(raster)
    nan_msk = np.isnan(raster.data)
    radius_pixels = int(round(spatial_radius / raster.rio.resolution()[0]))
    # Apply a gaussian filter to the data
    result.data = _filter_gaussian_nan_conserve(raster.data, radius_pixels, sigma)
    # Set the nodata values back to NaN
    result.data[nan_msk] = np.nan
    return result


def _filter_gaussian_nan_conserve(arr, radius_pixels, sigma):
    """Apply a gaussian filter to an array with nans.
    modified from:
    https://stackoverflow.com/a/61481246

    Intensity is only shifted between not-nan pixels and is hence conserved.
    The intensity redistribution with respect to each single point
    is done by the weights of available pixels according
    to a gaussian distribution.
    All nans in arr, stay nans in gauss.
    """
    nan_msk = np.isnan(arr)

    loss = np.zeros(arr.shape)
    loss[nan_msk] = 1
    loss = gaussian_filter(
        loss, sigma=sigma, mode="constant", cval=1, radius=radius_pixels
    )

    gauss = arr.copy()
    gauss[nan_msk] = 0
    gauss = gaussian_filter(
        gauss, sigma=sigma, mode="constant", cval=0, radius=radius_pixels
    )
    gauss[nan_msk] = np.nan

    gauss += loss * arr
    return gauss
