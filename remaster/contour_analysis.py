import numpy as np
import pandas as pd
from xrspatial import slope as calc_slope

"""
analyze contours in the REM at a set interval (5 meters)

Look for changes between contours in mean slope, and in the area of the contour

where the floor transitions to the walls, the contours should closer together and their areas smaller

"""


def analyze_rem_contours(rem_raster, interval=5):
    slope_raster = calc_slope(rem_raster)
    # everything below 0 becomes 0
    min_elev = rem_raster.min().item()
    rem_raster = rem_raster.where(rem_raster >= 0, 0)
    max_elev = rem_raster.max().item()
    intervals = np.arange(0, max_elev + interval, interval)

    # Loop through intervals
    results = []
    for i in range(len(intervals) - 1):
        min_val = intervals[i]
        max_val = intervals[i + 1]

        # Create mask for this interval
        mask = (rem_raster >= min_val) & (rem_raster < max_val)

        # Skip if no pixels in this interval
        if not mask.any():
            continue

        # Get mean slope for this interval
        mean_slope = slope_raster.where(mask).mean().item()

        num_pixels = mask.sum().item()
        pixel_area = np.abs(
            rem_raster.rio.resolution()[0] * rem_raster.rio.resolution()[1]
        )
        area = num_pixels * pixel_area
        if i == 0:
            min_val = min_elev
        results.append(
            {
                "min": min_val,
                "max": max_val,
                "mean_slope": mean_slope,
                "area": area,
            }
        )

    # Create DataFrame
    df = pd.DataFrame(results)
    return df
