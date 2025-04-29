import numpy as np
import pandas as pd

"""
analyze contours in the REM at a set interval 

Look for changes between contours in mean slope, and in the area of the contour

where the floor transitions to the walls, the contours should closer together and their areas smaller

"""


def analyze_rem_contours(rem_raster, slope_raster, interval):
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
        median_slope = slope_raster.where(mask).median().item()

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
                "median_slope": median_slope,
                "area": area,
            }
        )

    # Create DataFrame
    df = pd.DataFrame(results)
    return df
