import numpy as np
import pandas as pd

"""
analyze contours in the REM at a set interval 

Look for changes between contours in mean slope, and in the area of the contour

where the floor transitions to the walls, the contours should closer together and their areas smaller

"""


def _generate_intervals(raster, interval):
    # Find the appropriate starting point by rounding down the minimum value to the nearest step
    start = interval * np.floor(raster.min().item() / interval)
    end = interval * np.ceil(raster.max().item() / interval)
    intervals = np.arange(int(start), int(end) + interval, interval)
    return intervals


def identify_rem_threshold(rem_raster, slope_raster, interval, threshold):
    # find the contour that has a slope greater than the threshold
    # return the min_value of that contour
    intervals = _generate_intervals(rem_raster, interval)
    for i in range(len(intervals) - 1):
        min_val = intervals[i]
        max_val = intervals[i + 1]
        if max_val < 1:
            continue
        # Create mask for this interval
        mask = (rem_raster >= min_val) & (rem_raster < max_val)

        # Skip if no pixels in this interval
        if not mask.any():
            continue

        # Get mean slope for this interval
        median_slope = slope_raster.where(mask).median().item()

        if median_slope > threshold:
            if min_val < 1:
                min_val = max_val
            return min_val

    # if no contours exceed the threshold, return None
    return None


def analyze_rem_contours(rem_raster, slope_raster, interval):
    intervals = _generate_intervals(rem_raster, interval)
    pixel_area = np.abs(rem_raster.rio.resolution()[0] * rem_raster.rio.resolution()[1])

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
        area = num_pixels * pixel_area
        if i == 0:
            min_val = rem_raster.min().item()
        if i == len(intervals) - 2:
            max_val = rem_raster.max().item()
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
