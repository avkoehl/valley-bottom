from dataclasses import dataclass


@dataclass
class Config:
    """Complete Configuration for the Valley Bottom Detection Algorithm
    Parameters
    ----------
    sample_distance : int, default=10
        the distance between points on a flowline to sample for reach segmentation
    pelt_penalty : int, default=10
        the penalty for the PELT algorithm to segment the flowline slope series
    minsize : int, default=100
        minsize * sample_distance is the minimum length of a reach
    min_reach_length : int, default=150
        the minimum length of a reach in meters
    gradient_threshold : int, default=3
        reaches with mean slope less than this are considered low gradient reaches otherwise are high gradient
    spatial_radius : int, default=30
        the radius of the gaussian filter to smooth the DEM
    sigma : int, default=5
        the standard deviation of the gaussian filter to smooth the DEM
    cost_threshold : int, default=50
        the cost threshold for valley extent definition
    rem_sample_distance : int, default=30
        the distance between points on a flowline to sample for REM
    lg_interval : int, default=10
        the interval for contours to analayze for low gradient reaches
    lg_default_threshold : int, default=30
        the default threshold for low gradient reaches if no contours exceed the slope threshold
    lg_slope_threshold : int, default=8
        first contour that has median slope greater than this value is used to define the threshold
    hg_interval : int, default=5
        the interval for contours to analayze for high gradient reaches
    hg_default_threshold : int, default=5
        the default threshold for high gradient reaches if no contours exceed the slope threshold
    hg_slope_threshold : int, default=12
        first contour that has median slope greater than this value is used to define the threshold
    floor_max_slope : int, default=15
        slopes greater than this are removed from the floor mask
    min_hole_to_keep_area : float, default=100_000
        the minimum area of a hole to keep it in the floor mask

    Examples
    --------
    Create a configuration with default parameters:

    >>> config = Config()

    Create a configuration with custom parameters:

    >>> config = Config()
    >>> config.pelt_penalty = 20

    """

    sample_distance: int = 10
    pelt_penalty: int = 10
    minsize: int = 100
    min_reach_length: int = 150
    gradient_threshold: int = 3
    spatial_radius: int = 30
    sigma: int = 5

    cost_threshold: int = 50
    rem_sample_distance: int = 30
    lg_interval: int = 10
    lg_default_threshold: int = 20
    lg_slope_threshold: int = 8

    hg_interval: int = 5
    hg_default_threshold: int = 5
    hg_slope_threshold: int = 12

    floor_max_slope: int = 15
    min_hole_to_keep_area: float = 100000
