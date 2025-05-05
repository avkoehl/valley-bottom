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
    low_gradient_threshold : int, default=1
        reaches with mean slope less than this are considered low gradient reaches
    medium_gradient_threshold : int, default=3
        reaches with mean slope greater than this and less than low_gradient_threshold are considered medium gradient reaches, everything else is high gradient
    spatial_radius : int, default=30
        the radius of the gaussian filter to smooth the DEM
    sigma : int, default=5
        the standard deviation of the gaussian filter to smooth the DEM
    lg_max_extent : int, default=50
        the cost threshold for valley extent definition for low gradient reaches
    rem_sample_distance : int, default=30
        the distance between points on a flowline to sample for REM for low gradient reaches
    lg_interval : int, default=10
        the interval for contours to analayze for low gradient reaches
    lg_default_threshold : int, default=20
        the default threshold for low gradient reaches if no contours exceed the slope threshold
    lg_slope_threshold : int, default=8
        first contour that has median slope greater than this value is used to define the threshold
    mg_max_extent : int, default=50
        the cost threshold for valley extent definition for medium gradient reaches
    mg_interval : int, default=10
        the interval for contours to analayze for medium gradient reaches
    mg_default_threshold : int, default=10
        the default threshold for medium gradient reaches if no contours exceed the slope threshold
    mg_slope_threshold : int, default=10
        first contour that has median slope greater than this value is used to define the threshold
    hg_max_extent : int, default=20
        the cost threshold for valley extent definition for high gradient reaches
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

    # reach segmentation parameters
    sample_distance: int = 10
    pelt_penalty: int = 10
    minsize: int = 100
    min_reach_length: int = 150

    # reach gradient thresholds
    low_gradient_threshold: int = 1
    medium_gradient_threshold: int = 3  # high gradient is anything above this

    # smoothing parameters
    spatial_radius: int = 30
    sigma: int = 5

    # low gradient reach analysis parameters
    lg_max_extent: int = 50
    rem_sample_distance: int = 30
    lg_interval: int = 10
    lg_slope_threshold: int = 8
    lg_default_threshold: int = 20

    # medium gradient reach analysis parameters
    mg_max_extent: int = 50
    mg_interval: int = 10
    mg_slope_threshold: int = 10
    mg_default_threshold: int = 10

    # high gradient reach analysis parameters
    hg_max_extent: int = 20
    hg_interval: int = 5
    hg_slope_threshold: int = 12
    hg_default_threshold: int = 5

    # post process
    floor_max_slope: int = 15
    min_hole_to_keep_area: float = 100000
