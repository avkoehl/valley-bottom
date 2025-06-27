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
        minsize * sample_distance is the minimum length of a reach to be segmented
    spatial_radius : int, default=30
        the radius of the gaussian filter to smooth the DEM for slope calculation
    sigma : int, default=5
        the standard deviation of the gaussian filter to smooth the DEM

    low_gradient_threshold : int, default=1
        the threshold for low gradient reaches
    medium_gradient_threshold : int, default=3
        the threshold for medium gradient reaches, anything above this is high gradient

    hand_lg_strahler_one : int, default=5
        the hand threshold for low gradient reaches with strahler 1
    hand_lg_strahler_two : int, default=10
        the hand threshold for low gradient reaches with strahler 2
    hand_lg_strahler_threeplus : int, default=20
        the hand threshold for low gradient reaches with strahler 3 or greater
    hand_mg : int, default=10
        the hand threshold for medium gradient reaches
    hand_hg : int, default=5
        the hand threshold for high gradient reaches

    floor_max_slope : int, default=12
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

    # smoothing parameters
    spatial_radius: int = 30
    sigma: int = 5

    # reach gradient thresholds
    low_gradient_threshold: int = 1
    medium_gradient_threshold: int = 3  # high gradient is anything above this

    # hand thresholds for gradient X strahler
    hand_lg_strahler_one: int = 5
    hand_lg_strahler_two: int = 10
    hand_lg_strahler_threeplus: int = 20
    hand_mg: int = 10
    hand_hg: int = 5

    # post process
    floor_max_slope: int = 12
    min_hole_to_keep_area: float = 100000
