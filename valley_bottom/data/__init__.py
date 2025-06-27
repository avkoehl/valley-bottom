from importlib import resources
import rioxarray as rxr
import geopandas as gpd


def load_sample_dem():
    dem_file = resources.files("valley_bottom.data") / "180701020604-dem.tif"
    return rxr.open_rasterio(dem_file, masked=True).squeeze()


def load_sample_flowlines():
    flowlines_file = (
        resources.files("valley_bottom.data") / "180701020604-flowlinesmr.gpkg"
    )
    return gpd.read_file(flowlines_file)
