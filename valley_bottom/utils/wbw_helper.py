import tempfile
import shutil
import os

import rioxarray as rxr


def wbeRaster_to_rxr(wbe_raster, wbe):
    # Convert the WbEnvironment raster to an xarray DataArray
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
    try:
        wbe.write_raster(wbe_raster, temp_file.name)
        raster = rxr.open_rasterio(temp_file.name, masked=True).squeeze()
    finally:
        temp_file.close()
        os.remove(temp_file.name)
    return raster


def rxr_to_wbeRaster(raster, wbe):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
    try:
        raster.rio.to_raster(temp_file.name, driver="GTiff")
        wbe_raster = wbe.read_raster(temp_file.name)
    finally:
        temp_file.close()
        os.remove(temp_file.name)
    return wbe_raster


def gpd_to_wbeVector(gdf, wbe):
    temp_dir = tempfile.mkdtemp()
    try:
        temp_shp_file = os.path.join(temp_dir, "temp.shp")
        gdf.to_file(temp_shp_file)
        vector = wbe.read_vector(temp_shp_file)
    finally:
        shutil.rmtree(temp_dir)
    return vector
