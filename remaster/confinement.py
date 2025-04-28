# analyze a cross section to find confining slopes

"""
 input:
    - flowlines
    - dem

steps:
1. rem < 5 -> polygon
2. centerline of polygon
3. cross sections along centerline
4. intersect cross section with flowline
5. splitting cross section into two parts, one on each side of the flowline, where flowline is first coordinate
6. get data along the cross section
    - elevation, slope, profile curvature
7. find confining point:
    - foreach point with negative curvature
        - if max ascent path has sustained slope and that sustained slope leads to rem gain > X meteres
        - save that point
8. return all confining points (to get rem values of those points)

"""

fp_mask = reach_rem < 5
poly = binary_raster_to_polygon(fp_mask)
if isinstance(poly, list):
    polygons = gpd.GeoSeries(poly, crs=dem.rio.crs)
    # keep only ones that inersect with the flowline
    polygons = polygons[polygons.intersects(reach)]
    # keep only the largest one
    polygons = polygons[polygons.area == polygons.area.max()]
    poly = polygons.iloc[0]

closed_poly = remove_holes(poly, maxarea=None)
# get the centerline of the polygon

baseline = slope.where(reach_rem <= 5).mean()
# iterate through each 5m contour and get mean slope
for 

