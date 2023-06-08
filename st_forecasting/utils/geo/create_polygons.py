import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from pyproj import Proj, transform


def _adjust_max_sides(coord: float, size: int, side_size: int) -> int:
    return size * side_size + int(coord)


def create_polygons(latitude: float, longitude: float, size: int, side: int) -> gpd.GeoDataFrame:
    # Projection
    p1 = Proj(init='epsg:4326')  # WGS84
    p2 = Proj(init='epsg:3857')  # pseudo mercator projection, units meters

    # Transform points into 3857 projection
    llx, lly = transform(p1, p2, longitude, latitude)

    urx = _adjust_max_sides(llx, size, side)
    ury = _adjust_max_sides(lly, size, side)

    polygons = []

    # Create squares within the bounding box
    for x in range(int(llx), int(urx), size):
        for y in range(int(lly), int(ury), size):
            # Create a square (as a polygon)
            polygon = Polygon([
                (x, y),
                (x + size, y),
                (x + size, y + size),
                (x, y + size)
            ])

            polygons.append(polygon)

    df = pd.DataFrame({'square_id': np.arange(0, len(polygons)), 'geometry': polygons})
    geo_df = gpd.GeoDataFrame(data=df, geometry=df.geometry, crs='EPSG:3857')
    # Transform polygons back to WGS84
    geo_df['geometry'] = geo_df.geometry.to_crs('EPSG:4326')

    return geo_df
