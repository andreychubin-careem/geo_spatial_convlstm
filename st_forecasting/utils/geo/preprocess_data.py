import shapely
import pandas as pd
import geopandas as gpd


def _geo_preprocess(frame: pd.DataFrame, squares: gpd.GeoDataFrame) -> pd.DataFrame:
    geodf = frame.drop(['latitude', 'longitude', 'userid'], axis=1) \
        .rename(columns={'call_point': 'geometry'}) \
        .sort_values('ts') \
        .assign(intents=1)

    geodf = gpd.GeoDataFrame(geodf, geometry=geodf['geometry'], crs='EPSG:4326')

    gdf = gpd.tools.sjoin_nearest(
        geodf,
        squares,
        max_distance=0.005,
        how='left'
    ).drop(['index_right', 'geometry'], axis=1) \
        .dropna(how='any')

    gdf = pd.DataFrame(data=gdf.values, columns=['ts', 'intents', 'square_id'])
    gdf['square_id'] = gdf['square_id'].astype(int)

    return gdf


def _geo_time_reduce(frame: pd.DataFrame, resampling_window: str) -> pd.DataFrame:
    snapshots = (
        frame.groupby('square_id')
        .resample(resampling_window, on='ts')
        .agg({'intents': 'sum'})
        .convert_dtypes()
        .rename(columns={'square_id': 'intents'})
        .reset_index(drop=False)
    )

    snapshots['ts'] = snapshots['ts'].astype(str)

    return snapshots


def preprocess(
        frame: pd.DataFrame,
        squares: gpd.GeoDataFrame,
        setup: dict,
        filtering_window: str = '10T',
        resampling_window: str = '30T'
) -> pd.DataFrame:
    """
    Data preprocessing
    :param frame: DataFrame with intents
    :param squares: GeoDataFrame with squares (zones)
    :param setup: setup dict with coordinates
    :param filtering_window: window, whithin which a new intents of one customer will not be considered
    :param resampling_window: window for resampling aggregation (mean)
    """
    sub = frame.copy()
    sub = sub.dropna(how='any')
    sub['ts'] = pd.to_datetime(sub['ts'])

    sub = (
        sub[
            (sub.latitude >= setup['lat_min']) &
            (sub.latitude <= setup['lon_min']) &
            (sub.longitude >= setup['lat_max']) &
            (sub.longitude <= setup['lon_max'])
            ]
    )

    mask = sub[['userid', 'ts']].groupby('userid')['ts'].diff().lt(filtering_window).sort_index()
    sub = sub[~mask]
    sub['call_point'] = sub.apply(lambda row: shapely.Point((row['longitude'], row['latitude'])), axis=1)

    sub = _geo_preprocess(sub, squares)
    sub = _geo_time_reduce(sub, resampling_window)

    return sub
