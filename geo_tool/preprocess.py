import shapely
import pandas as pd
import geopandas as gpd


class GeoAggregator:
    def __init__(
            self,
            squares: gpd.GeoDataFrame,
            geo_setup: dict,
            key_column: str = 'userid',
            target_column: str = 'intents'
    ):
        """
        GeoData preprocessing
        :param squares: GeoDataFrame with squares (zones)
        :param geo_setup: setup dict with coordinates
        """
        self.squares = squares
        self.geo_setup = geo_setup
        self.key = key_column
        self.target = target_column

    def _geo_preprocess(self, frame: pd.DataFrame) -> pd.DataFrame:
        geodf = frame.rename(columns={'call_point': 'geometry'}) \
            .sort_values('ts')

        geodf = gpd.GeoDataFrame(geodf, geometry=geodf['geometry'], crs='EPSG:4326')

        gdf = gpd.tools.sjoin_nearest(
            geodf,
            self.squares,
            max_distance=0.005,
            how='left'
        ).drop(['index_right', 'geometry'], axis=1) \
            .dropna(how='any')

        gdf = pd.DataFrame(data=gdf.values, columns=['ts', self.target, 'square_id'])
        gdf['square_id'] = gdf['square_id'].astype(int)

        return gdf

    def _geo_time_reduce(self, frame: pd.DataFrame, resampling_window: str) -> pd.DataFrame:
        snapshots = (
            frame.groupby('square_id')
            .resample(resampling_window, on='ts')
            .agg({self.target: 'sum'})
            .convert_dtypes()
            .rename(columns={'square_id': self.target})
            .reset_index(drop=False)
        )

        snapshots['ts'] = snapshots['ts'].astype(str)

        return snapshots

    def preprocess(
            self,
            frame: pd.DataFrame,
            resampling_window: str = '30T',
            filtering_window: str = None
    ) -> pd.DataFrame:
        """
        Data preprocessing
        :param frame: DataFrame with intents
        :param filtering_window: window, within which a new intents of one customer will not be considered
        :param resampling_window: window for resampling aggregation (mean)
        """
        sub = frame.copy()
        sub = sub.dropna(how='any')
        sub['ts'] = pd.to_datetime(sub['ts'])

        sub = (
            sub[
                (sub.latitude >= self.geo_setup['lat_min']) &
                (sub.latitude <= self.geo_setup['lon_min']) &
                (sub.longitude >= self.geo_setup['lat_max']) &
                (sub.longitude <= self.geo_setup['lon_max'])
            ]
        )

        if filtering_window is not None:
            mask = sub[[self.key, 'ts']].groupby(self.key)['ts'].diff().lt(filtering_window).sort_index()
            sub = sub[~mask]

        sub['call_point'] = sub.apply(lambda row: shapely.Point((row['longitude'], row['latitude'])), axis=1)
        sub = sub.drop(['latitude', 'longitude', self.key], axis=1)

        sub = self._geo_preprocess(sub)
        sub = self._geo_time_reduce(sub, resampling_window)

        return sub
