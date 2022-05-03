from .base import AbstractDataset

import pandas as pd


class SantanderDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'santander'

    @classmethod
    def is_zipfile(cls):
        return False

    @classmethod
    def url(cls):
        pass

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('ratings.csv')
        df = pd.read_csv(file_path, header=None)
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        print(df)
        return df
