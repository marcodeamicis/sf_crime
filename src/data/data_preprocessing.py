import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List


def address_split(word):
    """This function get the 'Address' attribute and return the main street.
    """
    if ' of ' in word:
        return word.lower().partition('block of ')[2].lower().strip()
    elif ' / ' in word:
        return word.partition(' / ')[0].lower().strip()
    else:
        return np.nan


class TransformCordinates(BaseEstimator, TransformerMixin):
    def __init__(self, columns: List = ['x', 'y'], groupby: str = 'pd_district') -> None:
        self.columns = columns
        self.groupby = groupby
        self._df = pd.DataFrame()

    def fit(self, X: pd.DataFrame):
        _x = self.columns[0]
        _y = self.columns[1]
        df_replaced = X.copy()
        df_replaced[_x] = np.where((df_replaced[_x] >= -120.5), np.nan, df_replaced[_x])
        df_replaced[_y] = np.where((df_replaced[_y] >= 90), np.nan, df_replaced[_y])

        # TODO: Use the mean of each categories 'dates_year', 'pd_district', 'resolution', 'category'
        # as the imputed number.
        self._df = df_replaced.groupby(by=self.groupby).agg({_x: 'mean', _y: 'mean'}).reset_index()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        _x = self.columns[0]
        _y = self.columns[1]
        df_replaced = X.copy()

        lst_district = self._df.loc[(self._df[_x].isna()) & (self._df[_y].isna()), self.groupby].unique().tolist()

        for district in lst_district:
            # Inputing mean values in 'x'
            df_replaced.loc[
                (df_replaced[self.groupby] == district) & (df_replaced[_x].isna()),
                _x] = self._df.loc[self._df[self.groupby] == district, _x].mean()

            df_replaced.loc[
                (df_replaced[self.groupby] == district) & (df_replaced[_y].isna()),
                _y] = self._df.loc[self._df[self.groupby] == district, _y].mean()

        return df_replaced
