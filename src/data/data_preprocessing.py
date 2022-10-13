# from datetime import datetime

import humps
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List


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


def snake_case_columns(df: pd.DataFrame) -> pd.DataFrame:
    df_prepared = df.copy()
    df_prepared.columns = [humps.decamelize(x.strip()).lower() for x in df_prepared.columns.tolist()]

    return df_prepared


def address_split(word: str) -> str:
    """This function get the 'Address' attribute and return the main street.
    """
    if ' of ' in word:
        return word.lower().partition('block of ')[2].lower().strip()
    elif ' /' in word:
        return word.partition(' /')[0].lower().strip()
    else:
        return np.nan


def create_simplified_address_column(df: pd.DataFrame, address_column: str = 'address') -> pd.DataFrame:
    """This function get the 'Address' attribute and return the main street.
    """
    df_transformed = df.copy()

    if 'address' in df_transformed.columns:
        df_transformed['simplified_address'] = df_transformed[address_column].apply(address_split)
        df_transformed.drop(columns=[address_column], inplace=True)

    return df_transformed


def create_date_based_columns(df: pd.DataFrame, date_column: str = 'dates') -> pd.DataFrame:
    df_evaluation = df.copy()

    if date_column in df_evaluation.columns:
        lst_original_columns = df.columns.tolist()

        df_evaluation[date_column + '_year'] = df_evaluation['dates'].dt.year
        df_evaluation[date_column + '_month'] = df_evaluation['dates'].dt.month
        df_evaluation[date_column + '_hour'] = df_evaluation['dates'].dt.hour
        # df_evaluation[date_column + '_day'] = df_evaluation['dates'].dt.date
        df_evaluation[date_column + '_day'] = df_evaluation['dates'].dt.day
        # TODO: Deactive the line above and active two lines above.

        # df_evaluation['is_daytime'] = np.where(
        #     (df_evaluation[date_column + '_hour'] > datetime.strptime('06:00:00', '%H:%M:%S').time()) &
        #     (df_evaluation[date_column + '_hour'] < datetime.strptime('18:00:00', '%H:%M:%S').time()),
        #     1, 0
        #     )

        df_evaluation['is_daytime'] = np.where(
            (df_evaluation[date_column + '_hour'] > 6) & (df_evaluation[date_column + '_hour'] < 18),
            1, 0
            )

        lst_original_columns.remove(date_column)
        lst_new_columns = [
            date_column + '_year', date_column + '_month', date_column + '_hour', date_column + '_day', 'is_daytime'
            ]
        lst_new_columns.extend(lst_original_columns)

        df_evaluation = df_evaluation[lst_new_columns]

    else:
        print(f"Column '{date_column}' was not detected.")

    return df_evaluation
