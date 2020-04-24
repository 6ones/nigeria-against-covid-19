import pandas as pd
import numpy as np
from functools import reduce


def transform_df(df, columns_to_drop=["Lat", "Long", "Province/State"]):
    """Transform dataframe to required format for analysis
    Params:
        df: Pandas.DataFrame
            dataframe containing the data
        columns_to_drop: list,
            list of columns to remove from the dataframe
            [Lat, Long, Province/State]
    """
    df = df.drop(columns_to_drop, axis=1)
    df = df.groupby("Country/Region").sum().T
    df = df.rename_axis("date")
    df.index = pd.to_datetime(df.index)
    del df.columns.name
    return df


def get_merged_country_data(country, dfs):
    """Perform 3 way merge on confirmed, deaths and recovered dataframes
    on the {country} column
    Params:
        country: string, country to locate
        dfs: dict, containing dataframes of all the data.
    Return:
        df: Pandas.DataFrame, dataframe containing all data
    """

    options = ["confirmed", "deaths", "recovered"]
    for option in options:
        if option not in dfs.keys():
            raise KeyError(f"{option} does not exist in dataframe")

    confirmed = dfs["confirmed"]
    deaths = dfs["deaths"]
    recovered = dfs["recovered"]

    confirmed = confirmed.loc[:, country].to_frame()
    deaths = deaths.loc[:, country].to_frame()
    recovered = recovered.loc[:, country].to_frame()

    confirmed.columns = ["confirmed"]
    deaths.columns = ["deaths"]
    recovered.columns = ["recovered"]

    all_dfs = [confirmed, deaths, recovered]
    df = reduce(
        lambda left, right: pd.merge(
            left, right, right_index=True, left_index=True
        ),
        all_dfs,
    )
    df.index = pd.to_datetime(df.index)
    return df


def list_all_countries(df, country_column):
    """List all countries in dataframe
    Params:
        df: Pandas.DataFrame, dataframe in use
        country_column: string, name of the column containing countries in the dataframe"""
    if country_column in df:
        return list(df[country_column].unique())
    else:
        raise KeyError(f"{country_column} not in dataframe")


def mobility_per_country(country, gm_report_df):
    """Get country data in the mobility dataframe
    Params:
        country: str, name of the country in the df
        gm_report_df: Pandas.DataFrame, dataframe containing mobility report of countries"""

    if country not in list(gm_report_df.country_region.unique()):
        raise KeyError(f"{country} not in dataframe")

    df = gm_report_df[gm_report_df.country_region == country].reset_index(
        drop=True
    )
    df.sub_region_1 = df.sub_region_1.replace(np.nan, "all")
    df = df.set_index("date")
    df.index = pd.to_datetime(df.index)
    df = df.drop(
        ["country_region_code", "country_region", "sub_region_2"], axis=1
    )
    return df
