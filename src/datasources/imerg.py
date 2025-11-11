import ocha_stratus as stratus
import pandas as pd
import xarray as xr
from tqdm.auto import tqdm
import numpy as np
from typing import List

import pandas as pd

from src.utils.db_utils import get_engine


PROCESSED_RASTER_BLOB_NAME = (
    "imerg/daily/late/v7/processed/imerg-daily-late-{date}.tif"
)


def get_blob_name(date: pd.Timestamp):
    return PROCESSED_RASTER_BLOB_NAME.format(date=date.date())


def open_imerg_raster(date: pd.Timestamp):
    blob_name = get_blob_name(date)
    return stratus.open_blob_cog(
        blob_name, container_name="raster", stage="prod"
    )


def open_imerg_raster_dates(dates, disable_progress_bar: bool = True):
    das = []
    error_dates = []
    for date in tqdm(dates, disable=disable_progress_bar):
        try:
            da_in = open_imerg_raster(date)
        except Exception as e:
            print(date)
            print(e)
            error_dates.append(date)
            continue
        da_in.attrs["_FillValue"] = np.nan
        da_in = da_in.rio.write_crs(4326)
        da_in = da_in.where(da_in >= 0).squeeze(drop=True)
        da_in["date"] = date
        da_in = da_in.persist()
        das.append(da_in)
    da = xr.concat(das, dim="date")
    if len(error_dates) > 0:
        print(f"Error dates: {error_dates}")
    return da

def fetch_imerg_data(
    pcodes: List[str], start_date: pd.Timestamp, end_date: pd.Timestamp
) -> pd.DataFrame:
    """Fetch IMERG data for a list of PCODES and a date range

    Parameters
    ----------
    pcodes
        List of PCODES
    start_date
        Start date
    end_date
        End date

    Returns
    -------
    pd.DataFrame
        DataFrame with IMERG data
    """
    pcodes_query_str = ", ".join([f"'{p}'" for p in pcodes])
    query = f"""
    SELECT *
    FROM public.imerg
    WHERE
        valid_date BETWEEN '{start_date.date()}' AND '{end_date.date()}'
        AND pcode IN ({pcodes_query_str})
    """
    return pd.read_sql(query, get_engine(stage="prod"))