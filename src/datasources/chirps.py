import datetime
from typing import Literal

import rioxarray as rxr
import xarray as xr

CHIRPS_COG_URL = (
    "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/cogs/"
    "p{pitch}/{valid_date:%Y}/chirps-v2.0.{valid_date:%Y.%m.%d}.cog"
)


def open_chirps_cog(
    d: datetime.datetime, pitch: Literal["05", "25"] = "05"
) -> xr.DataArray:
    """
    Open CHIRPS COG for a specific date.
    :param d: date
    :param pitch: pitch
    :return: xarray DataArray
    """
    url = CHIRPS_COG_URL.format(valid_date=d, pitch=pitch)
    return rxr.open_rasterio(url, chunks=True)

