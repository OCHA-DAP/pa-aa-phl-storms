import ocha_stratus as stratus
import requests

from src.constants import ISO3, PROJECT_PREFIX

FIELDMAPS_URL = (
    "https://data.fieldmaps.io/{data_type}/originals/{iso3}.shp.zip"
)
BLOB_NAME = "{project_prefix}/raw/codab/{iso3}.shp.zip"


def get_blob_name(iso3: str = ISO3):
    iso3 = iso3.lower()
    return BLOB_NAME.format(project_prefix=PROJECT_PREFIX, iso3=iso3)


def download_codab_to_blob(iso3: str = ISO3):
    iso3 = iso3.lower()
    url = FIELDMAPS_URL.format(iso3=iso3, data_type="cod")
    response = requests.get(url)
    response.raise_for_status()
    blob_name = get_blob_name(iso3)
    stratus.upload_blob_data(response.content, blob_name)


def load_codab_from_blob(
    admin_level: int = 0, aoi_only: bool = False, iso3: str = ISO3
):
    iso3 = iso3.lower()
    shapefile = f"{iso3}_adm{admin_level}.shp"
    gdf = stratus.load_shp_from_blob(
        blob_name=get_blob_name(iso3),
        shapefile=shapefile,
        stage="dev",
    )
    if aoi_only:
        raise ValueError("No AOI defined")
    return gdf
