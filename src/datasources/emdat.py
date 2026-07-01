import duckdb
import ocha_stratus as stratus

from src.constants import ISO3


def load_emat():
    blob_name = "emdat/processed/emdat_all.parquet"
    url = (
        stratus.get_container_client(container_name="global")
        .get_blob_client(blob_name)
        .url
    )

    con = duckdb.connect()
    df_emdat = con.execute(
        f"""
        SELECT *
        FROM read_parquet('{url}')
        WHERE ISO = '{ISO3.upper()}' AND "Disaster Subtype" = 'Tropical cyclone' AND Historic = 'No'
    """
    ).df()
    return df_emdat
