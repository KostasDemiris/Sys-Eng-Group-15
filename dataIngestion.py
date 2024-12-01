import geopandas as gpd
import matplotlib.pyplot as plt
import fiona
import numpy as np
import rasterio
from rasterio import features
from rasterio.enums import MergeAlg
from rasterio.plot import show


geo_database = "TRAFFIC_PavementMarkings.gdb"


def ingest_ground_truth_data(save=False, save_as="ground_truth_raster.tif", show_data=False, resolution=1):
    # Resolution is in terms of meters...
    # Trying to visually inspect the raster file using this resolution will look like its all zeros - the data is very sparse
    gdf = gpd.read_file(geo_database, layer='TRAFFIC_PavementMarkings')

    unique_types = gdf["TYPE"].unique()
    type_mapping = {type_value: idx for idx, type_value in enumerate(unique_types)}
    # The particular type_mapping that we're looking  for is the PM-CROSSWALK Type - which is 3 in type_numeric
    gdf['TYPE_NUMERIC'] = gdf['TYPE'].map(type_mapping)

    geometry = gdf['geometry']

    minx, miny, maxx, maxy = gdf.total_bounds  # Use the bounds of the geometries
    width, height = int((maxx - minx)), int((maxy - miny))
    raster_dims = (width//resolution, height//resolution)

    transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, width, height)
    output_raster = features.rasterize(
        [(geo, data) for geo, data in zip(geometry, gdf['TYPE_NUMERIC'])],  # This is specific to the columns provided in the cambridge Dataset
        out_shape=raster_dims,
        transform=rasterio.transform.from_bounds(minx, miny, maxx, maxy, width, height),
        fill=0, dtype='int32', all_touched=True,  # If it begins to predict excessively large bounding boxes,
        # we might have to switch it over to all_touched false...
    )

    if save:
        # In case we want to save it as a .tif file as well, if it's more convenient.
        with rasterio.open(
                save_as, 'w',
                drive="GTiff", height=raster_dims[0], width=raster_dims[1], count=1, dtype=output_raster.dtype,
                crs=gdf.crs,  # For compatability we have this, but we might have to enforce BGS or (lat, lon)
                transform=transform
        ) as writer:
            writer.write(output_raster, 1)

    if show_data:
        gdf.plot()
        plt.show()

    return output_raster


raster_data = ingest_ground_truth_data()
