# Requires overpy, the library that lets us access the Overpass API through python, DON'T DELETE
import overpy
from geopy.geocoders import Nominatim
import requests

MAPBOX_API_KEY = ("pk.eyJ1Ijoia29uc3RhbnRpbm9zLWRlbWlyaXMiLCJhIjoiY20zdmVuOGtiMDlzdjJsc2dsaHg3d2w3dSJ9.Kc4Z9eZ7_"
                  "nycHeacJueb0A")
# This is (my) kostas's personal API key, might have to replace with a business one... I don't want to get charged
crosswalk_images_folder = "crosswalk_images"


def get_cross_walk_location(location_geo_code, bounds=0.001):
    over_api = overpy.Overpass()
    geo_locator = Nominatim(user_agent="crosswalk_locator")
    location = geo_locator.geocode(location_geo_code)

    if not location:
        raise ValueError("This location could not be geocoded.")

    geo_bounding_box = (location.latitude - bounds, location.longitude - bounds,
                        location.latitude + bounds, location.longitude + bounds)  # South, West, North, East

    query = f"""
    [out:json][timeout:25];
    (
      node["highway"="crossing"]({geo_bounding_box[0]},{geo_bounding_box[1]},{geo_bounding_box[2]},{geo_bounding_box[3]});
    );
    out body;
    """  # Should return all crosswalks within {bounds} longitude/ latitude points,

    result = over_api.query(query)

    crosswalks = []
    for node in result.nodes:
        crosswalks.append((node.lat, node.lon))
    # Now we have a list of all the (latitude, longitude) pairs in that area,
    # and can use it access the images of those areas for annotation

    return crosswalks


def get_mapbox_aerial_image(location, output_file_name, zoom=18, size="600x600", style="satellite-v9",
                            api_key=MAPBOX_API_KEY):
    longitude, latitude = location  # We split here, so it's easier to map this function onto a set of (lon, lat) coords
    base_url = f"https://api.mapbox.com/styles/v1/mapbox/{style}/static"

    coords = f"{latitude},{longitude},{zoom}"
    url = f"{base_url}/{coords}/{size}?access_token={api_key}"

    response = requests.get(url)

    if response.status_code == 200:
        with open(output_file_name, "wb") as file:
            file.write(response.content)
        print(f"Retrieved aerial image and saved it as {output_file_name}")

    else:
        print("Error, Error, Error!", response.status_code, response.text)


def get_crosswalk_images(geo_code, folder=crosswalk_images_folder, file_extension="png"):
    crosswalk_image_files = []
    crosswalk_set = get_cross_walk_location(geo_code, bounds=0.1)
    for crosswalk in crosswalk_set:
        formatted_crosswalk = list(map(float, crosswalk))
        stored_file_name = str(formatted_crosswalk)
        try:
            get_mapbox_aerial_image(formatted_crosswalk,
                                    f"{folder}/{formatted_crosswalk}.{file_extension}")
            crosswalk_image_files.append(stored_file_name)
        except Exception as e:
            print(e)

    return crosswalk_image_files


print(get_crosswalk_images("Bloomsbury, London, UK"))