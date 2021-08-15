
import numpy as np
import pandas as pd
import seaborn as sns
import geopy
from geopy import distance
from geopy.extra.rate_limiter import RateLimiter


def main():
    properties_df = pd.read_csv(
        "properties-susquehannavalley2020MAR29.csv", index_col=0
    )

    properties_df = properties_df[properties_df["list-card-type"] != "Lot / Land for sale"]
    properties_df = properties_df[properties_df["list-card-type"] != "New construction"]
    properties_df = properties_df[
        properties_df["list-card-type"] != "Apartment for sale"
    ]

    properties_df["list-card-price"] = properties_df["list-card-price"].str.replace(
        r"\$|,|\+|-|Est| ", ""
    )

    properties_df["list-card-price"] = pd.to_numeric(properties_df["list-card-price"])

    properties_df["beds"] = properties_df["list-card-details"].str.extract("(.+) bds")
    properties_df["beds"] = properties_df["beds"].str.replace("--", "")
    properties_df["beds"] = pd.to_numeric(properties_df["beds"])

    properties_df["baths"] = properties_df["list-card-details"].str.extract(
        "bds(.+) ba"
    )
    properties_df["baths"] = properties_df["baths"].str.replace("--", "")
    properties_df["baths"] = pd.to_numeric(properties_df["baths"])

    properties_df["floorspace"] = properties_df["list-card-details"].str.extract(
        "ba(.+) sqft"
    )
    properties_df["floorspace"] = properties_df["floorspace"].str.replace(",", "")
    properties_df["floorspace"] = properties_df["floorspace"].str.replace("--", "")
    properties_df["floorspace"] = pd.to_numeric(properties_df["floorspace"])

    properties_df["street"] = properties_df["list-card-addr"].str.extract(
        "[0-9]+ (.+) "
    )

    street_df = properties_df["street"].str.split(",", n=2, expand=True)
    street_df.columns = pd.Index(["street_name", "city", "state"])
    properties_df = properties_df.join(street_df)

    locator_Nominatim = geopy.Nominatim(user_agent="myGeocoder")
    olin_science = locator_Nominatim.geocode(
        "Olin Science, Vaughan Lit Drive, College Park, Union County, PA, 17837"
    )

    buckenell_math_lat_long = tuple(olin_science.point)[:2]
    buckenell_math_lat_long

    locator_ArcGIS = geopy.ArcGIS(user_agent="myGeocoder")
    locator_ArcGIS.geocode("614 Maclay Ave, Lewisburg, PA 17837")

    geocode_Nominatim_rate_limited = RateLimiter(
        locator_Nominatim.geocode, min_delay_seconds=1
    )
    geocode_ArcGIS_rate_limited = RateLimiter(
        locator_ArcGIS.geocode, min_delay_seconds=1
    )

    properties_df["location_ArcGIS"] = properties_df["list-card-addr"].apply(
        geocode_ArcGIS_rate_limited
    )

    properties_df["location_ArcGIS"].apply(
        lambda loc: loc.latitude if "latitude" in dir(loc) else np.nan
    )

    properties_df["latitude"] = properties_df["location_ArcGIS"].apply(
        lambda loc: loc.latitude if "latitude" in dir(loc) else np.nan
    )
    properties_df["longitude"] = properties_df["location_ArcGIS"].apply(
        lambda loc: loc.longitude if "longitude" in dir(loc) else np.nan
    )
    properties_df["point"] = properties_df["location_ArcGIS"].apply(
        lambda loc: (loc.latitude, loc.longitude)
        if "longitude" in dir(loc)
        else (np.nan, np.nan)
    )

    properties_df.plot.scatter(x="longitude", y="latitude")

    def distance_to_math(location):
        try:
            return distance.distance(location, buckenell_math_lat_long).miles
        except:
            return np.nan

    properties_df["distance_to_math_dept"] = properties_df["point"].apply(
        distance_to_math
    )

    sns.scatterplot(
        data=properties_df, x="longitude", y="latitude", size="distance_to_math_dept"
    )

    properties_df.to_csv("properties-susquahannavalley-enriched2020MAR29.csv")


if __name__ == "__main__":
    main()
