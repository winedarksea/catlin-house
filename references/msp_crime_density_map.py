"""
Prompt for this:
We want to make a map like this of both Minneapolis and St Paul Crime, ideally pulled from the city APIs, data for at least 3 years, with two separate maps, one of violent crime and gun crime, and the other of house burglaries, with the goal of showing density maps of areas of crime.
We are trying to make it not be too sensational (as with 3 years of crimes, almost all areas will probably show some crime).

Comfort depends on crime type, disorder cues, lighting, vacancy, and familiarity. 
Shot spotter data or not. Different density areas.
Freaky how roads and other barriers really show up. Was something like Hadrian's wall just a barrier like this?
Relative density is a bit of a squishy metric, here from 99th percentile
There is possibly a cost to the city of making these API calls, so don't run it unless you need to.
Reporting bias is definite
https://communitycrimemap.com/ doesn't have all communities, for example I think it has Ramsey County police reports but not City of St Paul
Main surbuban crime area not shown is Brooklyn Park and surrounding commmunities there, and South St Paul and West St Paul
"""

from __future__ import annotations

import os
import re
import sys
import math
import json
import requests
import warnings
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from shapely.geometry import box, Point
try:
    from shapely import concave_hull as shapely_concave_hull
    HAS_CONCAVE_HULL = True
except ImportError:
    HAS_CONCAVE_HULL = False

from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter

# Optional basemap
try:
    import contextily as cx
    HAS_CONTEXTILY = True
except Exception:
    HAS_CONTEXTILY = False


# =========================
# CONFIG
# =========================

YEARS_BACK = 4
PULLED_UTC = datetime.now(timezone.utc)
SINCE_UTC = PULLED_UTC - timedelta(days=365 * YEARS_BACK)

# Set True (or pass --sample) to pull only SAMPLE_SIZE records per dataset for quick tests
SAMPLE_MODE: bool = "--sample" in sys.argv  # use when testing downloads, but full data is fine if cached already 
SAMPLE_SIZE: int = 2000

# Pass --download to force a fresh API pull even if cached parquet files exist.
# Without --download, main() loads cached annotated GDFs from CACHE_DIR.
FORCE_DOWNLOAD: bool = "--download" in sys.argv
CACHE_DIR: str = "_cache"

# Colormap positions are % of peak density (99th-pct vmax), so the scale is
# directly human-readable: 0.02 = 2% of peak, 0.40 = 40% of peak, etc.
# Used with a plain linear Normalize — no power warping needed.
CRIME_CMAP = LinearSegmentedColormap.from_list(
    "crime_density",
    [
        (0.000, (1.00, 0.92, 0.80, 0.00)),  # 0%   — transparent
        (0.020, (1.00, 0.92, 0.80, 0.00)),  # 2%   — transparent (no fill below 2%)
        (0.035, (1.00, 0.90, 0.76, 0.14)),  # 3.5% — very faint warm peach
        (0.050, (1.00, 0.90, 0.74, 0.17)),  # 5%   — faint warm peach
        (0.075, (1.00, 1.00, 0.72, 0.26)),  # 7.5% — soft pale yellow
        (0.100, (1.00, 0.97, 0.40, 0.40)),  # 10%  — pale yellow
        (0.200, (1.00, 0.88, 0.00, 0.56)),  # 20%  — yellow
        (0.400, (1.00, 0.50, 0.00, 0.72)),  # 40%  — orange
        (0.700, (0.88, 0.08, 0.00, 0.85)),  # 70%  — deep orange-red
        (1.000, (0.35, 0.00, 0.00, 0.96)),  # 100% — dark red
    ],
)

# Official portal pages
DATASETS = {
    "minneapolis_crime": {
        "city": "Minneapolis",
        "theme": "crime",
        "about_url": "https://opendata.minneapolismn.gov/datasets/cityoflakes::crime-data/about",
        "layer_url": "https://services.arcgis.com/afSMGVsC7QlRK1kZ/arcgis/rest/services/Crime_Data/FeatureServer/0",
        # (min_lon, min_lat, max_lon, max_lat) — clips outliers and 0,0 points
        "city_bounds": (-93.70, 44.87, -93.19, 45.07),
        # Neighborhood polygons dissolved to produce the city boundary outline
        "boundary_layer_url": "https://services.arcgis.com/afSMGVsC7QlRK1kZ/arcgis/rest/services/mpls_neighborhoods/FeatureServer/0",
    },
    "minneapolis_shots": {
        "city": "Minneapolis",
        "theme": "shots",
        "about_url": "https://opendata.minneapolismn.gov/datasets/shots-fired/about",
        "layer_url": "https://services.arcgis.com/afSMGVsC7QlRK1kZ/arcgis/rest/services/Shots_Fired/FeatureServer/0",
        "city_bounds": (-93.70, 44.87, -93.19, 45.07),
    },
    "stpaul_crime": {
        "city": "Saint Paul",
        "theme": "crime",
        "about_url": "https://information.stpaul.gov/datasets/stpaul::crime-incident-report/about",
        "layer_url": "https://services1.arcgis.com/9meaaHE3uiba0zr8/arcgis/rest/services/Crime_Incident_Report_-_Dataset/FeatureServer/0",
        "city_bounds": (-93.25, 44.88, -92.97, 45.02),
        # No geometry in this service — use district boundaries + random jitter
        "neighborhood_layer_url": "https://services1.arcgis.com/9meaaHE3uiba0zr8/arcgis/rest/services/District_Councils/FeatureServer/0",
        "neighborhood_id_field": "districtnumber",
        "record_neighborhood_field": "NEIGHBORHOOD_NUMBER",
        # District_Councils dissolved also serves as the city boundary
        "boundary_layer_url": "https://services1.arcgis.com/9meaaHE3uiba0zr8/arcgis/rest/services/District_Councils/FeatureServer/0",
    },
    "roseville_crime": {
        "city": "Roseville",
        "theme": "crime",
        "about_url": "https://data-roseville.opendata.arcgis.com/",
        "layer_url": "https://services3.arcgis.com/dty2kHktVXHrqO8i/arcgis/rest/services/Crime_Incidents/FeatureServer/0",
        # Roseville sits just north of Saint Paul; clips stray 0,0 sentinel points
        "city_bounds": (-93.24, 44.97, -93.09, 45.08),
        # City_Boundary_Line is a polyline layer — fetch_city_boundary will warn and
        # return None, triggering a concave-hull fallback from the crime point cloud.
        # If a polygon boundary layer becomes available, substitute it here.
        "boundary_layer_url": "https://services3.arcgis.com/dty2kHktVXHrqO8i/arcgis/rest/services/City_Boundary_Line/FeatureServer/0",
    },
}

# Category matching
VIOLENT_TERMS = [
    "HOMICIDE",
    "MURDER",
    "ROBBERY",
    "AGGRAVATED ASSAULT",
    "ASSAULT",
    "RAPE",
    "CRIMINAL SEXUAL CONDUCT",
    "CARJACKING",
    "CAR JACKING",
]

GUN_TERMS = [
    "FIREARM DISCHARGE",
    "FIREARM DISCHARGES",
    "DISCHARGING A FIREARM",
    "DISCHARGE",
    "SHOOTING",
    "SHOTS HEARD",
    "SHOTS FIRED",
    "SHOTSPOTTER",
    "SOUND OF SHOTS FIRED",
    "GUNFIRE",
    # "GUN",
    "WEAPON",
    "PERSON WITH A GUN",
    "PERSON WITH GUN",
    "WEAPON LAW VIOLATION",
    "WEAPON LAW VIOLATIONS",
    "WEAPONS VIOLATION",
    "FELON IN POSSESSION",
]

# For house/residential burglary, try strict residential terms first.
RESIDENTIAL_BURGLARY_TERMS = [
    "BURGLARY OF DWELLING",
    "RESIDENTIAL BURGLARY",
    "BURGLARY - RESIDENCE",
    "BURGLARY-RESIDENCE",
    "BURGLARY/HOUSE",
    "BURGLARY HOME",
    "BURGLARY DWELLING",
    "BURGLARY DWLNG",
    "BURGLARY DWL",
]

GENERIC_BURGLARY_TERMS = [
    "BURGLARY",
]

# Canonical categories used across both Minneapolis and Saint Paul datasets.
# These definitions are intentionally broad enough to align common labeling
# differences between the two ArcGIS feeds.
CANONICAL_CATEGORY_DEFS = {
    "violent": {
        "description": "Homicide, robbery, aggravated assault, rape/CSC, and carjacking-like violent offenses",
        "terms": VIOLENT_TERMS,
    },
    "gun": {
        "description": "Shots fired, firearm discharge, gunfire, weapons violations, and felon-in-possession style labels",
        "terms": GUN_TERMS,
    },
    "residential_burglary": {
        "description": "Residential/home/dwelling burglary labels",
        "terms": RESIDENTIAL_BURGLARY_TERMS,
    },
    "generic_burglary": {
        "description": "Fallback burglary category when residential subtype is unavailable",
        "terms": GENERIC_BURGLARY_TERMS,
    },
}

# Possible field names seen in ArcGIS crime layers
DATE_FIELD_CANDIDATES = [
    "reportedDateTime", "ReportedDate", "Reported_Date", "reported_date", "responseDate",
    "Response_Date", "report_date", "occurred_date", "occurred_datetime",
    "date", "Date", "incident_date", "INCIDENT_DATE", "event_datetime", "TimeDate"
]

OFFENSE_FIELD_CANDIDATES = [
    "Offense", "offense", "OFFENSE", "OFFENSE_NAME", "offense_name",
    "Offense_Category", "NIBRS_OFFENSE", "UCR_Crime",
    # Roseville uses UCRdesc (UCR description) and StatDesc (statute description)
    "UCRdesc", "ucrdesc", "StatDesc", "statdesc",
    # Saint Paul uses TYPETEXT for the human-readable crime label; INCIDENT is often a numeric code
    "TYPETEXT", "typetext", "TypeText", "CRIME_TYPE", "crime_type", "CrimeType",
    "OFFENSE_TYPE", "offense_type", "OffenseType",
    "INCIDENT", "incident", "INCIDENT_TYPE", "incident_type",
    "Category", "category", "CRIME", "crime", "description", "DESCRIPTION",
    "Detail", "detail", "Type", "type"
]

DETAIL_FIELD_CANDIDATES = [
    "detail", "Detail", "details", "Details", "DESCRIPTION", "description",
    "Offense", "offense_detail", "subtype", "Subtype",
    "TYPETEXT", "typetext", "TypeText",
    "StatDesc", "statdesc",
]

# Additional text fields used for robust cross-source categorization.
# All matching fields are concatenated into _all_text before category matching.
TEXT_FIELD_CANDIDATES = [
    "Offense", "offense", "OFFENSE", "OFFENSE_NAME", "offense_name",
    "Offense_Category", "NIBRS_OFFENSE", "UCR_Crime",
    # Roseville-specific
    "UCRdesc", "ucrdesc", "StatDesc", "statdesc",
    # Saint Paul-specific
    "TYPETEXT", "typetext", "TypeText", "CRIME_TYPE", "crime_type", "CrimeType",
    "OFFENSE_TYPE", "offense_type", "OffenseType",
    "INCIDENT", "incident", "INCIDENT_TYPE", "incident_type",
    "Problem_Initial", "Problem_Final",
    "Category", "category", "Type", "type", "DESCRIPTION", "description",
]

# Reasonable projection for local maps
PROJECT_CRS = "EPSG:3857"


# =========================
# HELPERS
# =========================

def discover_arcgis_layer_url(about_url: str) -> str | None:
    """
    Tries to discover the ArcGIS FeatureServer layer URL from an ArcGIS Hub / Open Data page.
    """
    patterns = [
        r"https://services\d*\.arcgis\.com/[^\"' <]+/arcgis/rest/services/[^\"' <]+/FeatureServer/\d+",
        r"https://services\d*\.arcgis\.com/[^\"' <]+/arcgis/rest/services/[^\"' <]+/MapServer/\d+",
    ]

    candidate_urls = [about_url]

    # Also try likely API pages
    if about_url.endswith("/about"):
        candidate_urls.append(about_url[:-6] + "/api?layer=0")
        candidate_urls.append(about_url[:-6] + "/explore")
    elif "/about?" in about_url:
        base = about_url.split("/about?")[0]
        candidate_urls.append(base + "/api?layer=0")
        candidate_urls.append(base + "/explore")

    headers = {"User-Agent": "Mozilla/5.0"}

    for url in candidate_urls:
        try:
            r = requests.get(url, headers=headers, timeout=30)
            r.raise_for_status()
            text = r.text
            for pat in patterns:
                m = re.search(pat, text)
                if m:
                    return m.group(0)
        except Exception:
            pass

    return None


def get_layer_url(dataset_cfg: dict) -> str:
    if dataset_cfg.get("layer_url"):
        return dataset_cfg["layer_url"]

    discovered = discover_arcgis_layer_url(dataset_cfg["about_url"])
    if not discovered:
        raise RuntimeError(
            f"Could not discover layer URL for {dataset_cfg['about_url']}. "
            "Open the dataset page, click 'API Explorer' or 'View Data Source', "
            "and paste the FeatureServer layer URL into the config."
        )
    return discovered


def arcgis_json(url: str, params: dict) -> dict:
    params = dict(params)
    params["f"] = "json"
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    if "error" in data:
        raise RuntimeError(f"ArcGIS error from {url}: {data['error']}")
    return data


def get_layer_metadata(layer_url: str) -> dict:
    return arcgis_json(layer_url, {})


def pick_field(existing_fields, candidates):
    lower_map = {f.lower(): f for f in existing_fields}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def fetch_all_features(layer_url: str, where="1=1", out_fields="*", page_size=None) -> pd.DataFrame:
    """
    Page through an ArcGIS FeatureServer layer and return a flat DataFrame with lon/lat.
    """
    meta = get_layer_metadata(layer_url)
    max_record_count = meta.get("maxRecordCount", 1000)
    page_size = page_size or min(max_record_count, 2000)

    count_resp = arcgis_json(
        f"{layer_url}/query",
        {
            "where": where,
            "returnCountOnly": "true",
        },
    )
    actual_total = count_resp["count"]

    if SAMPLE_MODE:
        total = min(actual_total, SAMPLE_SIZE)
        # Start near the end of the table so we get the most recent records
        offset = max(0, actual_total - SAMPLE_SIZE)
        print(f"[SAMPLE] Pulling {total:,} records (offset {offset:,}) from {layer_url}")
    else:
        total = actual_total
        offset = 0
        print(f"Pulling {total:,} records from {layer_url}")

    rows = []
    fetched = 0

    while fetched < total:
        params = {
            "where": where,
            "outFields": out_fields,
            "returnGeometry": "true",
            "outSR": 4326,
            "resultOffset": offset,
            "resultRecordCount": page_size,
        }
        data = arcgis_json(f"{layer_url}/query", params)

        feats = data.get("features", [])
        if not feats:
            if fetched < total:
                warnings.warn(
                    f"fetch_all_features: server returned empty page at offset {offset} "
                    f"after {fetched:,}/{total:,} records — stopping early."
                )
            break

        for feat in feats:
            attrs = feat.get("attributes", {}).copy()
            geom = feat.get("geometry", {}) or {}
            x = geom.get("x")
            y = geom.get("y")

            # Some layers may return points differently
            if x is None or y is None:
                if "longitude" in attrs and "latitude" in attrs:
                    x = attrs["longitude"]
                    y = attrs["latitude"]
                elif "Longitude" in attrs and "Latitude" in attrs:
                    x = attrs["Longitude"]
                    y = attrs["Latitude"]

            attrs["_lon"] = x
            attrs["_lat"] = y
            rows.append(attrs)

        fetched += len(feats)
        offset += len(feats)
        print(f"  fetched {min(fetched, total):,}/{total:,}")

    df = pd.DataFrame(rows)
    return df


def try_parse_arcgis_datetime(series: pd.Series) -> pd.Series:
    """
    ArcGIS dates are often either epoch milliseconds or strings.
    """
    if pd.api.types.is_numeric_dtype(series):
        # ArcGIS commonly uses ms since epoch
        return pd.to_datetime(series, unit="ms", utc=True, errors="coerce")

    return pd.to_datetime(series, utc=True, errors="coerce")


def prepare_gdf(df: pd.DataFrame, require_geometry: bool = True, city_bounds: tuple | None = None) -> tuple[gpd.GeoDataFrame, dict]:
    if df.empty:
        return gpd.GeoDataFrame(df, geometry=[], crs="EPSG:4326"), {}

    fields = list(df.columns)

    date_field = pick_field(fields, DATE_FIELD_CANDIDATES)
    offense_field = pick_field(fields, OFFENSE_FIELD_CANDIDATES)
    detail_field = pick_field(fields, DETAIL_FIELD_CANDIDATES)

    if date_field is None:
        raise RuntimeError(f"Could not identify a date field. Available fields: {fields}")

    df = df.copy()
    df["_date"] = try_parse_arcgis_datetime(df[date_field])
    df = df[df["_date"].notna()].copy()

    # Last 3 years only
    df = df[df["_date"] >= SINCE_UTC].copy()

    if offense_field:
        df["_offense_text"] = df[offense_field].astype(str).str.upper().str.strip()
    else:
        df["_offense_text"] = ""

    if detail_field:
        df["_detail_text"] = df[detail_field].astype(str).str.upper().str.strip()
    else:
        df["_detail_text"] = ""

    # Build a combined normalized text field from multiple columns so category
    # matching stays robust when source schemas vary across cities/datasets.
    text_fields = []
    for c in TEXT_FIELD_CANDIDATES:
        if c in fields and c not in text_fields:
            text_fields.append(c)

    if text_fields:
        joined = df[text_fields].fillna("").astype(str).agg(" | ".join, axis=1)
        df["_all_text"] = joined.str.upper().str.strip()
    else:
        df["_all_text"] = (df["_offense_text"].fillna("") + " | " + df["_detail_text"].fillna("")).str.upper().str.strip()

    info = {
        "date_field": date_field,
        "offense_field": offense_field,
        "detail_field": detail_field,
        "text_fields": text_fields,
        "fields": fields,
    }

    if not require_geometry:
        # Return without geometry — assign_coords_from_neighborhood will add it
        gdf = gpd.GeoDataFrame(df, geometry=gpd.GeoSeries([None] * len(df), dtype="geometry"), crs="EPSG:4326")
        return gdf, info

    # Geometry cleanup — drop nulls, global range violations, and known bad (0, 0) sentinel
    df = df[df["_lon"].notna() & df["_lat"].notna()].copy()
    df = df[(df["_lon"].between(-180, 180)) & (df["_lat"].between(-90, 90))].copy()
    df = df[(df["_lon"] != 0) | (df["_lat"] != 0)].copy()

    if city_bounds is not None:
        min_lon, min_lat, max_lon, max_lat = city_bounds
        df = df[df["_lon"].between(min_lon, max_lon) & df["_lat"].between(min_lat, max_lat)].copy()

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["_lon"], df["_lat"]),
        crs="EPSG:4326"
    )
    return gdf, info


def assign_coords_from_neighborhood(
    gdf: gpd.GeoDataFrame,
    neighborhood_layer_url: str,
    neighborhood_id_field: str,
    record_neighborhood_field: str,
    rng_seed: int = 42,
) -> gpd.GeoDataFrame:
    """
    For datasets with no point geometry, fetch district/neighborhood polygons,
    then place each crime at a random point within its district polygon.
    This gives a reasonable approximation for KDE density maps.
    """
    print(f"  No point geometry found — fetching neighborhood boundaries for coordinate fallback…")
    # Fetch polygon geometry directly as GeoJSON in a single call.
    poly_map: dict[int, object] = {}
    meta = get_layer_metadata(neighborhood_layer_url)
    max_rc = min(meta.get("maxRecordCount", 1000), 2000)
    params = {
        "where": "1=1",
        "outFields": "*",
        "returnGeometry": "true",
        "outSR": 4326,
        "resultOffset": 0,
        "resultRecordCount": max_rc,
        "f": "geojson",
    }
    r = requests.get(f"{neighborhood_layer_url}/query", params=params, timeout=60)
    r.raise_for_status()
    nb_poly_gdf = gpd.GeoDataFrame.from_features(r.json()["features"], crs="EPSG:4326")
    nb_poly_gdf[neighborhood_id_field] = pd.to_numeric(nb_poly_gdf[neighborhood_id_field], errors="coerce").astype("Int64")

    for _, row in nb_poly_gdf.iterrows():
        nb_id_val = row[neighborhood_id_field]
        if pd.isna(nb_id_val):
            continue
        poly_map[int(nb_id_val)] = row.geometry

    # Assign random points within each district polygon
    rng = np.random.default_rng(rng_seed)
    lons = np.full(len(gdf), np.nan)
    lats = np.full(len(gdf), np.nan)

    gdf = gdf.copy()
    gdf[record_neighborhood_field] = pd.to_numeric(
        gdf[record_neighborhood_field], errors="coerce"
    )

    for nb_id, poly in poly_map.items():
        mask = gdf[record_neighborhood_field] == nb_id
        n = mask.sum()
        if n == 0 or poly is None:
            continue
        minx, miny, maxx, maxy = poly.bounds
        pts_lon = np.empty(n)
        pts_lat = np.empty(n)
        filled = 0
        attempts = 0
        while filled < n and attempts < n * 50:
            batch = min(n - filled + 50, 500)
            rx = rng.uniform(minx, maxx, batch)
            ry = rng.uniform(miny, maxy, batch)
            inside = [poly.contains(Point(rx[i], ry[i])) for i in range(batch)]
            good = np.where(inside)[0]
            take = min(len(good), n - filled)
            pts_lon[filled:filled + take] = rx[good[:take]]
            pts_lat[filled:filled + take] = ry[good[:take]]
            filled += take
            attempts += batch
        if filled < n:
            # Fallback: jitter around centroid rather than stacking identical
            # coordinates, which would create artificial KDE spikes.
            remaining = n - filled
            cent_x, cent_y = poly.centroid.x, poly.centroid.y
            jitter_scale = max(maxx - minx, maxy - miny) * 0.05
            pts_lon[filled:] = cent_x + rng.uniform(-jitter_scale, jitter_scale, remaining)
            pts_lat[filled:] = cent_y + rng.uniform(-jitter_scale, jitter_scale, remaining)
        idx = gdf.index[mask]
        lons[gdf.index.get_indexer(idx)] = pts_lon
        lats[gdf.index.get_indexer(idx)] = pts_lat

    gdf["_lon"] = lons
    gdf["_lat"] = lats

    # Diagnostic: surface how many records are being dropped for having no matching
    # district polygon, and what their top offense labels are — so category-specific
    # undercounting (e.g. violent crimes with null districts) is visible rather than silent.
    unmatched_mask = gdf["_lon"].isna() | gdf["_lat"].isna()
    n_unmatched = int(unmatched_mask.sum())
    if n_unmatched > 0 and "_offense_text" in gdf.columns:
        top_unmatched = gdf.loc[unmatched_mask, "_offense_text"].value_counts().head(8)
        print(f"  Dropping {n_unmatched:,} records with no matching district (top labels: {dict(top_unmatched)})")
    elif n_unmatched > 0:
        print(f"  Dropping {n_unmatched:,} records with no matching district")

    gdf = gdf[gdf["_lon"].notna() & gdf["_lat"].notna()].copy()
    gdf = gpd.GeoDataFrame(
        gdf,
        geometry=gpd.points_from_xy(gdf["_lon"], gdf["_lat"]),
        crs="EPSG:4326",
    )
    print(f"  Assigned approximate coordinates to {len(gdf):,} records via district polygons.")
    return gdf


def contains_any(series: pd.Series, terms: list[str]) -> pd.Series:
    pattern = "|".join(re.escape(t) for t in terms)
    return series.str.contains(pattern, case=False, na=False, regex=True)


def annotate_canonical_categories(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Add canonical category booleans so Minneapolis and Saint Paul records
    are classified with a shared definition before filtering.
    """
    if gdf.empty:
        gdf = gdf.copy()
        for col in ["_cat_violent", "_cat_gun", "_cat_residential_burglary", "_cat_generic_burglary"]:
            gdf[col] = False
        return gdf

    gdf = gdf.copy()
    offense = gdf.get("_offense_text", pd.Series("", index=gdf.index)).astype(str)
    detail = gdf.get("_detail_text", pd.Series("", index=gdf.index)).astype(str)
    all_text = gdf.get("_all_text", offense + " | " + detail).astype(str)

    gdf["_cat_violent"] = contains_any(all_text, CANONICAL_CATEGORY_DEFS["violent"]["terms"])
    gdf["_cat_gun"] = (
        contains_any(all_text, CANONICAL_CATEGORY_DEFS["gun"]["terms"])
    )

    gdf["_cat_residential_burglary"] = (
        contains_any(all_text, CANONICAL_CATEGORY_DEFS["residential_burglary"]["terms"])
    )
    gdf["_cat_generic_burglary"] = (
        contains_any(all_text, CANONICAL_CATEGORY_DEFS["generic_burglary"]["terms"])
    )
    return gdf


def print_alignment_audit(city_name: str, gdf: gpd.GeoDataFrame, top_n_unmapped: int = 12) -> None:
    """
    Print a compact audit of how many records in each city map into the
    shared canonical categories and which top labels remain unmapped.
    """
    total = len(gdf)
    if total == 0:
        print(f"[{city_name}] alignment audit: no records")
        return

    violent = int(gdf["_cat_violent"].sum()) if "_cat_violent" in gdf.columns else 0
    gun = int(gdf["_cat_gun"].sum()) if "_cat_gun" in gdf.columns else 0
    res_burg = int(gdf["_cat_residential_burglary"].sum()) if "_cat_residential_burglary" in gdf.columns else 0
    any_burg = int(gdf["_cat_generic_burglary"].sum()) if "_cat_generic_burglary" in gdf.columns else 0

    mapped_any = (
        gdf.get("_cat_violent", False) |
        gdf.get("_cat_gun", False) |
        gdf.get("_cat_residential_burglary", False) |
        gdf.get("_cat_generic_burglary", False)
    )
    mapped_count = int(mapped_any.sum())
    mapped_pct = (100.0 * mapped_count / total) if total else 0.0

    print(f"\n[{city_name}] canonical alignment audit")
    print(f"  total incidents (3y): {total:,}")
    print(f"  violent matched: {violent:,}")
    print(f"  gun matched: {gun:,}")
    print(f"  residential burglary matched: {res_burg:,}")
    print(f"  any burglary matched: {any_burg:,}")
    print(f"  mapped to at least one target category: {mapped_count:,} ({mapped_pct:.1f}%)")

    # Sample _all_text so we can verify which source fields are actually being concatenated
    if "_all_text" in gdf.columns:
        sample_vals = gdf["_all_text"].dropna().head(3).tolist()
        print(f"  _all_text sample (3 rows):")
        for v in sample_vals:
            print(f"    {str(v)[:120]}")

    # Show top matched violent labels — if empty, the terms are missing from _all_text
    if "_cat_violent" in gdf.columns and violent > 0 and "_offense_text" in gdf.columns:
        matched_violent_labels = gdf.loc[gdf["_cat_violent"], "_offense_text"].value_counts().head(6)
        print(f"  top matched violent offense labels:")
        for label, count in matched_violent_labels.items():
            print(f"    + {label}: {count:,}")

    if "_offense_text" in gdf.columns:
        unmapped = gdf.loc[~mapped_any, "_offense_text"].fillna("(missing)")
        top_unmapped = unmapped.value_counts().head(top_n_unmapped)
        if len(top_unmapped):
            print("  top unmapped offense labels (for term tuning):")
            for label, count in top_unmapped.items():
                print(f"    - {label}: {count:,}")


def print_cross_city_alignment_summary(mpls_gdf: gpd.GeoDataFrame, stp_gdf: gpd.GeoDataFrame, top_n: int = 8) -> None:
    """
    Show side-by-side category totals and common matched offense labels by city
    so mapping equivalence can be reviewed quickly.
    """
    categories = [
        ("_cat_violent", "Violent"),
        ("_cat_gun", "Gun"),
        ("_cat_residential_burglary", "Residential burglary"),
        ("_cat_generic_burglary", "Any burglary"),
    ]

    print("\n[Cross-city category alignment]")
    for col, label in categories:
        m_count = int(mpls_gdf[col].sum()) if col in mpls_gdf.columns else 0
        s_count = int(stp_gdf[col].sum()) if col in stp_gdf.columns else 0
        print(f"  {label}: Minneapolis={m_count:,} | Saint Paul={s_count:,}")

        m_labels = mpls_gdf.loc[mpls_gdf[col], "_offense_text"].value_counts().head(top_n)
        s_labels = stp_gdf.loc[stp_gdf[col], "_offense_text"].value_counts().head(top_n)

        if len(m_labels) or len(s_labels):
            print("    Minneapolis top labels:")
            for k, v in m_labels.items():
                print(f"      - {k}: {v:,}")
            print("    Saint Paul top labels:")
            for k, v in s_labels.items():
                print(f"      - {k}: {v:,}")


def filter_violent(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf.copy()
    if "_cat_violent" in gdf.columns:
        return gdf[gdf["_cat_violent"]].copy()
    return gdf[contains_any(gdf["_offense_text"], VIOLENT_TERMS)].copy()


def filter_gun(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf.copy()
    if "_cat_gun" in gdf.columns:
        return gdf[gdf["_cat_gun"]].copy()
    mask = contains_any(gdf["_offense_text"], GUN_TERMS) | contains_any(gdf["_detail_text"], GUN_TERMS)
    return gdf[mask].copy()


def filter_residential_burglary(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf.copy()
    if "_cat_residential_burglary" in gdf.columns and "_cat_generic_burglary" in gdf.columns:
        strict = gdf[gdf["_cat_residential_burglary"]].copy()
        if len(strict) == 0:
            strict = gdf[gdf["_cat_generic_burglary"]].copy()
        return strict

    strict_mask = (
        contains_any(gdf["_offense_text"], RESIDENTIAL_BURGLARY_TERMS) |
        contains_any(gdf["_detail_text"], RESIDENTIAL_BURGLARY_TERMS)
    )
    strict = gdf[strict_mask].copy()

    # Fallback if the dataset does not distinguish residential burglary clearly
    if len(strict) == 0:
        generic_mask = (
            contains_any(gdf["_offense_text"], GENERIC_BURGLARY_TERMS) |
            contains_any(gdf["_detail_text"], GENERIC_BURGLARY_TERMS)
        )
        strict = gdf[generic_mask].copy()

    return strict


def kde_surface(
    gdf_proj: gpd.GeoDataFrame,
    gridsize=700,
    bandwidth_adjust=1.0,
    bandwidth_meters: float | None = None,
):
    """
    Returns meshgrid X,Y and KDE values Z in projected coordinates.

    bandwidth_meters: if set, fixes the kernel radius in map units (meters for
    EPSG:3857).  Research places tangible crime influence at ~100-200 m (1-2
    blocks) and perceived fear at ~400-800 m (0.25-0.5 mi).  A value of ~300-400 m
    prevents smearing while still merging nearby incidents into readable hotspots.
    bandwidth_adjust is applied on top as a fine-tuning multiplier.
    When bandwidth_meters is None the old Scott's-rule auto-bandwidth is used.
    """
    if len(gdf_proj) < 5:
        raise ValueError("Not enough points for KDE.")

    x = gdf_proj.geometry.x.values
    y = gdf_proj.geometry.y.values

    xmin, ymin, xmax, ymax = gdf_proj.total_bounds
    pad_x = (xmax - xmin) * 0.08 if xmax > xmin else 1000
    pad_y = (ymax - ymin) * 0.08 if ymax > ymin else 1000
    xmin -= pad_x
    xmax += pad_x
    ymin -= pad_y
    ymax += pad_y

    xx, yy = np.mgrid[xmin:xmax:complex(gridsize), ymin:ymax:complex(gridsize)]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])

    if bandwidth_meters is not None:
        # Convert a fixed metric bandwidth to a Scott-factor relative to data spread.
        # RMS std gives an isotropic approximation when x/y scales are similar (they
        # are in EPSG:3857 for mid-latitude cities like Minneapolis/Saint Paul).
        rms_std = np.sqrt(0.5 * (np.var(x) + np.var(y)))
        rms_std = max(rms_std, 1.0)
        bw_factor = (bandwidth_meters * bandwidth_adjust) / rms_std
        kde = gaussian_kde(values, bw_method=bw_factor)
    else:
        kde = gaussian_kde(values)
        kde.set_bandwidth(kde.factor * bandwidth_adjust)

    zz = np.reshape(kde(positions).T, xx.shape)
    return xx, yy, zz


def plot_density_map(
    gdf: gpd.GeoDataFrame,
    title: str,
    subtitle: str,
    outfile: str,
    gridsize=2500,
    bandwidth_adjust=1.0,
    bandwidth_meters: float = 250.0,
    percentile_cap=99.0,
    dpi=300,
    figsize=(20, 24),
    basemap_zoom=14,
    overlay_alpha: float = 0.82,
    sharpen_sigma: float = 4.0,
    city_boundary_gdf: gpd.GeoDataFrame | None = None,
):
    if gdf.empty:
        raise ValueError(f"No data to plot for {title}")

    gdf_proj = gdf.to_crs(PROJECT_CRS)

    xx, yy, zz = kde_surface(gdf_proj, gridsize=gridsize, bandwidth_adjust=bandwidth_adjust, bandwidth_meters=bandwidth_meters)

    if sharpen_sigma > 0:
        # Unsharp mask: amplify edges in the density surface without changing
        # the statistical model. sigma is in KDE grid cells; 4.0 ≈ 50m at
        # gridsize=2500 over a 30km city extent.
        blurred = gaussian_filter(zz, sigma=sharpen_sigma)
        zz = np.clip(zz + 0.7 * (zz - blurred), 0, None)

    vmax = np.nanpercentile(zz, percentile_cap)

    # Prefer real city boundary; fall back to concave hull of data points
    if city_boundary_gdf is not None:
        boundary_geom = city_boundary_gdf.to_crs(PROJECT_CRS).geometry.union_all()
    else:
        union_geom = gdf_proj.geometry.union_all()
        if HAS_CONCAVE_HULL:
            boundary_geom = shapely_concave_hull(union_geom, ratio=0.3, allow_holes=False)
        else:
            boundary_geom = union_geom.convex_hull

    boundary_gdf = gpd.GeoDataFrame(geometry=[boundary_geom], crs=PROJECT_CRS)
    map_extent = box(xx.min(), yy.min(), xx.max(), yy.max())
    outside_geom = map_extent.difference(boundary_geom)
    outside_gdf = gpd.GeoDataFrame(geometry=[outside_geom], crs=PROJECT_CRS)

    fig, ax = plt.subplots(figsize=figsize)

    # Optional basemap — explicit zoom keeps tile detail matched to print scale
    if HAS_CONTEXTILY:
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        cx.add_basemap(ax, crs=PROJECT_CRS, source=cx.providers.CartoDB.Positron, zoom=basemap_zoom)
    else:
        ax.set_facecolor("#e8e4dd")

    im = ax.imshow(
        np.rot90(zz),
        cmap=CRIME_CMAP,
        norm=Normalize(vmin=0, vmax=vmax),
        interpolation="nearest",
        alpha=overlay_alpha,
        extent=[xx.min(), xx.max(), yy.min(), yy.max()],
    )

    # Gray hatch outside city boundary — low alpha so basemap shows through
    outside_gdf.plot(ax=ax, color="#999999", alpha=0.15, hatch="////", edgecolor="#999999", linewidth=0)
    # City boundary line — solid when using real boundary, dashed when falling back to hull
    line_style = "-" if city_boundary_gdf is not None else "--"
    boundary_gdf.boundary.plot(ax=ax, color="#222222", linewidth=1.5, linestyle=line_style, alpha=0.9)

    ax.set_title(title, fontsize=28, weight="bold", pad=16)
    ax.text(
        0.5, 0.98, subtitle,
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=14,
    )

    date_note = (
        f"Data: {SINCE_UTC.strftime('%b %d, %Y')} – {PULLED_UTC.strftime('%b %d, %Y')}"
        f"  •  pulled {PULLED_UTC.strftime('%Y-%m-%d')}"
    )
    ax.text(
        0.5, 0.01, date_note,
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=11, color="#555555",
    )
    ax.text(
        1.0, 0.01,
        "Made by Colin Catlin. Data may have errors, confirm with other sources.",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=7, color="#888888",
    )

    ax.set_axis_off()

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("% of peak density", fontsize=13)
    tick_fracs = [0.02, 0.05, 0.10, 0.20, 0.40, 0.70, 1.00]
    cbar.set_ticks([f * vmax for f in tick_fracs])
    cbar.set_ticklabels(["2%", "5%", "10%", "20%", "40%", "70%", "100%"])
    cbar.ax.tick_params(labelsize=11)

    boundary_label = "City boundary" if city_boundary_gdf is not None else "Dataset boundary (approx.)"
    legend_elements = [
        Line2D([0], [0], color="#222222", linewidth=1.5, linestyle=line_style, label=boundary_label),
        mpatches.Patch(facecolor="#999999", alpha=0.15, hatch="////", edgecolor="#999999", label="Outside city boundary"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=11, framealpha=0.75)

    plt.tight_layout()
    plt.savefig(outfile, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# =========================
# PULL + PREP
# =========================

def fetch_city_boundary(layer_url: str, target_crs: str = PROJECT_CRS) -> gpd.GeoDataFrame | None:
    """
    Fetch all polygons from a neighborhood/district layer, dissolve them into
    a single city-boundary polygon, and return it in target_crs.
    Returns None on any fetch failure so callers can fall back gracefully.
    """
    try:
        meta = get_layer_metadata(layer_url)
        max_rc = min(meta.get("maxRecordCount", 1000), 2000)
        params = {
            "where": "1=1",
            "outFields": "OBJECTID",
            "returnGeometry": "true",
            "outSR": 4326,
            "resultOffset": 0,
            "resultRecordCount": max_rc,
            "f": "geojson",
        }
        r = requests.get(f"{layer_url}/query", params=params, timeout=60)
        r.raise_for_status()
        gdf = gpd.GeoDataFrame.from_features(r.json()["features"], crs="EPSG:4326")
        if gdf.empty:
            return None
        if len(gdf) >= max_rc:
            warnings.warn(
                f"fetch_city_boundary: got {len(gdf)} polygons = page limit ({max_rc}); "
                "boundary may be incomplete. Increase max_rc or paginate if needed."
            )
        dissolved = gdf.dissolve().to_crs(target_crs)
        # Reject non-polygon geometry (e.g. a polyline "boundary line" layer) —
        # a line union has no area so map_extent.difference(line) = the whole map.
        geom_types = set(dissolved.geometry.geom_type.dropna())
        if not geom_types.issubset({"Polygon", "MultiPolygon"}):
            warnings.warn(
                f"fetch_city_boundary: layer returned {geom_types} geometry (not polygon) "
                f"— skipping {layer_url}. Will fall back to point hull."
            )
            return None
        print(f"  City boundary fetched: {len(gdf)} polygons dissolved from {layer_url}")
        return dissolved
    except Exception as e:
        warnings.warn(f"Could not fetch city boundary from {layer_url}: {e}")
        return None


def probe_dataset_fields(cfg_key: str) -> None:
    """
    Fetch 5 records and print every field name + sample value so field
    mismatches can be spotted without a full data pull.
    """
    cfg = DATASETS[cfg_key]
    layer_url = get_layer_url(cfg)
    params = {
        "where": "1=1", "outFields": "*", "returnGeometry": "false",
        "resultOffset": 0, "resultRecordCount": 5, "f": "json",
    }
    data = arcgis_json(f"{layer_url}/query", params)
    feats = data.get("features", [])
    if not feats:
        print(f"[{cfg_key}] probe: no features returned")
        return
    print(f"\n[{cfg_key}] field probe (5 records):")
    all_keys = list(feats[0].get("attributes", {}).keys())
    print(f"  fields: {all_keys}")
    for i, feat in enumerate(feats):
        attrs = feat.get("attributes", {})
        print(f"  record {i}: { {k: str(v)[:40] for k, v in attrs.items()} }")


def load_dataset(cfg_key: str) -> tuple[gpd.GeoDataFrame, dict]:
    cfg = DATASETS[cfg_key]
    layer_url = get_layer_url(cfg)
    print(f"\n[{cfg_key}] using layer: {layer_url}")

    nb_url = cfg.get("neighborhood_layer_url")
    nb_id = cfg.get("neighborhood_id_field")
    rec_nb = cfg.get("record_neighborhood_field")
    has_nb_fallback = bool(nb_url and nb_id and rec_nb)

    city_bounds = cfg.get("city_bounds")
    df = fetch_all_features(layer_url)
    gdf, info = prepare_gdf(df, require_geometry=not has_nb_fallback, city_bounds=city_bounds)

    if has_nb_fallback:
        gdf = assign_coords_from_neighborhood(gdf, nb_url, nb_id, rec_nb)

    print(f"[{cfg_key}] rows after date/geometry filtering: {len(gdf):,}")
    print(f"[{cfg_key}] chosen fields: {info}")
    return gdf, info


def _cache_path(name: str) -> str:
    return os.path.join(CACHE_DIR, f"{name}.parquet")


def _save_cache(name: str, gdf: gpd.GeoDataFrame) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _cache_path(name)
    tmp = path + ".tmp"
    gdf.to_parquet(tmp)
    os.replace(tmp, path)  # atomic on POSIX; safe if process is killed mid-write
    print(f"  Cached {len(gdf):,} rows → {path}")


_CACHE_REQUIRED_COLS = {
    "mpls_crime":       {"_date", "_offense_text", "_all_text", "_cat_violent", "_cat_gun",
                         "_cat_residential_burglary", "_cat_generic_burglary"},
    "stp_crime":        {"_date", "_offense_text", "_all_text", "_cat_violent", "_cat_gun",
                         "_cat_residential_burglary", "_cat_generic_burglary"},
    "roseville_crime":  {"_date", "_offense_text", "_all_text", "_cat_violent", "_cat_gun",
                         "_cat_residential_burglary", "_cat_generic_burglary"},
    "mpls_shots":       {"_date"},
    "boundary_mpls":      set(),
    "boundary_stp":       set(),
    "boundary_roseville": set(),
}


def _load_cache(name: str) -> gpd.GeoDataFrame | None:
    path = _cache_path(name)
    if not os.path.exists(path):
        return None
    gdf = gpd.read_parquet(path)

    # Validate schema — if required columns are missing the cache is stale/corrupt.
    required = _CACHE_REQUIRED_COLS.get(name, set())
    missing = required - set(gdf.columns)
    if missing:
        warnings.warn(
            f"Cache {path} is missing columns {missing} — discarding and re-downloading."
        )
        return None

    # Show age and date range so stale caches are obvious
    mtime = datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc)
    age_days = (datetime.now(timezone.utc) - mtime).days
    if "_date" in gdf.columns:
        d_min = gdf["_date"].min()
        d_max = gdf["_date"].max()
        print(f"  Loaded {len(gdf):,} rows from cache ({age_days}d old, data {d_min:%Y-%m-%d}–{d_max:%Y-%m-%d}): {path}")
    else:
        print(f"  Loaded {len(gdf):,} rows from cache ({age_days}d old): {path}")
    return gdf


def _fetch_or_cache(
    cfg_key: str,
    cache_name: str,
    force: bool,
    probe: bool = False,
) -> gpd.GeoDataFrame:
    """
    Load from parquet cache when available and force=False; otherwise download,
    annotate canonical categories, and save to cache.  Returns an annotated GDF.
    Raises on download failure — callers that want a fallback empty GDF should
    catch explicitly.
    """
    if not force:
        cached = _load_cache(cache_name)
        if cached is not None:
            return cached

    if probe:
        probe_dataset_fields(cfg_key)

    gdf, _ = load_dataset(cfg_key)
    gdf = annotate_canonical_categories(gdf)
    _save_cache(cache_name, gdf)
    return gdf


def _hull_boundary(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame | None:
    """Derive a city-boundary polygon from the convex/concave hull of crime points."""
    proj = gdf.to_crs(PROJECT_CRS)
    if proj.empty:
        return None
    union_geom = proj.geometry.union_all()
    if HAS_CONCAVE_HULL:
        boundary_geom = shapely_concave_hull(union_geom, ratio=0.3, allow_holes=False)
    else:
        boundary_geom = union_geom.convex_hull
    return gpd.GeoDataFrame(geometry=[boundary_geom], crs=PROJECT_CRS)


def main():
    using_cache = not FORCE_DOWNLOAD

    # City boundaries are small and stable; cache them separately so a fully
    # cached run makes zero crime-data API calls.
    print("\nLoading city boundaries…")
    mpls_boundary      = None
    stp_boundary       = None
    roseville_boundary = None
    if not FORCE_DOWNLOAD:
        mpls_boundary      = _load_cache("boundary_mpls")
        stp_boundary       = _load_cache("boundary_stp")
        roseville_boundary = _load_cache("boundary_roseville")

    if mpls_boundary is None:
        mpls_boundary = fetch_city_boundary(DATASETS["minneapolis_crime"]["boundary_layer_url"])
        if mpls_boundary is not None:
            _save_cache("boundary_mpls", mpls_boundary)

    if stp_boundary is None:
        stp_boundary = fetch_city_boundary(DATASETS["stpaul_crime"]["boundary_layer_url"])
        if stp_boundary is not None:
            _save_cache("boundary_stp", stp_boundary)

    if roseville_boundary is None:
        rv_boundary_url = DATASETS["roseville_crime"].get("boundary_layer_url")
        if rv_boundary_url:
            roseville_boundary = fetch_city_boundary(rv_boundary_url)
            if roseville_boundary is not None:
                _save_cache("boundary_roseville", roseville_boundary)
    # If GIS fetch failed, boundary is derived from point hull after crime data loads below.

    # Each dataset is fetched/cached independently so a missing cache for one
    # city does not force a re-download of the other.
    print("\nLoading Minneapolis crime data…")
    mpls_crime = _fetch_or_cache("minneapolis_crime", "mpls_crime", FORCE_DOWNLOAD, probe=not using_cache)

    print("\nLoading Saint Paul crime data…")
    stp_crime = _fetch_or_cache("stpaul_crime", "stp_crime", FORCE_DOWNLOAD, probe=not using_cache)

    print("\nLoading Roseville crime data…")
    try:
        roseville_crime = _fetch_or_cache("roseville_crime", "roseville_crime", FORCE_DOWNLOAD, probe=not using_cache)
    except Exception as e:
        warnings.warn(f"Could not load Roseville crime dataset: {e}")
        roseville_crime = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")

    # Derive Roseville boundary from point hull if not cached yet
    if roseville_boundary is None and not roseville_crime.empty:
        roseville_boundary = _hull_boundary(roseville_crime)
        if roseville_boundary is not None:
            _save_cache("boundary_roseville", roseville_boundary)

    print_alignment_audit("Minneapolis", mpls_crime)
    print_alignment_audit("Saint Paul", stp_crime)
    print_alignment_audit("Roseville", roseville_crime)
    print_cross_city_alignment_summary(mpls_crime, stp_crime)

    # Shots-fired is supplemental; an empty fallback is acceptable if it fails.
    print("\nLoading Minneapolis shots-fired data…")
    try:
        mpls_shots = _fetch_or_cache("minneapolis_shots", "mpls_shots", FORCE_DOWNLOAD)
    except Exception as e:
        warnings.warn(f"Could not load Minneapolis shots dataset: {e}")
        mpls_shots = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")

    # Theme 1: violent + gun-related
    mpls_violent = filter_violent(mpls_crime)
    mpls_gun = filter_gun(mpls_crime)
    mpls_violent_gun_core = pd.concat([mpls_violent, mpls_gun], ignore_index=True)
    mpls_violent_gun_core = gpd.GeoDataFrame(mpls_violent_gun_core, geometry="geometry", crs="EPSG:4326")

    if not mpls_shots.empty:
        mpls_violent_gun = pd.concat([mpls_violent_gun_core, mpls_shots], ignore_index=True)
        mpls_violent_gun = gpd.GeoDataFrame(mpls_violent_gun, geometry="geometry", crs="EPSG:4326")
    else:
        mpls_violent_gun = mpls_violent_gun_core.copy()

    stp_violent = filter_violent(stp_crime)
    stp_gun = filter_gun(stp_crime)
    stp_violent_gun = pd.concat([stp_violent, stp_gun], ignore_index=True)
    stp_violent_gun = gpd.GeoDataFrame(stp_violent_gun, geometry="geometry", crs="EPSG:4326")

    roseville_violent = filter_violent(roseville_crime)
    roseville_gun = filter_gun(roseville_crime)
    roseville_violent_gun = pd.concat([roseville_violent, roseville_gun], ignore_index=True)
    roseville_violent_gun = gpd.GeoDataFrame(roseville_violent_gun, geometry="geometry", crs="EPSG:4326")

    # Metro uses crime-reports only (no ShotSpotter) so all cities are equally represented
    metro_violent_gun = pd.concat([mpls_violent_gun_core, stp_violent_gun, roseville_violent_gun], ignore_index=True)
    metro_violent_gun = gpd.GeoDataFrame(metro_violent_gun, geometry="geometry", crs="EPSG:4326")

    # Theme 2: residential burglary
    mpls_burg      = filter_residential_burglary(mpls_crime)
    stp_burg       = filter_residential_burglary(stp_crime)
    roseville_burg = filter_residential_burglary(roseville_crime)

    metro_burg = pd.concat([mpls_burg, stp_burg, roseville_burg], ignore_index=True)
    metro_burg = gpd.GeoDataFrame(metro_burg, geometry="geometry", crs="EPSG:4326")

    # Combined metro boundary — union of all available city boundaries
    boundary_parts = [b for b in [mpls_boundary, stp_boundary, roseville_boundary] if b is not None]
    if len(boundary_parts) >= 2:
        metro_boundary = gpd.GeoDataFrame(
            pd.concat(boundary_parts, ignore_index=True),
            crs=PROJECT_CRS,
        )
    elif boundary_parts:
        metro_boundary = boundary_parts[0]
    else:
        metro_boundary = None

    # Drop duplicate geometry+date rows after unioning categories
    for name, gdf in {
        "mpls_violent_gun":      mpls_violent_gun,
        "stp_violent_gun":       stp_violent_gun,
        "roseville_violent_gun": roseville_violent_gun,
        "metro_violent_gun":     metro_violent_gun,
        "mpls_burg":             mpls_burg,
        "stp_burg":              stp_burg,
        "roseville_burg":        roseville_burg,
        "metro_burg":            metro_burg,
    }.items():
        if not gdf.empty:
            cols = [c for c in ["_date", "_lon", "_lat", "_offense_text", "_detail_text"] if c in gdf.columns]
            deduped = gdf.drop_duplicates(subset=cols).copy()
            if name == "mpls_violent_gun":
                mpls_violent_gun = deduped
            elif name == "stp_violent_gun":
                stp_violent_gun = deduped
            elif name == "roseville_violent_gun":
                roseville_violent_gun = deduped
            elif name == "metro_violent_gun":
                metro_violent_gun = deduped
            elif name == "mpls_burg":
                mpls_burg = deduped
            elif name == "stp_burg":
                stp_burg = deduped
            elif name == "roseville_burg":
                roseville_burg = deduped
            elif name == "metro_burg":
                metro_burg = deduped

    # Metro maps: wider figure, lower basemap zoom for regional context.
    # 500 m bandwidth — slightly wider than single-city to keep hotspots visible
    # at the larger viewing scale without smearing across neighborhoods.
    METRO_KW = {"figsize": (28, 22), "gridsize": 3000, "bandwidth_meters": 350, "basemap_zoom": 13}

    METRO_TITLE_SUFFIX = "Minneapolis + Saint Paul + Roseville"

    # (gdf, title, outfile, extra_kwargs, city_boundary_gdf)
    outputs = [
        (mpls_violent_gun,  "Minneapolis — Violent + Gun-Related Crime Density",                              "minneapolis_violent_gun_density.png", {},        mpls_boundary),
        (stp_violent_gun,   "Saint Paul — Violent + Gun-Related Crime Density",                               "saint_paul_violent_gun_density.png",  {},        stp_boundary),
        (metro_violent_gun, f"{METRO_TITLE_SUFFIX} — Violent + Gun-Related Crime Density",                   "metro_violent_gun_density.png",       METRO_KW,  metro_boundary),
        (mpls_burg,         "Minneapolis — Residential Burglary Density",                                     "minneapolis_burglary_density.png",    {},        mpls_boundary),
        (stp_burg,          "Saint Paul — Residential Burglary Density",                                      "saint_paul_burglary_density.png",     {},        stp_boundary),
        (metro_burg,        f"{METRO_TITLE_SUFFIX} — Residential Burglary Density",                          "metro_burglary_density.png",          METRO_KW,  metro_boundary),
    ]

    created = []
    for gdf_plot, title, outfile, extra_kw, boundary in outputs:
        print(f"\nRendering: {title}")
        plot_density_map(
            gdf_plot, title,
            f"Last {YEARS_BACK} years • smoothed density • relative intensity",
            outfile,
            city_boundary_gdf=boundary,
            **extra_kw,
        )
        created.append(outfile)
        print(f"  → {outfile}")

    print("\nDone. Created:")
    for f in created:
        print(f"  {f}")


if __name__ == "__main__":
    main()
