"""Tests for the sampled-cluster linking in src/locations.py (synthetic data only, no participant data).

Run from table_analysis/:  python -m pytest tests/
"""
import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from src.locations import assign_sampled_clusters, link_schedule

# Three sampled clusters ~11 km apart (0.1 degree boxes), named as in the sample file
SAMPLE = gpd.GeoDataFrame(
    {"Name": ["Kévé; Assahoun", "Oyo; Atakpamé", "Kpota"], "province": ["Maritime", "Plateaux", "Maritime"],
     "population": [1000, 2000, 3000], "proportion": [0.1, 0.2, 0.3]},
    geometry=[box(1.0, 6.0, 1.1, 6.1), box(1.2, 6.0, 1.3, 6.1), box(1.4, 6.0, 1.5, 6.1)],
    crs="epsg:4326",
)
SCHEDULE = pd.DataFrame({
    "Settlement": ["Assahoun", "Assahoun", "Atakpamé", "Lomnava", "Edzi"],
    "Region": ["Maritime", "Maritime", "Plateaux", "Maritime", "Maritime"],
    "GPS coordinate": ["6.05, 1.05", "6.05, 1.05", "6.05, 1.25", "6.105, 1.45", ""],  # Lomnava ~0.5 km outside Kpota
})


def test_schedule_settlements_take_the_sampled_cluster_they_fall_in():
    links = link_schedule(SCHEDULE, SAMPLE).set_index("settlement")
    assert len(links) == 4                                            # repeated visit day kept once
    assert links.loc["Assahoun", ["sample_cluster", "sample_province", "how"]].tolist() == ["Kévé; Assahoun", "Maritime", "gps inside"]
    assert links.loc["Atakpamé", "sample_cluster"] == "Oyo; Atakpamé"
    assert links.loc["Lomnava", ["sample_cluster", "how"]].tolist() == ["Kpota", "gps within 1000 m"]
    assert pd.isna(links.loc["Edzi", "sample_cluster"])                # no coordinate, name in no cluster


DATA = pd.DataFrame({
    "KEY": ["both", "name_gps_near", "disagree", "gps_only", "nothing", "village", "override", "snap"],
    "location": ["Assahoun", "Lomnava", "Assahoun", "Somewhere", None, "Kévé", "Assahoun", "Somewhere"],
    "province": ["Maritime"] * 8,
    "gps-Latitude":  [6.05, 6.05, 6.05, 6.05, 6.50, None, 6.05, 6.105],
    "gps-Longitude": [1.05, 1.15, 1.45, 1.45, 1.90, None, 1.05, 1.45],   # "snap": ~0.5 km outside Kpota
})


def _got(res):
    return dict(zip(DATA["KEY"], zip(res["sample_cluster"], res["location_source"])))


def test_spatial_join_wins_then_manual_review_name_then_nearest():
    res = assign_sampled_clusters(DATA, SAMPLE, link_schedule(SCHEDULE, SAMPLE), overrides=pd.Series({"override": "Kpota"}))
    got = _got(res)
    assert got["both"] == ("Kévé; Assahoun", "gps")                   # GPS and name agree
    assert got["disagree"] == ("Kpota", "gps")                        # GPS inside Kpota beats the name
    assert res.loc[DATA["KEY"] == "disagree", "name_disagrees"].item()
    assert got["name_gps_near"] == ("Kpota", "name")                  # GPS in no cluster: the name (via schedule)
    assert got["village"] == ("Kévé; Assahoun", "name")               # a village of a grouped sampled name
    assert got["snap"] == ("Kpota", "gps_snap")                       # no name link: nearest cluster within 1 km
    assert got["gps_only"] == ("Kpota", "gps")
    assert got["nothing"][1] == "unresolved"
    assert got["override"] == ("Kpota", "override")


def test_name_first_option_keeps_gps_only_for_clear_disagreements():
    res = assign_sampled_clusters(DATA, SAMPLE, link_schedule(SCHEDULE, SAMPLE), spatial_first=False)
    got = _got(res)
    assert got["both"] == ("Kévé; Assahoun", "name")
    assert got["disagree"] == ("Kpota", "gps (name disagrees)")       # inside Kpota, ~40 km from Assahoun's
