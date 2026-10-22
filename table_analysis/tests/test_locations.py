"""Tests for src/locations.py on synthetic data only (no participant data).

Run from table_analysis/:  python -m pytest tests/
"""
import geopandas as gpd
import pytest
import pandas as pd
from shapely.geometry import box

from src.locations import normalize_place, resolve_cluster_assignments

# Three fake clusters near Lomé; two share the name "Kpota" (names aren't unique, ids are).
SETTLEMENTS = gpd.GeoDataFrame(
    {"geom_id": ["id_agb", "id_tse", "id_kp1"], "Name": ["Agbodrafo", "Tsévié", "Kpota"]},
    geometry=[box(1.20, 6.20, 1.21, 6.21), box(1.26, 6.20, 1.27, 6.21), box(1.40, 6.20, 1.41, 6.21)],
    crs="epsg:4326",
)
SETTLEMENTS = pd.concat([SETTLEMENTS, gpd.GeoDataFrame(
    {"geom_id": ["id_kp2"], "Name": ["KPOTA"]}, geometry=[box(1.50, 6.20, 1.51, 6.21)], crs="epsg:4326")],
    ignore_index=True)


def _data(rows):
    return pd.DataFrame(rows, columns=["KEY", "gps-Latitude", "gps-Longitude", "location"])


def _by_key(data, res):
    return dict(zip(data["KEY"], zip(res["cluster_id"], res["location_source"])))


def test_normalize_place_absorbs_accents_case_and_mojibake():
    assert normalize_place("Tsévié") == normalize_place("TSEVIE") == normalize_place("TsËviË")
    assert pd.isna(normalize_place(None))


def test_priority_and_sources():
    data = _data([
        ("k_gps", 6.205, 1.205, None),        # inside Agbodrafo
        ("k_qa", 6.30, 1.90, None),           # far from everything; QA says Tsévié
        ("k_snap", 6.205, 1.2145, None),      # ~500m east of Agbodrafo, no QA note
        ("k_none", 6.30, 1.90, None),         # far, no QA, no label
        ("k_conflict", 6.205, 1.205, None),   # inside Agbodrafo but QA says Tsévié
        ("k_override", 6.30, 1.90, None),     # researcher override by id
        ("k_badqa", 6.30, 1.90, None),        # QA text that isn't a cluster
        ("k_label", 6.30, 1.90, "tsevie"),    # only the survey's own location label
        ("k_ambig", 6.30, 1.90, "Kpota"),     # label matches two clusters -> not guessed
    ])
    qa = pd.Series({"k_qa": "tsevie", "k_conflict": "Tsévié ", "k_badqa": "Somewhere else"})
    overrides = pd.Series({"k_override": "id_agb"})

    res = resolve_cluster_assignments(data, SETTLEMENTS, qa_locations=qa, overrides=overrides, label_col="location")
    got = _by_key(data, res)

    assert got["k_gps"] == ("id_agb", "gps")
    assert got["k_qa"] == ("id_tse", "qa")
    assert got["k_snap"] == ("id_agb", "gps_snap")
    assert got["k_none"][1] == "unresolved" and pd.isna(got["k_none"][0])
    assert got["k_conflict"] == ("id_agb", "gps")   # GPS wins by default
    assert got["k_override"] == ("id_agb", "override")
    assert got["k_badqa"][1] == "unresolved"
    assert got["k_label"] == ("id_tse", "label")
    assert got["k_ambig"][1] == "unresolved"
    assert res.loc[data["KEY"] == "k_badqa", "qa_location_unmatched"].item()
    assert res.loc[data["KEY"] == "k_ambig", "label_unmatched"].item()
    assert res.loc[data["KEY"] == "k_qa", "cluster_name"].item() == "Tsévié"

    res_qa = resolve_cluster_assignments(data, SETTLEMENTS, qa_locations=qa, qa_overrides_gps=True)
    assert res_qa.loc[data["KEY"] == "k_conflict", "cluster_id"].item() == "id_tse"


def test_same_name_different_polygons_resolved_by_gps():
    data = _data([("a", 6.205, 1.405, "Kpota"), ("b", 6.205, 1.505, "Kpota")])
    res = resolve_cluster_assignments(data, SETTLEMENTS, label_col="location")
    assert res["cluster_id"].tolist() == ["id_kp1", "id_kp2"]


def test_02a_notebook_end_to_end_on_synthetic_data(tmp_path):
    """Executes 02a's code cells with every config path redirected to synthetic files."""
    import json
    from pathlib import Path

    # Sampled clusters (grouped names, as in togo_sample_58.geojson) and the field schedule with GPS
    gpd.GeoDataFrame(
        {"name": ["Kévé; Assahoun", "Oyo; Atakpamé", "Kpota"], "ADM1_FR": ["Maritime", "Plateaux", "Maritime"],
         "population": [1000, 2000, 3000], "proportion": [0.1, 0.2, 0.3]},
        geometry=[box(1.0, 6.0, 1.1, 6.1), box(1.2, 6.0, 1.3, 6.1), box(1.4, 6.0, 1.5, 6.1)],
        crs="epsg:4326",
    ).to_file(tmp_path / "sample.geojson")
    pd.DataFrame({"Settlement": ["Assahoun", "Assahoun", "Atakpamé", "Lomnava", ""],    # repeats, blanks
                  "Region": ["Maritime", "Maritime", "Plateaux", "Maritime", ""],
                  "GPS coordinate": ["6.05, 1.05", "6.05, 1.05", "6.05, 1.25", "6.105, 1.45", ""]}) \
        .to_csv(tmp_path / "schedule.csv", index=False)
    _data([("a", 6.05, 1.05, "Assahoun"),      # name, GPS agrees
           ("b", 6.05, 1.15, "Lomnava"),       # name via the schedule; GPS outside every cluster
           ("c", 6.05, 1.45, "Assahoun"),      # GPS inside Kpota, ~40 km from Assahoun's cluster
           ("d", 6.50, 1.90, "Nowhere"),       # override
           ("e", 6.50, 1.90, "Nowhere")]) \
        .assign(province="Maritime").to_csv(tmp_path / "mapped.csv", index=False)
    pd.DataFrame({"KEY": ["d"], "location": ["Oyo; Atakpamé"]}).to_csv(tmp_path / "overrides.csv", index=False)
    # Researcher's review: respondents recorded as Lomnava (placed by name in Kpota) are kept as "Kpota"
    pd.DataFrame({"location": ["Lomnava"], "province": ["Maritime"], "sample_cluster": ["Kpota"],
                  "sample_province": ["Maritime"], "location_source": ["name"], "selected_name": ["Kpota"]}) \
        .to_csv(tmp_path / "reviewed.csv", index=False)

    redirect = dict(
        DIR_SETTLEMENTS_SAMPLED=tmp_path / "sample.geojson", DIR_FIELD_SCHEDULE=tmp_path / "schedule.csv",
        SETTLEMENTS_SAMPLED_RENAME={"name": "Name", "ADM1_FR": "province"}, CLUSTER_KEY="Name",
        DIR_MAPPED_DATA=tmp_path / "mapped.csv", DIR_MANUAL_LOCATION_OVERRIDES=tmp_path / "overrides.csv",
        DIR_VALIDATE=tmp_path, DIR_LOCATED_DATA=tmp_path / "located.csv", UTM_CRS="epsg:32631",
        DIR_LOCATION_REVIEW=tmp_path / "reviewed.csv", DIR_SETTLEMENTS_CHOSEN=tmp_path / "clusters.csv",
    )
    nb = Path(__file__).resolve().parents[1] / "02a_correct_locations.ipynb"
    cells = [c for c in json.loads(nb.read_text())["cells"] if c["cell_type"] == "code"]
    g = {}
    exec("".join(cells[0]["source"]), g)            # imports (incl. real config paths)
    g.update(redirect)                              # ...replaced before anything is read
    for cell in cells[1:]:
        exec("".join(cell["source"]), g)

    out = pd.read_csv(tmp_path / "located.csv").set_index("KEY")
    cols = ["location", "sample_cluster", "sample_province", "sample_proportion", "location_source"]
    assert out.loc["a", cols].tolist() == ["Assahoun", "Kévé; Assahoun", "Maritime", 0.1, "gps"]  # manual name kept
    assert out.loc["b", ["sample_cluster", "location_source"]].tolist() == ["Kpota", "name"]
    assert out.loc["c", ["sample_cluster", "location_source"]].tolist() == ["Kpota", "gps"]      # spatial join wins
    assert out.loc["d", ["sample_cluster", "location_source"]].tolist() == ["Oyo; Atakpamé", "override"]
    assert out.loc["e", "location_source"] == "unresolved" and pd.isna(out.loc["e", "sample_cluster"])  # kept, no cluster
    # Final clusters: spatial-join name, else manual-review name, with the researcher's review applied
    assert out["cluster_name"].to_dict() == {"a": "Kévé; Assahoun", "b": "Kpota", "c": "Kpota",
                                             "d": "Nowhere", "e": "Nowhere"}
    clusters = pd.read_csv(tmp_path / "clusters.csv").set_index("Name")
    assert set(clusters.index) == {"Kévé; Assahoun", "Kpota", "Nowhere"}
    assert clusters.loc["Kpota", ["respondents", "sampled_cluster"]].tolist() == [2, True]
    assert pd.isna(clusters.loc["Nowhere", "point"])                  # not sampled, not in the schedule
    assert pd.read_csv(tmp_path / "02a_unresolved_locations.csv")["KEY"].tolist() == ["e"]
    mapping = pd.read_csv(tmp_path / "02a_manual_review_vs_sample.csv")
    assert {"location", "province", "sample_cluster", "sample_province", "n"} <= set(mapping.columns)
    selected = mapping.set_index(["location", "sample_cluster"])["selected_name"]
    assert selected[("Assahoun", "Kpota")] == "Kpota"                 # spatial-join name
    assert selected[("Lomnava", "Kpota")] == "Lomnava"                # no spatial join: manual-review name
    row = mapping[(mapping["location"] == "Assahoun") & (mapping["sample_cluster"] == "Kévé; Assahoun")].iloc[0]
    assert (row["review_lat"], row["review_lon"]) == (6.05, 1.05)          # schedule's settlement coordinate
    assert row["sample_lat"] == pytest.approx(6.05) and row["sample_lon"] == pytest.approx(1.05)  # polygon centre
    assert row["review_to_sample_km"] == pytest.approx(0, abs=0.1)


def test_missing_gps_falls_back_to_qa():
    data = _data([("k1", None, None, None)])
    res = resolve_cluster_assignments(data, SETTLEMENTS, qa_locations=pd.Series({"k1": "Agbodrafo"}))
    assert res.loc[0, "cluster_id"] == "id_agb" and res.loc[0, "location_source"] == "qa"
