"""Tests for the 02b weighting code on synthetic data only (no participant data).

Run from table_analysis/:  python -m pytest tests/
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.funcs import load_sampling_frames, load_surveyed_clusters, parse_n, read_table

NOTEBOOK = Path(__file__).resolve().parents[1] / "02b_calculate_weights.ipynb"


def test_parse_n_uses_last_token():
    assert parse_n("1 2 3") == 2
    assert parse_n("1 2 3 4 5 6 7 8 9 10 11") == 10   # last *character* would give 0
    assert parse_n(3.0) == 2
    assert np.isnan(parse_n(np.nan)) and np.isnan(parse_n(""))


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "cp1252", "mac_roman"])
def test_read_table_detects_excel_encodings(tmp_path, encoding):
    f = tmp_path / "chosen.csv"
    f.write_bytes("geom_id,Name\ng1,Kévé\ng2,Tsévié\n".encode(encoding))
    assert read_table(f)["Name"].tolist() == ["Kévé", "Tsévié"]


def test_read_table_handles_mixed_encodings(tmp_path):
    f = tmp_path / "chosen.csv"
    f.write_bytes("geom_id,Name\n".encode() + "g1,Kévé\n".encode("utf-8") + "g2,Tsévié\n".encode("mac_roman"))
    assert read_table(f)["Name"].tolist() == ["Kévé", "Tsévié"]


def _write_frames(tmp_path, eligible_rows=None, chosen_rows=None):
    eligible = pd.DataFrame(eligible_rows or [
        # geom_id, admin area, population (text with thousands separators)
        ("g1", "Maritime", "1,000"), ("g2", "Maritime", "3,000"),
        ("g3", "Kara", "2,000"), ("g4", "Kara", "4,000"),
        ("g5", "Savanes", "5,000"),     # area outside the strata -> never sampled
        ("g6", "Kara", "100"),          # below the minimum population -> never sampled
    ], columns=["ID", "ADM1", "X_sum"])
    chosen = pd.DataFrame(chosen_rows or [
        # cluster name, geom_id, region label typed in the QA sheet (informational only)
        ("A", "g1", "maritime"), ("B", "g3", "Kara"),
    ], columns=["Cluster", "ID", "Region"])
    e, c = tmp_path / "eligible.csv", tmp_path / "chosen.csv"
    eligible.to_csv(e, index=False)
    chosen.to_csv(c, index=False)
    return e, c


ELIG_RENAME = {"ID": "geom_id", "ADM1": "province", "X_sum": "population"}
CHOSEN_RENAME = {"Cluster": "Name", "ID": "geom_id", "Region": "province"}
STRATA = {"Maritime": "Maritime", "Kara": "Kara"}
FRAME = dict(strata_map=STRATA, min_population=500, key="geom_id")


def test_loader_rebuilds_frame_and_takes_strata_from_it(tmp_path):
    e, c = _write_frames(tmp_path)
    eligible, chosen = load_sampling_frames(e, c, ELIG_RENAME, CHOSEN_RENAME, **FRAME)
    assert sorted(eligible["geom_id"]) == ["g1", "g2", "g3", "g4"]      # g5 (Savanes) and g6 (small) dropped
    assert eligible["proportion"].tolist() == eligible["population"].tolist() == [1000, 3000, 2000, 4000]
    # stratum + population come from the frame by geom_id, not from the QA sheet's label
    assert chosen.set_index("Name")["province"].to_dict() == {"A": "Maritime", "B": "Kara"}
    assert chosen.set_index("Name")["population"].to_dict() == {"A": 1000, "B": 2000}


def test_loader_explains_surveyed_clusters_missing_from_frame(tmp_path):
    e, c = _write_frames(tmp_path, chosen_rows=[("A", "g1", "x"), ("S", "g5", "x"), ("T", "g6", "x"),
                                                ("Z", "g9", "x"), ("X", "g2", "x")])
    with pytest.raises(ValueError) as err:
        load_sampling_frames(e, c, ELIG_RENAME, CHOSEN_RENAME, exclude_clusters=["g2"], **FRAME)
    msg = str(err.value)
    assert "S (g5" in msg and "T (g6" in msg and "outside the strata or below the minimum" in msg
    assert "Z (g9" in msg and "not in the frame file" in msg
    assert "X (g2" in msg and "exclusion list" in msg


def test_loader_excludes_geom_ids(tmp_path):
    e, c = _write_frames(tmp_path)
    eligible, _ = load_sampling_frames(e, c, ELIG_RENAME, CHOSEN_RENAME, exclude_clusters=["g2", "g_unknown"], **FRAME)
    assert "g2" not in set(eligible["geom_id"]) and len(eligible) == 3


def test_loader_names_missing_columns(tmp_path):
    e, c = _write_frames(tmp_path)
    with pytest.raises(KeyError, match="SETTLEMENTS_ELIGIBLE_RENAME"):
        load_sampling_frames(e, c, {}, CHOSEN_RENAME, key="geom_id")


def test_notebook_end_to_end_on_synthetic_data(tmp_path):
    """Executes 02b's code cells with every config path redirected to synthetic files."""
    e, c = _write_frames(tmp_path)
    rng = np.random.default_rng(0)
    n = 12
    pd.DataFrame({
        "KEY": [f"uuid:{i}" for i in range(n)],
        "deviceid": "dev", "gps-Latitude": rng.uniform(6, 7, n), "gps-Longitude": rng.uniform(1, 2, n),
        "gps-Altitude": 10,
        "geom_id": ["g1"] * 6 + ["g3"] * 5 + ["g9"],                 # set by 02a
        "location": ["A"] * 6 + ["B"] * 5 + ["Elsewhere"],
        "province": ["typo"] * n,                                    # survey label -- replaced by the frame's stratum
        "hh_member_index_list": ["1 2 3"] * 6 + ["1 2"] * 6,
        "random_draw_attempts": 1,
        "urban_rural": ["Urbain"] * 5 + ["Rural"] + ["Rural"] * 6,   # g1 mostly urban, g3 rural
    }).to_csv(tmp_path / "located.csv", index=False)

    redirect = dict(
        DIR_SETTLEMENTS_ELIGIBLE=e, DIR_SETTLEMENTS_CHOSEN=c,
        SETTLEMENTS_ELIGIBLE_RENAME=ELIG_RENAME, SETTLEMENTS_CHOSEN_RENAME=CHOSEN_RENAME,
        STRATA_MAP=STRATA, MIN_CLUSTER_POPULATION=500, EXCLUDED_CLUSTERS=["g4"], CLUSTER_KEY="geom_id", RESPONDENT_CLUSTER_COL="geom_id",
        DIR_LOCATED_DATA=tmp_path / "located.csv", DIR_WEIGHTED_DATA=tmp_path / "weighted.csv",
        DIR_VALIDATE=tmp_path, AVG_HOUSEHOLD_SIZE=5, USE_STRATUM_WEIGHT=False, HOUSEHOLD_SIZE_BY_SETTING={"rural": 4.8, "urban": 3.8}, URBAN_RURAL_COL="urban_rural",
    )
    cells = [c for c in json.loads(NOTEBOOK.read_text())["cells"] if c["cell_type"] == "code"]
    g = {}
    exec("".join(cells[0]["source"]), g)            # imports (incl. real config paths)
    g.update(redirect)                              # ...replaced before anything is read
    for cell in cells[1:]:
        exec("".join(cell["source"]), g)

    out = pd.read_csv(tmp_path / "weighted.csv")
    assert len(out) == 11                           # g9 respondent excluded, not crashed
    assert out["normalized_weight"].notna().all()
    assert out["normalized_weight"].mean() == pytest.approx(1.0)
    assert set(out["province"]) == {"Maritime", "Kara"}
    # post-stratified to the frame (g4 excluded): Maritime 1000 + 3000 vs Kara 2000
    share = out.groupby("province")["normalized_weight"].sum() / out["normalized_weight"].sum()
    assert share["Maritime"] == pytest.approx(2 / 3)
    assert "normalized_weight_design" in out.columns
    assert (tmp_path / "02b_respondents_without_cluster.csv").exists()
    clusters = pd.read_csv(tmp_path / "02b_cluster_weights.csv")
    assert set(clusters["cluster_key"]) == {"g1", "g3"}
    assert clusters.set_index("cluster_key")["household_size"].to_dict() == {"g1": 3.8, "g3": 4.8}


NAME_ELIG = {"name": "Name", "ADM1": "province", "X_sum": "population"}
NAME_CHOSEN = {"Settlement": "Name", "Region": "province"}


def _write_named_frames(tmp_path, chosen_rows):
    eligible = pd.DataFrame([
        # no ids: settlements are identified by name; "Kpota" exists in both strata
        ("Tsévié", "Maritime", "1,000"), ("Kpota", "Maritime", "3,000"),
        ("Kpota", "Kara", "2,000"), ("Bafilo", "Kara", "4,000"),
    ], columns=["name", "ADM1", "X_sum"])
    chosen = pd.DataFrame(chosen_rows, columns=["Settlement", "Region"])
    e, c = tmp_path / "eligible.csv", tmp_path / "chosen.csv"
    eligible.to_csv(e, index=False)
    chosen.to_csv(c, index=False)
    return e, c


def test_loader_links_on_name_and_uses_region_for_repeated_names(tmp_path):
    e, c = _write_named_frames(tmp_path, [("TSEVIE", "Maritime"), ("Kpota", "Kara")])
    eligible, chosen = load_sampling_frames(e, c, NAME_ELIG, NAME_CHOSEN, strata_map=STRATA, key="Name")
    assert len(eligible) == 4                                        # same-named settlements are both kept
    got = chosen.set_index("cluster_key")[["province", "population", "proportion"]]
    assert got.loc["tsevie"].tolist() == ["Maritime", 1000, 1000]    # accents/case ignored
    assert got.loc["kpota"].tolist() == ["Kara", 2000, 2000]         # the Kara Kpota, not Maritime's


def _grouped_frame(tmp_path):
    import geopandas as gpd
    from shapely.geometry import box
    frame = gpd.GeoDataFrame(
        {"name": ["Kévé; Assahoun", "Agou Gadzépé; Agou Tavie", "Oyo; Atakpamé", "Kpota", "Kpota"],
         "ADM1": ["Maritime", "Plateaux", "Plateaux", "Maritime", "Kara"],
         "X_sum": [1000, 2000, 3000, 4000, 5000]},
        geometry=[box(1.0, 6.0, 1.1, 6.1), box(1.2, 6.0, 1.3, 6.1), box(1.4, 6.0, 1.5, 6.1),
                  box(1.6, 6.0, 1.7, 6.1), box(1.8, 6.0, 1.9, 6.1)],
        crs="epsg:4326",
    )
    frame.to_file(tmp_path / "frame.geojson")
    return frame


GROUPED = dict(eligible_rename={"name": "Name", "ADM1": "province", "X_sum": "population"},
               chosen_rename={"Settlement": "Name", "Region": "province"},
               strata_map={"Maritime": "Maritime", "Kara": "Kara", "Plateaux": "Plateaux"},
               exclude_clusters=[], key="Name", point_col="GPS coordinate")


def test_loader_links_schedule_villages_to_grouped_frame_names_and_gps(tmp_path):
    _grouped_frame(tmp_path)
    pd.DataFrame({
        "Settlement": ["Assahoun", "Agou", "Somewhere new", "Kpota", "Kévé"],
        "Region": ["Maritime", "Plateaux", "Plateaux", "Kara", "Maritime"],
        "GPS coordinate": ["", "", "6.05, 1.45", "", ""],      # inside the Oyo; Atakpamé polygon
    }).to_csv(tmp_path / "schedule.csv", index=False)
    _, chosen = load_sampling_frames(tmp_path / "frame.geojson", tmp_path / "schedule.csv", **GROUPED)
    got = chosen.set_index("Name")[["frame_name", "population"]]                # surveyed file's names kept
    assert got.loc["Assahoun"].tolist() == ["Kévé; Assahoun", 1000]             # one of the grouped villages
    assert got.loc["Agou"].tolist() == ["Agou Gadzépé; Agou Tavie", 2000]       # whole word of a village
    assert got.loc["Somewhere new"].tolist() == ["Oyo; Atakpamé", 3000]         # by GPS point
    assert got.loc["Kpota"].tolist() == ["Kpota", 5000]                         # region picks the Kara one
    assert len(chosen) == 4                                                     # Kévé = same cluster as Assahoun
    assert "geometry" in chosen.columns                                         # polygons kept for 02a


def test_loader_links_sample_polygons_by_location(tmp_path):
    import geopandas as gpd
    from shapely.geometry import box
    _grouped_frame(tmp_path)
    gpd.GeoDataFrame({"name": ["renamed in the sample file"]}, geometry=[box(1.62, 6.02, 1.68, 6.08)],
                     crs="epsg:4326").to_file(tmp_path / "sample.geojson")
    _, chosen = load_sampling_frames(tmp_path / "frame.geojson", tmp_path / "sample.geojson",
                                     **{**GROUPED, "chosen_rename": {"name": "Name"}})
    assert chosen[["Name", "frame_name", "province", "population"]].values.tolist() == [
        ["renamed in the sample file", "Kpota", "Maritime", 4000]]
    # 02a reads the same file without the frame and gets the same names / keys
    alone = load_surveyed_clusters(tmp_path / "sample.geojson", rename={"name": "Name"}, key="Name")
    assert alone[["Name", "cluster_key"]].values.tolist() == chosen[["Name", "cluster_key"]].values.tolist()


def test_loader_reports_names_it_cannot_tell_apart(tmp_path):
    e, c = _write_named_frames(tmp_path, [("Kpota", "")])
    with pytest.raises(ValueError, match="more than one frame settlement"):
        load_sampling_frames(e, c, NAME_ELIG, NAME_CHOSEN, strata_map=STRATA, key="Name")


def _pps_weighted_share_of_a(use_stratum_weight):
    """Stratum A: 25% of the population but 4 of 5 clusters (over-allocated). Returns A's weighted share."""
    from src.funcs import calculate_three_stage_weights
    eligible = pd.DataFrame({"cluster_key": [f"a{i}" for i in range(50)] + [f"b{i}" for i in range(50)],
                             "province": ["A"] * 50 + ["B"] * 50, "population": [100] * 50 + [300] * 50})
    eligible["proportion"] = eligible["population"]
    chosen = eligible[eligible["cluster_key"].isin(["a0", "a1", "a2", "a3", "b0"])].assign(Name=lambda d: d["cluster_key"])
    data = pd.DataFrame({"KEY": range(25), "cluster_key": [k for k in chosen["cluster_key"] for _ in range(5)]})
    data["household_id"] = data["KEY"]
    data["eligible_adults_in_hh"] = 1
    out, _ = calculate_three_stage_weights(eligible, chosen.copy(), data, avg_household_size=5,
                                           use_stratum_weight=use_stratum_weight)
    return out.groupby("province")["normalized_weight"].sum()["A"] / out["normalized_weight"].sum()


def test_cluster_weight_alone_reproduces_stratum_population_shares():
    assert _pps_weighted_share_of_a(False) == pytest.approx(0.25, abs=0.01)   # population share
    assert _pps_weighted_share_of_a(True) < 0.05                              # stratum weight double-corrects


def _pps_example(stage2_population_scale=1.0):
    """Two strata, PPS clusters, 5 households each. stage2_population_scale != 1 makes the stage 2
    population differ from the stage 1 size measure (the failure check_stage_weights looks for)."""
    from src.funcs import calculate_three_stage_weights
    pops = [100, 200, 400, 800, 1600] * 10
    eligible = pd.DataFrame({"cluster_key": [f"a{i}" for i in range(50)] + [f"b{i}" for i in range(50)],
                             "province": ["A"] * 50 + ["B"] * 50, "population": pops + pops})
    eligible["proportion"] = eligible["population"]
    chosen = eligible[eligible["cluster_key"].isin(["a0", "a4", "b1", "b2", "b3"])].copy()
    chosen["Name"] = chosen["cluster_key"]
    chosen["population"] = chosen["population"] * stage2_population_scale
    data = pd.DataFrame({"cluster_key": [k for k in chosen["cluster_key"] for _ in range(5)]})
    data["household_id"] = range(len(data))
    data["eligible_adults_in_hh"] = 1
    out, _ = calculate_three_stage_weights(eligible, chosen, data, avg_household_size=5, use_stratum_weight=False)
    return eligible, chosen, out


def test_stage_check_passes_when_frame_population_cancels():
    from src.funcs import check_stage_weights
    eligible, chosen, out = _pps_example()
    c = check_stage_weights(eligible, chosen, out, target_clusters={"A": 2, "B": 3})
    assert c["size_measure_ratio"].tolist() == pytest.approx([1.0] * 5)
    assert (c["ratio"] >= 1 - 1e-9).all()                       # only the 1/(1-(1-p)^k) form can push it up
    assert c["ratio"].to_numpy() == pytest.approx(c["nonlinear"].to_numpy())


def test_stage_check_flags_mismatched_size_measure():
    from src.funcs import check_stage_weights
    eligible, chosen, out = _pps_example(stage2_population_scale=3.0)
    c = check_stage_weights(eligible, chosen, out)
    assert (c["ratio"] > 2.5).all()


def test_poststratify_matches_controls_and_keeps_within_cell_ratios():
    from src.funcs import poststratify
    _, _, out = _pps_example()
    w = poststratify(out, {"A": 300, "B": 100}, weight_col="combined_weight")
    assert w[out["province"] == "A"].sum() / w.sum() == pytest.approx(0.75)
    assert w.sum() == pytest.approx(len(out))
    a = out["province"] == "A"
    factor = w[a] / out.loc[a, "combined_weight"]
    assert np.allclose(factor, factor.iloc[0])                  # one factor per cell


def test_certainty_cluster_stops_overstating_the_stratum_total():
    """One cluster holds half of stratum A; with 2 draws it is a certainty unit (k*p = 1)."""
    from src.funcs import calculate_three_stage_weights, check_stage_weights
    eligible = pd.DataFrame({"cluster_key": ["big"] + [f"a{i}" for i in range(10)],
                             "province": "A", "population": [1000] + [100] * 10})
    eligible["proportion"] = eligible["population"]
    chosen = eligible[eligible["cluster_key"].isin(["big", "a0"])].assign(Name=lambda d: d["cluster_key"])
    data = pd.DataFrame({"cluster_key": ["big"] * 5 + ["a0"] * 5, "household_id": range(10),
                         "eligible_adults_in_hh": 1})

    def ht_total(certainty_units):
        frame = chosen.copy()                                # calculate_three_stage_weights adds columns to it
        out, _ = calculate_three_stage_weights(eligible, frame, data, avg_household_size=5,
                                               use_stratum_weight=False, certainty_units=certainty_units)
        return (out["combined_weight"] * 5).sum(), frame, out    # households x hh size = people

    old, _, _ = ht_total(False)
    new, frame, out = ht_total(True)
    assert old > 2000 * 1.15                                 # old form overstates the 2000 people
    assert new == pytest.approx(2000)
    assert frame["certainty_unit"].tolist() == [True, False]
    c = check_stage_weights(eligible, frame, out)
    assert c["ratio"].to_numpy() == pytest.approx([1.0, 1.0])


def test_poststratify_requires_every_cell_controlled():
    from src.funcs import poststratify
    _, _, out = _pps_example()
    with pytest.raises(ValueError, match="No control total"):
        poststratify(out, {"A": 300}, weight_col="combined_weight")
