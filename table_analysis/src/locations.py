"""Assign every respondent to a surveyed cluster, combining GPS with the manual QA review.

Used by 02a_correct_locations.ipynb. Kept here (not in the notebook) so it can be
tested against synthetic fixtures in tests/ without touching participant data.

Clusters are identified by their ID (geom_id) throughout; names are only used to read the
QA reviewers' free-text notes and to label the output. A QA note that matches more than one
cluster name is treated as unmatched rather than guessed.

Priority, highest first:
  1. override  -- researcher-maintained manual_location_overrides.csv (KEY -> geom_id or name)
  2. gps       -- respondent's GPS point falls inside a cluster polygon
  3. qa        -- QA reviewer's manually noted location matches exactly one cluster name
  4. gps_snap  -- GPS point is within `snap_distance_m` of a cluster polygon
  5. label     -- the location name recorded in the survey matches exactly one cluster name
  6. unresolved -- none of the above; exported for manual review, dropped downstream

With qa_overrides_gps=True, (3) is moved above (2), i.e. the QA note wins whenever the
reviewer and the GPS disagree. Benin trusted GPS on disagreement, hence the default.
"""
import re
import unicodedata

import geopandas as gpd
import pandas as pd


def normalize_place(name):
    """Comparable form of a settlement name: accents stripped, lower case, punctuation
    collapsed. Also absorbs the mojibake seen in the Benin QA export ("TohouË" vs
    "Tohouè") since both collapse to the same unaccented letter."""
    if pd.isna(name):
        return pd.NA
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^0-9a-z]+", " ", s.lower()).strip()
    return s or pd.NA


def name_lookup(clusters, id_col, name_col, aliases=None):
    """{normalised name: cluster id} for names that identify exactly one cluster.
    Ambiguous names are left out. `aliases` maps extra spellings to a cluster name or id."""
    norm = clusters[name_col].map(normalize_place)
    counts = norm.value_counts()
    lookup = {n: i for n, i in zip(norm, clusters[id_col]) if pd.notna(n) and counts[n] == 1}
    ids = set(clusters[id_col])
    for raw, target in (aliases or {}).items():
        lookup[normalize_place(raw)] = target if target in ids else lookup.get(normalize_place(target), pd.NA)
    return lookup


def match_to_clusters(values, clusters, id_col, name_col, aliases=None):
    """Map free-text place names (or cluster ids) onto cluster ids; NaN where there is
    no unique match. Returns a Series aligned with `values`."""
    ids = set(clusters[id_col])
    lookup = name_lookup(clusters, id_col, name_col, aliases)
    return pd.Series(values).map(lambda v: v if v in ids else lookup.get(normalize_place(v), pd.NA))


def resolve_cluster_assignments(data, settlements, qa_locations=None, overrides=None, label_col=None,
                                aliases=None, snap_distance_m=1000, utm_crs="epsg:32631",
                                qa_overrides_gps=False, key_col="KEY", id_col="geom_id",
                                name_col="Name", lat_col="gps-Latitude", lon_col="gps-Longitude"):
    """Return a DataFrame indexed like `data` with one row per respondent:

    cluster_gps, cluster_qa, cluster_override, cluster_snap, cluster_label (all cluster ids),
    snap_distance_m, cluster_id (final), cluster_name, location_source,
    qa_location_unmatched / label_unmatched (bool), n_gps_clusters.

    qa_locations: Series indexed by submission KEY -> reviewer's noted location text.
    label_col:    column of `data` holding the location name recorded in the survey.
    overrides:    Series indexed by submission KEY -> cluster id or cluster name.
    """
    clusters = settlements[[id_col, name_col, "geometry"]].copy()
    clusters[id_col] = clusters[id_col].astype(str).str.strip()
    pts = gpd.GeoDataFrame(
        data[[key_col]].copy(),
        geometry=gpd.points_from_xy(data[lon_col], data[lat_col]),
        crs="epsg:4326",
    ).to_crs(settlements.crs)
    has_gps = data[lat_col].notna() & data[lon_col].notna()

    out = pd.DataFrame(index=data.index)

    # 2. Inside a polygon. Overlapping polygons can match a point twice -- keep the
    # first and record how many there were so the caller can report it.
    joined = pts.loc[has_gps].sjoin(clusters[[id_col, "geometry"]], predicate="intersects")
    out["n_gps_clusters"] = joined.groupby(level=0).size().reindex(data.index, fill_value=0)
    out["cluster_gps"] = joined.groupby(level=0)[id_col].first().reindex(data.index)

    # 3. QA reviewer's note (free text -> unique cluster name -> id)
    keys = data[key_col]
    if qa_locations is not None:
        qa_raw = keys.map(qa_locations)
        out["cluster_qa"] = match_to_clusters(qa_raw, clusters, id_col, name_col, aliases).set_axis(data.index)
        out["qa_location_unmatched"] = qa_raw.notna() & out["cluster_qa"].isna()
    else:
        out["cluster_qa"] = pd.NA
        out["qa_location_unmatched"] = False

    # 5. Location name recorded in the survey (last resort)
    if label_col is not None:
        labels = data[label_col]
        out["cluster_label"] = match_to_clusters(labels, clusters, id_col, name_col, aliases).set_axis(data.index)
        out["label_unmatched"] = labels.notna() & out["cluster_label"].isna()
    else:
        out["cluster_label"] = pd.NA
        out["label_unmatched"] = False

    # 1. Researcher overrides (id or unique name)
    if overrides is not None:
        out["cluster_override"] = match_to_clusters(keys.map(overrides), clusters, id_col, name_col).set_axis(data.index)
    else:
        out["cluster_override"] = pd.NA

    # 4. Snap: nearest polygon within snap_distance_m, for points outside every polygon
    out["cluster_snap"] = pd.NA
    out["snap_distance_m"] = pd.NA
    outside = has_gps & out["cluster_gps"].isna()
    if outside.any():
        pts_m = pts.loc[outside].to_crs(utm_crs)
        clusters_m = clusters[[id_col, "geometry"]].to_crs(utm_crs)
        nearest = gpd.sjoin_nearest(pts_m, clusters_m, how="left", distance_col="dist_m")
        nearest = nearest[~nearest.index.duplicated(keep="first")]
        within = nearest["dist_m"] <= snap_distance_m
        out.loc[nearest.index, "snap_distance_m"] = nearest["dist_m"]
        out.loc[nearest.index[within], "cluster_snap"] = nearest.loc[within, id_col]

    order = (["cluster_override", "cluster_qa", "cluster_gps", "cluster_snap", "cluster_label"] if qa_overrides_gps
             else ["cluster_override", "cluster_gps", "cluster_qa", "cluster_snap", "cluster_label"])
    out["cluster_id"] = pd.NA
    out["location_source"] = "unresolved"
    for col in reversed(order):  # lowest priority first, so higher ones overwrite
        hit = out[col].notna()
        out.loc[hit, "cluster_id"] = out.loc[hit, col]
        out.loc[hit, "location_source"] = col.replace("cluster_", "").replace("snap", "gps_snap")
    out["cluster_name"] = out["cluster_id"].map(clusters.drop_duplicates(id_col).set_index(id_col)[name_col])
    return out


# ============================================================================
# Togo: sampled clusters, linked through the field schedule and the manual review's names
# ============================================================================

def parse_point(value):
    """'6.1234, 1.2345' (lat, lon) -> (lat, lon); None if there aren't two decimal numbers."""
    nums = re.findall(r"-?\d+\.\d+", str(value))
    return (float(nums[0]), float(nums[1])) if len(nums) >= 2 else None


def villages(name):
    """Normalised villages of a grouped cluster name: "Kévé; Assahoun" -> ["keve", "assahoun"]."""
    return [v for v in (normalize_place(p) for p in str(name).split(";")) if isinstance(v, str)]


def link_points(points, clusters, name_col="Name", utm_crs="epsg:32631"):
    """For each point (GeoSeries, may hold missing/empty points): the cluster it lies inside (first if
    several), the nearest cluster and the distance to it in metres (0 when inside)."""
    out = pd.DataFrame({"inside": pd.Series(pd.NA, index=points.index, dtype=object),
                        "nearest": pd.Series(pd.NA, index=points.index, dtype=object),
                        "dist_m": pd.Series(float("nan"), index=points.index)})
    ok = points.map(lambda g: g is not None and not g.is_empty)
    if not ok.any():
        return out
    pts = gpd.GeoDataFrame(geometry=points[ok]).to_crs(utm_crs)
    cl = clusters[[name_col, "geometry"]].to_crs(utm_crs)
    inside = pts.sjoin(cl, predicate="intersects").groupby(level=0)[name_col].first()
    near = gpd.sjoin_nearest(pts, cl, distance_col="dist_m")
    near = near[~near.index.duplicated()]
    out.loc[inside.index, "inside"] = inside
    out.loc[near.index, "nearest"] = near[name_col]
    out.loc[near.index, "dist_m"] = near["dist_m"]
    return out


def link_schedule(schedule, sample, name_col="Settlement", region_col="Region", point_col="GPS coordinate",
                  snap_m=1000, utm_crs="epsg:32631"):
    """Each selected settlement in the field schedule -> the sampled cluster its GPS coordinate falls in
    (or the nearest within snap_m); without a usable coordinate, the sampled cluster whose name or
    one of whose villages matches the settlement name. One row per settlement and region."""
    s = schedule.dropna(subset=[name_col]).copy()
    s[name_col] = s[name_col].astype(str).str.strip()
    s["_region"] = s[region_col].fillna("").astype(str).str.strip() if region_col in s.columns else ""
    parsed = s[point_col].map(parse_point) if point_col in s.columns else pd.Series(None, index=s.index)
    pts = gpd.GeoSeries(gpd.points_from_xy([p[1] if p else None for p in parsed],
                                           [p[0] if p else None for p in parsed]), index=s.index, crs="epsg:4326")
    links = link_points(pts, sample, "Name", utm_crs)
    inside = links["inside"].notna()
    near = ~inside & (links["dist_m"] <= snap_m)
    s["sample_cluster"] = links["inside"].where(inside, links["nearest"].where(near))
    s["how"] = pd.Series(pd.NA, index=s.index, dtype=object)
    s.loc[inside, "how"] = "gps inside"
    s.loc[near, "how"] = f"gps within {snap_m} m"
    s["dist_m"] = links["dist_m"]
    # No usable coordinate: the settlement name is a sampled cluster's name or one of its villages
    by_name = {}
    for n in sample["Name"]:
        for k in {normalize_place(n), *villages(n)}:
            by_name.setdefault(k, set()).add(n)
    for i in s.index[s["sample_cluster"].isna()]:
        hits = by_name.get(normalize_place(s.at[i, name_col]), set())
        if len(hits) == 1:
            s.at[i, "sample_cluster"], s.at[i, "how"] = next(iter(hits)), "name"
    # One row per settlement + region (the schedule repeats settlements across visit days)
    s = s.sort_values("how", na_position="last")
    conflicts = s.dropna(subset=["sample_cluster"]).groupby([name_col, "_region"])["sample_cluster"].nunique()
    s = s.drop_duplicates([name_col, "_region"], keep="first")
    s["conflicting_days"] = s.set_index([name_col, "_region"]).index.map(conflicts.gt(1)).fillna(False)
    meta = sample.drop(columns="geometry").drop_duplicates("Name").set_index("Name")
    s["sample_province"] = s["sample_cluster"].map(meta["province"]) if "province" in meta else pd.NA
    return s.rename(columns={name_col: "settlement", "_region": "region"})[
        ["settlement", "region", "sample_cluster", "sample_province", "how", "dist_m", "conflicting_days"]
    ].reset_index(drop=True)


def assign_sampled_clusters(data, sample, schedule_links, overrides=None, key_col="KEY", location_col="location",
                            province_col="province", lat_col="gps-Latitude", lon_col="gps-Longitude",
                            snap_m=1000, disagree_m=1000, utm_crs="epsg:32631", spatial_first=True):
    """Assign each respondent to a sampled cluster.

    spatial_first=True (default) -- the spatial join decides first:
      1. override  -- manual_location_overrides.csv (KEY -> sampled cluster name)
      2. gps       -- the GPS point lies inside a sampled cluster polygon
      3. name      -- otherwise the respondent's location (manual review, from 01) is a schedule settlement
                      (-> its sampled cluster), or a sampled cluster's name / one of its villages
      4. gps_snap  -- otherwise the nearest sampled cluster, if within snap_m
      5. unresolved
    spatial_first=False -- the manual review's name decides first (gps wins only on a clear disagreement:
    inside a different cluster and more than disagree_m from the named one), then gps, then gps_snap.

    Returns a DataFrame indexed like `data`: sample_cluster, location_source, cluster_by_name,
    cluster_by_gps, name_disagrees (GPS inside a different cluster than the name points to),
    gps_dist_m (to the nearest sampled cluster), named_dist_m (to the named one).
    """
    out = pd.DataFrame(index=data.index)
    # Name lookup: schedule settlement (+ region) first, then sampled names and their villages
    by_name_region, by_name = {}, {}
    for _, r in schedule_links.dropna(subset=["sample_cluster"]).iterrows():
        n, reg = normalize_place(r["settlement"]), normalize_place(r["region"])
        by_name_region[(n, reg)] = r["sample_cluster"]
        by_name.setdefault(n, set()).add(r["sample_cluster"])
    schedule_names = set(by_name)
    for c in sample["Name"]:
        for k in {normalize_place(c), *villages(c)}:
            if k not in schedule_names:   # the schedule's link wins for its own settlement names
                by_name.setdefault(k, set()).add(c)

    def by_location_name(loc, prov):
        n = normalize_place(loc)
        if not isinstance(n, str):
            return pd.NA
        hit = by_name_region.get((n, normalize_place(prov)))
        if hit is not None:
            return hit
        hits = by_name.get(n, set())
        return next(iter(hits)) if len(hits) == 1 else pd.NA

    out["cluster_by_name"] = [by_location_name(l, p) for l, p in zip(data[location_col], data[province_col])]

    pts = gpd.GeoSeries(gpd.points_from_xy(data[lon_col], data[lat_col]), index=data.index, crs="epsg:4326")
    links = link_points(pts, sample, "Name", utm_crs)
    out["cluster_by_gps"] = links["inside"]
    out["gps_dist_m"] = links["dist_m"]

    # Distance from each respondent to the cluster their name points to
    geoms = sample.drop_duplicates("Name").set_index("Name").geometry.to_crs(utm_crs)
    named = out["cluster_by_name"].notna() & pts.map(lambda g: g is not None and not g.is_empty)
    out["named_dist_m"] = float("nan")
    if named.any():
        p_m = pts[named].to_crs(utm_crs)
        out.loc[named, "named_dist_m"] = [pt.distance(geoms[c]) for pt, c in zip(p_m, out.loc[named, "cluster_by_name"])]

    out["name_disagrees"] = (out["cluster_by_name"].notna() & out["cluster_by_gps"].notna()
                             & (out["cluster_by_gps"] != out["cluster_by_name"]))
    snapped = links["nearest"].where(links["dist_m"] <= snap_m)
    snap_step = (snapped.notna() & links["inside"].isna(), snapped, "gps_snap")
    gps_step = (links["inside"].notna(), links["inside"], "gps")
    name_step = (out["cluster_by_name"].notna(), out["cluster_by_name"], "name")

    out["sample_cluster"], out["location_source"] = pd.NA, "unresolved"
    if spatial_first:   # lowest priority first, so higher ones overwrite
        steps = [snap_step, name_step, gps_step]
    else:
        clear = out["name_disagrees"] & (out["named_dist_m"] > disagree_m)
        steps = [snap_step, gps_step, name_step, (clear, out["cluster_by_gps"], "gps (name disagrees)")]
    if overrides is not None:
        valid = set(sample["Name"])
        ov = data[key_col].map(overrides).where(lambda s: s.isin(valid))
        steps.append((ov.notna(), ov, "override"))
    for mask, values, label in steps:
        out.loc[mask, "sample_cluster"] = values[mask]
        out.loc[mask, "location_source"] = label
    return out
