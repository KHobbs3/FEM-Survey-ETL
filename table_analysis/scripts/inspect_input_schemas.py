"""Print the SCHEMA of each input file -- never row values -- so it can be shared with
Claude (or anyone) without exposing participant data.

Run from table_analysis/:   python scripts/inspect_input_schemas.py

Output per file: whether it exists, row count, column names with non-null counts.
For the QA file it also reports (aggregates only) the validation-status counts and how
many of the reviewer's location notes match a surveyed cluster name.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import *  # noqa: E402,F403
from src.locations import match_to_clusters  # noqa: E402


def describe_columns(df):
    nn = df.notna().sum()
    for col in df.columns:
        if col == "geometry":
            print(f"    - geometry ({df.geometry.geom_type.value_counts().to_dict()})")
        else:
            print(f"    - {col!r}: {df[col].dtype}, non-null {nn[col]}/{len(df)}")


def section(label, path):
    print(f"\n=== {label}: {path}")
    if not Path(path).exists():
        print("    MISSING")
        return False
    return True


def read_csv(path, **kw):
    try:
        return pd.read_csv(path, low_memory=False, **kw)
    except UnicodeDecodeError:
        return pd.read_csv(path, low_memory=False, encoding="cp1252", **kw)


# Raw survey export: header + row count only
if section("Raw SurveyCTO export (DIR_RAW_DATA)", DIR_RAW_DATA):
    header = read_csv(DIR_RAW_DATA, nrows=0)
    n_rows = sum(1 for _ in open(DIR_RAW_DATA, "rb")) - 1
    print(f"    {n_rows} rows (approx., multi-line cells inflate this), {len(header.columns)} columns")

# XLSForm
if section("XLSForm (DIR_META_DATA)", DIR_META_DATA):
    xl = pd.ExcelFile(DIR_META_DATA)
    print(f"    sheets: {xl.sheet_names}")
    for sheet in ("survey", "choices"):
        if sheet in xl.sheet_names:
            print(f"    {sheet} columns: {list(pd.read_excel(xl, sheet_name=sheet, nrows=0).columns)}")

# Sampling files
import geopandas as gpd  # noqa: E402

from src.funcs import read_table  # noqa: E402

settlements = None
if section("Cluster polygons (DIR_CLUSTER_POLYGONS)", DIR_CLUSTER_POLYGONS):
    settlements = gpd.read_file(DIR_CLUSTER_POLYGONS)
    print(f"    {len(settlements)} rows, CRS {settlements.crs}")
    describe_columns(settlements)

for label, path in [("Surveyed clusters (DIR_SETTLEMENTS_CHOSEN)", DIR_SETTLEMENTS_CHOSEN),
                    ("Eligible frame (DIR_SETTLEMENTS_ELIGIBLE)", DIR_SETTLEMENTS_ELIGIBLE)]:
    if section(label, path):
        df = read_table(path)
        print(f"    {len(df)} rows")
        describe_columns(df)

# Cross-file cluster links are by CLUSTER_KEY (settlement name for Togo), compared as normalize_place
# does -- report overlaps (counts) and name ambiguity
from src.locations import normalize_place  # noqa: E402


def ids(df, col):
    return set(df[col].astype(str).str.strip().map(normalize_place)) if df is not None and col in df.columns else None


chosen = read_table(DIR_SETTLEMENTS_CHOSEN).rename(columns=SETTLEMENTS_CHOSEN_RENAME) if DIR_SETTLEMENTS_CHOSEN.exists() else None
eligible = read_table(DIR_SETTLEMENTS_ELIGIBLE).rename(columns=SETTLEMENTS_ELIGIBLE_RENAME) if DIR_SETTLEMENTS_ELIGIBLE.exists() else None
poly_col = CLUSTER_NAME_COL if CLUSTER_KEY == "Name" else CLUSTER_ID_COL
poly_ids, chosen_ids, frame_ids = ids(settlements, poly_col), ids(chosen, CLUSTER_KEY), ids(eligible, CLUSTER_KEY)
print(f"\n=== Cluster links ({CLUSTER_KEY})")
for label, a, b in [("polygons vs surveyed-clusters file", poly_ids, chosen_ids),
                    ("surveyed-clusters file vs eligible frame", chosen_ids, frame_ids)]:
    if a is None or b is None:
        print(f"    {label}: can't compare (file or {CLUSTER_KEY} column missing)")
    else:
        print(f"    {label}: {len(a & b)} shared, {len(a - b)} only in first, {len(b - a)} only in second")
if frame_ids is not None:
    excluded = {normalize_place(c) for c in EXCLUDED_CLUSTERS}
    print(f"    excluded clusters found in frame: {len(excluded & frame_ids)} / {len(excluded)}")
    frame_keys = eligible[CLUSTER_KEY].astype(str).str.strip().map(normalize_place)
    if chosen_ids:
        repeated = sorted(k for k in chosen_ids if (frame_keys == k).sum() > 1)
        print(f"    surveyed cluster names that repeat in the frame: {len(repeated)}")
if eligible is not None and "province" in eligible.columns:
    areas = eligible["province"].astype(str).str.strip()
    print(f"    frame areas: {areas.value_counts().to_dict()}; not in STRATA_MAP: {sorted(set(areas) - set(STRATA_MAP))}")
if settlements is not None and CLUSTER_NAME_COL in settlements.columns:
    names = settlements[CLUSTER_NAME_COL].dropna().astype(str)
    print(f"    polygon names shared by >1 cluster: {sorted(names[names.duplicated()].unique())}")

# QA review file
if section("QA review (DIR_QA)", DIR_QA):
    qa = read_csv(DIR_QA, dtype=str)
    print(f"    {len(qa)} rows")
    describe_columns(qa)
    for name, col in [("QA_KEY_COL", QA_KEY_COL), ("QA_STATUS_COL", QA_STATUS_COL),
                      ("QA_PROVINCE_COL", QA_PROVINCE_COL), ("QA_LOCATION_COL", QA_LOCATION_COL)]:
        print(f"    config {name} = {col!r}: {'found' if col in qa.columns else 'NOT FOUND'}")
    if QA_STATUS_COL in qa.columns:
        print(f"    {QA_STATUS_COL} counts: {qa[QA_STATUS_COL].value_counts(dropna=False).to_dict()}")
    if QA_KEY_COL in qa.columns:
        print(f"    duplicate submission keys: {qa[QA_KEY_COL].duplicated().sum()}")
    if (QA_LOCATION_COL in qa.columns and settlements is not None
            and {CLUSTER_ID_COL, CLUSTER_NAME_COL} <= set(settlements.columns)):
        notes = qa[QA_LOCATION_COL].dropna()
        matched = match_to_clusters(notes, settlements, CLUSTER_ID_COL, CLUSTER_NAME_COL).notna()
        print(f"    {QA_LOCATION_COL!r}: {len(notes)} notes, {notes.nunique()} distinct; "
              f"{matched.sum()} ({matched.mean():.0%}) match exactly one cluster name")

for label, path in [("Statement links (DIR_STATEMENT_LINKS)", DIR_STATEMENT_LINKS),
                    ("Manual location overrides (optional)", DIR_MANUAL_LOCATION_OVERRIDES)]:
    if section(label, path):
        describe_columns(read_csv(path, nrows=0))

print(f"\n=== Radio station polygons (DIR_STATION): {len(list(Path(DIR_STATION).parent.glob('*.gpkg')))} .gpkg files found")
