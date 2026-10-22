from pathlib import Path
import re
import unicodedata
import numpy as np

# ---------------------------------------------------------------------------
# Country -- this is the one thing that should need to change to run this
# pipeline on a new country's data, along with the file paths below that
# reference country-specific source files.
# ---------------------------------------------------------------------------
COUNTRY = "Togo"

# ---------------------------------------------------------------------------
# Dir paths (all directory/path variables are prefixed DIR_ for consistency)
# ---------------------------------------------------------------------------
DIR_RAW=Path("../raw_data")
DIR_CLEANED=Path("data/2_cleaned")
DIR_META=Path("data/meta")
DIR_FINAL=Path("data/3_final")
DIR_VALIDATE = Path("data/4_validation_checks/")
DIR_STATION = "/Users/kt/Documents/work/AIM Charities/FEM/2. Remote Desktop Notes/2. Family Planning/2. Everything Radio/Radio Reach/cloudrf/output/gpkg/togo/fem_togo_buildings_rollout/*"

# Metric CRS for distance/buffer work.
UTM_CRS = "epsg:32631"

# File paths
DIR_RAW_DATA=DIR_RAW/"[Ewe_Fr] Questionnaire de Sondage_WIDE.xlsx"
DIR_META_DATA=DIR_META/"[Ewe_Fr]+Questionnaire+de+Sondage.xlsx"   # XLSForm: 'survey' + 'choices' sheets
DIR_CHOICES=DIR_META_DATA
DIR_QA=DIR_META/"Togo QA Quantitative survey - Full Data with QA.csv"

DIR_MAPPED_DATA=DIR_CLEANED/ "fem_survey_togo_mapped.csv"
DIR_LOCATED_DATA=DIR_CLEANED/ "fem_survey_togo_located.csv"   # 02a output: mapped + corrected cluster
DIR_CLEANED_DATA=DIR_CLEANED/ "fem_survey_togo_cleaned.csv"
DIR_WEIGHTED_DATA=DIR_CLEANED/ "02_fem_survey_togo_weighted.csv"
# Country-specific (statement names and choice wording can differ), so each country has its own file,
# driver_barrier_statement_links_<country>.csv. Read it only through this constant, with
# load_statement_links() below. Rows are linked to answers on the French part of the label (link_key),
# so the local-language half doesn't have to match -- or even survive an Excel save.
DIR_STATEMENT_LINKS=DIR_META/ f"driver_barrier_statement_links_{COUNTRY.lower()}.csv"

# Letters only found in the local-language half of a label (Ewe/Fon), plus "_", which Excel's plain
# "CSV" save puts in place of the letters it can't store.
_LOCAL_LANGUAGE_MARKERS = re.compile(r"[ɐ-ʯŋŊƒƑƆƐƉƔƲ_]")


def link_key(label):
    """French part of a "French/local language" label, normalised for matching (case, accents' Unicode
    form, apostrophe style, spacing, trailing punctuation). The French part ends at the last '/' before the
    first local-language marker -- it can contain '/' itself ("éduquer/former") -- else at the first '/'."""
    if not isinstance(label, str):
        return None
    m = _LOCAL_LANGUAGE_MARKERS.search(label)
    cut = label.rfind("/", 0, m.start()) if m else label.find("/")
    text = label[:cut] if cut > 0 else label
    text = unicodedata.normalize("NFC", text).replace("’", "'").replace("“", '"').replace("”", '"')
    return " ".join(text.split()).casefold().rstrip(" .:")


def load_statement_links(path=None):
    """Read the statement links CSV, adding `key` = link_key(label) -- match answers with
    `links.key == link_key(answer)`. Accepts UTF-8 or an Excel "CSV" save (Mac Roman / Windows-1252);
    only the French part is used, and those encodings keep French accents."""
    import io
    import pandas as pd
    from src.funcs import decode_csv
    path = Path(path or DIR_STATEMENT_LINKS)
    links = pd.read_csv(io.StringIO(decode_csv(path)))
    links["key"] = links["label"].map(link_key)
    return links
# Free-text "other" station answers -> cleaned station names ("; "-separated), used by 00_clean_data
DIR_STATIONS_OTHER_CLEANED=DIR_META/ "stations_other_cleaned.csv"

# ---------------------------------------------------------------------------
# Manual QA review file -- column names.
# ---------------------------------------------------------------------------
QA_KEY_COL = "Submission Key"
QA_STATUS_COL = "Validation status"
QA_VALIDATED = "Validated"
QA_PROVINCE_COL = "department name"
QA_LOCATION_COL = "location name"   # the reviewer's manually noted cluster/settlement
DIR_MANUAL_LOCATION_OVERRIDES=DIR_META/"manual_location_overrides.csv"

# ---------------------------------------------------------------------------
# Sampling files -> column names expected by 02b ({file's column: expected name}).
#   eligible frame:    geom_id, province (admin area), population, proportion (optional PPS measure)
#   surveyed clusters: geom_id (+ optional Name). Stratum and population are taken from the
#                      eligible frame by geom_id, so they can't disagree between files.
# ---------------------------------------------------------------------------
# For survey weights, we need to know which clusters were drawn and which were eligible.
# Cluster polygons. 02a no longer reads this file: it takes the surveyed clusters' polygons from the
# eligible frame via load_sampling_frames (matched by name as in 02b). Still used by the schema inspector.
DIR_CLUSTER_POLYGONS=DIR_META/"0_togo_surveybounds.geojson"
CLUSTER_ID_COL = "geom_id"
CLUSTER_NAME_COL = "name" 
DIR_SETTLEMENTS_ELIGIBLE=DIR_META/"0_togo_surveybounds.geojson"
DIR_SETTLEMENTS_SAMPLED=DIR_META/ "togo_sample_58.geojson"   # the 58 clusters drawn (polygons + names)
DIR_FIELD_SCHEDULE=DIR_META/"Togo QA Quantitative survey - Settlements chosen.csv"   # field visit schedule
SETTLEMENTS_SAMPLED_RENAME = {"name": "Name", "ADM1_FR": "province"}   # togo_sample_58.geojson, read by 02a
# Researcher-reviewed cluster names: 02a_manual_review_vs_sample.csv with `selected_name` edited.
# 02a reads it from data/meta/ (preferred: decisions live there), else from 4_validation_checks/.
DIR_LOCATION_REVIEW=DIR_META/"02a_manual_review_vs_sample_reviewed.csv"
# Surveyed clusters = the clusters respondents were finally assigned to (reviewed `selected_name`), written
# by 02a with one settlement-level point each. 02b links them to the eligible frame -- by that point first,
# then by name / village -- for population (X_sum), stratum and PPS proportion.
DIR_SETTLEMENTS_CHOSEN=DIR_CLEANED/"02a_clusters.csv"

# How clusters link across files.
CLUSTER_KEY = "Name"
RESPONDENT_CLUSTER_COL = "cluster_name"   # written by 02a: reviewed selected_name for each respondent
SETTLEMENTS_ELIGIBLE_RENAME = {"X_sum": "population", "ADM1_FR": "province", CLUSTER_NAME_COL: "Name"}
SETTLEMENTS_CHOSEN_RENAME = {}            # 02a_clusters.csv already has Name, province, point
SETTLEMENTS_CHOSEN_POINT_COL = "point"    # "lat, lon"
# Clusters removed from the eligible frame (by CLUSTER_KEY) before weighting, e.g. out of bounds
EXCLUDED_CLUSTERS = []


# Rebuild the frame exactly as sampling/3_random_sample.R saw it:
STRATA_MAP = {"Kara": "Kara", "Centrale": "Centrale", "Plateaux": "Plateaux", "Maritime": "Maritime"}
_N_DESIRED_TOTAL, _N_CLUSTERS_DRAWN, _PROP_QUAL, _P_AVAILABLE = 750, 58, 0.45, 0.20  # sampling/config.R
MIN_CLUSTER_POPULATION = (_N_DESIRED_TOTAL / _N_CLUSTERS_DRAWN) / (_PROP_QUAL * _P_AVAILABLE)  # ~143.7


# Sampling design parameters used by 02b weights.
AVG_HOUSEHOLD_SIZE = 4.4 # Togo MICS 2017: https://dhsprogram.com/pubs/pdf/MIS29/MIS29.pdf
# Stage-1 stratum weight ((N_h/N) / (k_h/K)). The cluster weight 1/(1-(1-p)^k_h) already accounts for each
# stratum's number of clusters, so multiplying by this as well corrects for allocation twice. False = cluster
# weight only (correct for stratified PPS). True = previous behaviour (still Benin's).
USE_STRATUM_WEIGHT = False
# Clusters allocated per stratum (sampling/config.R, before the x2 for replacements); checked against those surveyed.
CLUSTERS_ALLOCATED = {"Kara": 4, "Centrale": 3, "Plateaux": 7, "Maritime": 15}
# Treat clusters with k_h * p_i >= 1 as selected with certainty (one Maritime cluster): weight 1, and the
# other clusters' probabilities come from the rest of the stratum. False = previous behaviour.
PPS_CERTAINTY_UNITS = True
# Post-stratify the final weights to the frame population by this column (N_h / N^_h per cell, then
# normalised). The design weight is kept as `normalized_weight_design`. None = no post-stratification.
POSTSTRATIFY_ON = "province"
# Stage-2 household size by setting: each cluster takes the size for its respondents' majority `urban_rural`
# ("ménages ... plus élevée en milieu rural (4,8) qu'en milieu urbain (3,8)"). TODO(Togo): cite the source.
# Set to None to use AVG_HOUSEHOLD_SIZE for every cluster (previous behaviour).
HOUSEHOLD_SIZE_BY_SETTING = {"rural": 4.8, "urban": 3.8}
URBAN_RURAL_COL = "urban_rural"

# Geographic split for the driver/barrier tables (03). 'province' holds each respondent's
# sampling stratum (from the frame via the cluster, see STRATA_MAP), so splitting on it gives
# results by department.
GEO_SPLIT_COL = "province"
GEO_SPLIT_LABEL = "DEPARTMENT"

# Optional coarser grouping of strata into regions, for countries where departments are too
# many/small to report on and differ by language (e.g. Benin). Set to None if not used (e.g. Togo).
PROVINCE_TO_REGION = None

# Output table paths --
DIR_DRIVERS_BARRIERS_TABLE=DIR_FINAL/f"{COUNTRY}_drivers-barriers_table_{{date}}.csv"
DIR_DRIVERS_BARRIERS_TABLE_GROUPED=DIR_FINAL/f"{COUNTRY}_drivers-barriers_table_grpd_nonusers_{{date}}.csv"
DIR_SUPPLY_QUESTIONS_TABLE=DIR_FINAL/f"{COUNTRY}_supply_question_table_{{date}}.xlsx"

# ---------------------------------------------------------------------------
# Survey-response label mappings (country/language-specific wording ->
# analysis-ready values). Update these when running a new country's data. 
# This corresponds to the likert_agreement mapping in the survey choices file.
# ---------------------------------------------------------------------------
STATEMENT_LIKERT_MAP = {
    "Je suis en désaccord/ ": 1,
    "Je suis d'accord/Me lɔ̃ ɖe edzi.": 0,
    "Je ne sais pas/ Me nya o": np.nan,
    'Je préfère ne pas répondre/ Medi be maɖo ŋu o': np.nan,
}

def map_likert(frame, mapping=None):
    """Statement answers -> codes, matching on the French part before '/' (case, spacing, apostrophe style and
    trailing full stop ignored), since the Ewe part and punctuation differ between this map and the export.
    Raises listing any answer label it doesn't recognise (answer labels only, no respondent data)."""
    def key(label):
        return " ".join(str(label).split("/")[0].lower().replace("’", "'").split()).rstrip(" .")
    codes = {key(k): v for k, v in (mapping or STATEMENT_LIKERT_MAP).items()}
    labels = frame.stack().dropna().astype(str).unique()
    unknown = sorted(l for l in labels if l.strip() and key(l) not in codes)
    if unknown:
        raise ValueError(f"Statement answers not in STATEMENT_LIKERT_MAP: {unknown}")
    return frame.apply(lambda col: col.map(
        lambda v: codes[key(v)] if isinstance(v, str) and v.strip() else np.nan)).astype(float)

#function
def find_col(data,search_str):
    relevant_cols=[col for col in data.columns.tolist() if search_str in col]
    return data.loc[:,relevant_cols]
