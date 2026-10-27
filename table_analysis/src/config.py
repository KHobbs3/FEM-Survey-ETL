from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------------
# Country -- this is the one thing that should need to change to run this
# pipeline on a new country's data, along with the file paths below that
# reference country-specific source files.
# ---------------------------------------------------------------------------
COUNTRY = "Benin"

# ---------------------------------------------------------------------------
# dir paths (all directory/path variables are prefixed DIR_ for consistency)
# ---------------------------------------------------------------------------
DIR_RAW=Path("data/1_raw")
DIR_CLEANED=Path("data/2_cleaned")
DIR_META=Path("../data/")
DIR_FINAL=Path("data/3_final")
DIR_SAMPLE = Path("../../0_sampling/output/")
DIR_VALIDATE = Path("data/4_validation_checks/")
DIR_STATION = "/Users/kt/Documents/work/AIM Charities/FEM/2. Remote Desktop Notes/2. Family Planning/2. Everything Radio/Radio Reach/cloudrf/output/gpkg/benin/fem_benin_buildings/*"

#file paths
DIR_RAW_DATA=DIR_RAW/"concatenated_data.csv"
DIR_META_DATA=DIR_META/"[Fon]+Benin+Questionnaire+d'Enquete.xlsx"
DIR_MAPPED_DATA=DIR_CLEANED/ "fem_survey_benin_mapped.csv"
DIR_CLEANED_DATA=DIR_CLEANED/ "fem_survey_benin_cleaned.csv"
DIR_WEIGHTED_DATA=DIR_CLEANED/ "02_fem_survey_benin_weighted.csv"
DIR_CHOICES=DIR_META/"[Fon]+Benin+Questionnaire+d'Enquete.xlsx"
DIR_STATEMENT_LINKS=DIR_META/"driver_barrier_statement_links.csv"
# Reviewed lookup for station_*_other answers (src/stations.py): column 1 = exact answer,
# column 2 = "; "-separated cleaned names ("Radio Tokpa 96.7; RFI"). Optional until created.
DIR_STATIONS_OTHER_CLEANED=DIR_VALIDATE/"stations_other_cleaned_reviewed.csv"
# A station mention without a frequency takes the nearest respondent's (same station) within this distance
STATIONS_MAX_KM = 10

# Output table paths -- {date} is filled in by the notebook (today's run date)
# at export time, e.g. str(DIR_DRIVERS_BARRIERS_TABLE).format(date=date_string)
DIR_DRIVERS_BARRIERS_TABLE=DIR_FINAL/f"{COUNTRY}_drivers-barriers_table_{{date}}.csv"
DIR_DRIVERS_BARRIERS_TABLE_GROUPED=DIR_FINAL/f"{COUNTRY}_drivers-barriers_table_grpd_nonusers_{{date}}.csv"
DIR_SUPPLY_QUESTIONS_TABLE=DIR_FINAL/f"{COUNTRY}_supply_question_table_{{date}}.xlsx"

# ---------------------------------------------------------------------------
# Survey-response label mappings (country/language-specific wording ->
# analysis-ready values). Update these when running a new country's data.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Weighting (02b_calculate_weights_KH.ipynb -> funcs.calculate_three_stage_weights)
# ---------------------------------------------------------------------------
# Stage 1 = PPS cluster weight x stratum weight (N_h/N)/(k_h/K) when True; cluster weight
# only when False. The cluster weight already corrects for allocation, so True applies
# that correction twice. False from 2026-10-26, aligning Benin with Niger and Togo.
USE_STRATUM_WEIGHT = False
# Divide the weight by (random_draw_attempts / eligible_adults_in_hh), with the Borgou
# day-1 and Plateau language rules. Off from 2026-10-26 (aligned with Niger and Togo):
# it multiplied the weight by household size a second time on top of stage 3, and
# households that didn't take part are already covered by stage 2.
NONRESPONSE_ADJUSTMENT = False
# Eligible-frame column used as the stage 1 PPS size measure (p_i = size_i / stratum total).
# "population" from 2026-10-26: the gpkg's "proportion" didn't match the population in
# Atlantique, Couffo and Oueme (02b stage check), and population is what stage 2 and N_h use.
PPS_SIZE_MEASURE = "population"
# Clusters with k_h * p_i >= 1 get inclusion probability 1 (funcs.pps_inclusion_probabilities).
# On from 2026-10-26 after reviewing check_stage_weights: Sékou and Ouidah (Atlantique) and Doga and
# Agonlin-Houégbo (Zou) have k_h * p_i >= 1; without this the other clusters there were overweighted
# (Horvitz-Thompson population 1.42x / 1.34x the frame).
PPS_CERTAINTY_UNITS = True
# Post-stratify the final weights (normalized_weight and combined_weight_adjusted, the
# pipeline's WEIGHT_COL) to frame population by this column, or None to skip. The
# design-only weights are kept as normalized_weight_design / combined_weight_design.
POSTSTRATIFY_ON = "province"
# Clusters allocated per stratum in sampling/config.R, before the x2 for replacements.
# Reference only (check_stage_weights prints it next to k_h). TODO: fill in from config.R.
CLUSTERS_ALLOCATED = {}

STATEMENT_LIKERT_MAP = {
    "Je suis d'accord/Un yí gbè": 1,
    'Je suis en désaccord/Un yí gbè na ǎ .': 0,
    "Je ne sais pas/Un tuùn ǎ": np.nan,
    'Je préfère ne pas répondre/Un jló ná yí gbe ǎ .': np.nan,
}


# Quality checks (03b_check_drivers_barriers_quality.py -> src/quality.py)
# Flag rates are also shown by this column. "province" is in the weighted file; 03's "region"
# grouping is only added inside that notebook.
GEO_SPLIT_COL = "province"
# SurveyCTO's submission key, used to identify flagged interviews in the export
QA_KEY_COL = "KEY"


def link_key(label):
    """Comparable form of a bilingual "French/Fon" label: the French part, lower case, straight
    apostrophes, spaces collapsed. The French/Fon separator is the last "/" with no space before
    it, so "Ma femme / je suis ménopausée/ ..." keeps "ma femme / je suis ménopausée". Survives an
    Excel plain-CSV save, which mangles only the non-French letters. None for a missing label."""
    import unicodedata
    if label is None or (isinstance(label, float) and np.isnan(label)):
        return None
    s = unicodedata.normalize("NFC", str(label)).replace("’", "'").replace("‘", "'")
    cuts = [i for i, ch in enumerate(s) if ch == "/" and i > 0 and not s[i - 1].isspace()]
    if cuts:
        s = s[:cuts[-1]]
    return " ".join(s.lower().split()).strip(" .")


def load_statement_links(path=None):
    """driver_barrier_statement_links.csv (list_name, label, linked_statements) plus 'key' =
    link_key(label). Reads UTF-8, or Excel's plain-CSV encodings (Mac Roman / Windows-1252)."""
    import pandas as pd
    raw = Path(path or DIR_STATEMENT_LINKS).read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        # Pick the decoding that gives the most French accented letters
        options = [raw.decode(enc, errors="replace") for enc in ("mac_roman", "cp1252")]
        text = max(options, key=lambda t: sum(t.count(c) for c in "éèàçêôûîù"))
    import io
    links = pd.read_csv(io.StringIO(text), dtype=str)
    links["key"] = links["label"].map(link_key)
    return links


#function
def find_col(data,search_str):
    relevant_cols=[col for col in data.columns.tolist() if search_str in col]
    return data.loc[:,relevant_cols]
