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
DIR_STATION = "/Users/kt/Documents/work/AIM Charities/FEM/Family Planning/2. Everything Radio/Radio Reach/cloudrf/output/gpkg/benin/fem_benin_buildings/*"

#file paths
DIR_RAW_DATA=DIR_RAW/"concatenated_data.csv"
DIR_META_DATA=DIR_META/"[Fon]+Benin+Questionnaire+d'Enquete.xlsx"
DIR_MAPPED_DATA=DIR_CLEANED/ "fem_survey_benin_mapped.csv"
DIR_CLEANED_DATA=DIR_CLEANED/ "fem_survey_benin_cleaned.csv"
DIR_WEIGHTED_DATA=DIR_CLEANED/ "02_fem_survey_benin_weighted.csv"
DIR_CHOICES=DIR_META/"[Fon]+Benin+Questionnaire+d'Enquete.xlsx"
DIR_STATEMENT_LINKS=DIR_META/"driver_barrier_statement_links.csv"

# Output table paths -- {date} is filled in by the notebook (today's run date)
# at export time, e.g. str(DIR_DRIVERS_BARRIERS_TABLE).format(date=date_string)
DIR_DRIVERS_BARRIERS_TABLE=DIR_FINAL/f"{COUNTRY}_drivers-barriers_table_{{date}}.csv"
DIR_DRIVERS_BARRIERS_TABLE_GROUPED=DIR_FINAL/f"{COUNTRY}_drivers-barriers_table_grpd_nonusers_{{date}}.csv"
DIR_SUPPLY_QUESTIONS_TABLE=DIR_FINAL/f"{COUNTRY}_supply_question_table_{{date}}.xlsx"

# ---------------------------------------------------------------------------
# Survey-response label mappings (country/language-specific wording ->
# analysis-ready values). Update these when running a new country's data.
# ---------------------------------------------------------------------------
STATEMENT_LIKERT_MAP = {
    "Je suis d'accord/Un yí gbè": 1,
    'Je suis en désaccord/Un yí gbè na ǎ .': 0,
    "Je ne sais pas/Un tuùn ǎ": np.nan,
    'Je préfère ne pas répondre/Un jló ná yí gbe ǎ .': np.nan,
}


#function
def find_col(data,search_str):
    relevant_cols=[col for col in data.columns.tolist() if search_str in col]
    return data.loc[:,relevant_cols]
