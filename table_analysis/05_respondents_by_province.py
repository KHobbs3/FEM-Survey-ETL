"""
Aggregate-only respondent counts by province.

Per CLAUDE.md: this script is meant to be run by the researcher, who reports
back the printed aggregate. Claude does not read/open/load the underlying
data file itself.
"""
import pandas as pd
from src.config import DIR_WEIGHTED_DATA

data = pd.read_csv(DIR_WEIGHTED_DATA)

counts = data["province"].value_counts().rename_axis("Province").reset_index(name="Number of respondents")
print(counts.to_string(index=False))
