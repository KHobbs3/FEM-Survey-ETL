"""Apply the reviewed "Autre" write-in recodes to the analysis data.

The recoding lives in pipeline_output/pipeline/other_recodes.py (questions and the recodes file
in pipeline_output/pipeline/config.py: OTHER_RECODE_QUESTIONS, OTHER_RECODES_PATH).
02z_recode_other.py writes the recoded answers into the data files themselves, so every later
step reads them; 03, 03b, 04 (both) and 06 also apply them when they load the data, which keeps
them right if the recodes file was edited after 02z last ran.

Applying again is harmless: each recoded question keeps its answers as given in
`_pre_recode__<question>`, and the recoding always starts from those.
"""
import sys
from pathlib import Path

_PIPELINE = Path(__file__).resolve().parents[2] / "pipeline_output"


def apply_writein_recodes(data):
    """data with each "Autre" answer replaced by its reviewed recode, where the write-in text is in
    the recodes file. No recodes file -> data unchanged (and a message says so)."""
    if str(_PIPELINE) not in sys.path:
        sys.path.insert(0, str(_PIPELINE))
    from pipeline.other_recodes import apply_other_recodes
    return apply_other_recodes(data)
