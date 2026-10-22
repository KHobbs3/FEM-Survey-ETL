"""Tests for src/stations.py on synthetic answers only (no participant data).

Run from table_analysis/:  python -m pytest tests/
"""
import numpy as np
import pandas as pd

from src.stations import NO_LISTENING, StationStandardiser, impute_frequencies

STD = StationStandardiser(extra_names=["Kara", "Sama", "Zéphyr"])


def names(answer):
    return STD.standardise(answer)[0]


def test_every_station_in_an_answer_is_separate_with_its_own_frequency():
    assert names("Radio rfi 89.0 Radio Lomé 99.5") == [("RFI", "89.0"), ("Radio Lomé", "99.5")]
    assert names("Radio Maria , Radio Lomé") == [("Radio Maria", None), ("Radio Lomé", None)]
    assert names("Radio tchaoudjo 100.9 et radio venus 103.9") == [("Radio Tchaoudjo", "100.9"), ("Radio Vénus", "103.9")]
    assert names("RADIO VGK 92,1 FM") == [("Radio VGK", "92.1")]          # comma decimal
    assert names("99.2 kara") == [("Radio Kara", "99.2")]                 # frequency written first
    assert names("VGK 91.3 Radio Maria et RFI") == [("Radio VGK", "91.3"), ("Radio Maria", None), ("RFI", None)]


def test_formatting_variants_are_one_station():
    assert names("DeltaFM") == [("Delta FM", None)]
    assert names("Lomé") == [("Radio Lomé", None)]
    assert names("Fafa") == [("Radio Fafa", None)]
    assert names("RADIO FIRMAMENT 96.1") == [("Radio Firmament", "96.1")]
    assert names("Radio Dawol (88.7)") == [("Radio Dawol", "88.7")]


def test_typos_are_fuzzy_matched_and_unknowns_kept():
    entries, method = STD.standardise("GVK,radio lomé 98.8FM")
    assert entries == [("Radio VGK", None), ("Radio Lomé", "98.8")]
    entries, method = STD.standardise("Zephir FM")                       # close to a questionnaire choice
    assert entries == [("Zéphyr", None)] and method == "fuzzy"
    entries, method = STD.standardise("Radio Nouvelle Etoile 94.4")      # unknown: kept, flagged
    assert entries == [("Radio Nouvelle Etoile", "94.4")] and method == "typed"


def test_did_not_listen_and_empty():
    assert names("Hier je n'ai ecouté la") == [(NO_LISTENING, None)]
    assert names("N'a pas écouté radio") == [(NO_LISTENING, None)]
    assert names("0") == [] and names(None) == []


def test_reviewed_lookup_wins():
    std = StationStandardiser(lookup={"Radio X special": "Radio Lomé 99.5; RFI"})
    assert std.standardise("radio x  SPECIAL") == ([("Radio Lomé", "99.5"), ("RFI", None)], "lookup")


def test_missing_frequency_comes_from_nearest_respondent_with_the_same_station():
    m = pd.DataFrame({
        "name": ["Radio Lomé", "Radio Lomé", "Radio Lomé", "Radio Lomé", "RFI"],
        "freq": ["99.5", "98.1", None, None, None],
        "lat": [6.13, 9.55, 6.20, np.nan, 6.13],
        "lon": [1.22, 1.19, 1.25, np.nan, 1.22],
    })
    out = impute_frequencies(m)
    assert out["freq"].tolist()[:3] == ["99.5", "98.1", "99.5"]   # nearest donor is the Lomé-area one
    assert out["freq_imputed"].tolist() == [False, False, True, False, False]
    assert pd.isna(out.loc[3, "freq"]) and pd.isna(out.loc[4, "freq"])   # no GPS / no donor

    capped = impute_frequencies(m, max_km=5)                      # nearest donor is ~8.5 km away
    assert pd.isna(capped.loc[2, "freq"]) and not capped.loc[2, "freq_imputed"]
    assert capped.loc[2, "freq_donor_km"] > 5
