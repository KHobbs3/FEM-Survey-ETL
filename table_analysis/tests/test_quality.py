"""Tests for src/quality.py on synthetic answers only (no participant data).

Run from table_analysis/:  python -m pytest tests/
"""
import numpy as np
import pandas as pd

from src import quality

LIKERT = {
    "Je suis en désaccord/ ": 1,
    "Je suis d'accord/Me lɔ̃ ɖe edzi.": 0,
    "Je ne sais pas/ Me nya o": np.nan,
    "Je préfère ne pas répondre/ Medi be maɖo ŋu o": np.nan,
}
AGREE, DISAGREE = "Je suis d’accord/Me lɔ̃ ɖe edzi", "Je suis en désaccord/ "   # curly apostrophe on purpose
DK = "Je ne sais pas/ Me nya o"


def statements(*rows):
    n = len(rows[0])
    return pd.DataFrame(list(rows), columns=[f"statement_{i}" for i in range(1, n + 1)])


def test_statement_columns_are_numeric_order_only():
    cols = ["statement_10", "statement_2", "statement_1", "statement_intro", "use"]
    assert quality.statement_columns(cols) == ["statement_1", "statement_2", "statement_10"]


def test_straightlining_and_near_straightlining():
    same = [AGREE] * 12
    near = [AGREE] * 11 + [DISAGREE]
    mixed = [AGREE, DISAGREE] * 6
    out = quality.statement_flags(statements(same, near, mixed), LIKERT)
    assert out["flag_statements_straightline"].tolist() == [True, False, False]
    assert out["flag_statements_near_straightline"].tolist() == [False, True, False]
    assert out["statements_longest_run"].tolist() == [12, 11, 1]


def test_straightlining_needs_enough_substantive_answers():
    few = [AGREE] * 5 + [DK] * 7
    out = quality.statement_flags(statements(few), LIKERT)
    assert not out["flag_statements_straightline"].iloc[0]
    assert out["flag_statements_mostly_nonsubstantive"].iloc[0]


def test_dont_know_missing_and_unknown_answers():
    all_dk = [DK] * 12
    missing = [None] * 12
    odd = [AGREE] * 11 + ["Peut-être/ ..."]
    out = quality.statement_flags(statements(all_dk, missing, odd), LIKERT)
    assert out["flag_statements_all_nonsubstantive"].tolist() == [True, False, False]
    assert out["flag_statements_none_answered"].tolist() == [False, True, False]
    assert out["flag_statements_unknown_answer"].tolist() == [False, False, True]


def drivers_frame():
    return pd.DataFrame({
        "reason_use_mc":    ["A|B", "A", None, "Je ne sais pas|A", "A|B|C|D", "A"],
        "drivers_nonusers": [None, "B", None, None, None, None],
        "reason_use_main":  ["A", "C", None, "A", "A", "Autres raisons non mentionnées ici, Veuillez préciser"],
        "reason_use_main_other": ["", "", "", "", "", " "],
    })


def test_driver_barrier_flags():
    out, combined = quality.driver_barrier_flags(drivers_frame(), "drivers", n_options=6)
    assert combined.tolist()[:2] == [["A", "B"], ["A"]]          # user version wins, as fillna in 03
    assert out["flag_drivers_both_versions_answered"].tolist() == [False, True, False, False, False, False]
    assert out["flag_drivers_none_recorded"].tolist() == [False, False, True, False, False, False]
    assert out["flag_drivers_main_not_selected"].tolist() == [False, True, False, False, False, True]
    assert out["flag_drivers_dont_know_with_reasons"].tolist() == [False, False, False, True, False, False]
    assert out["flag_drivers_many_selected"].tolist() == [False, False, False, False, True, False]
    assert out["flag_drivers_main_other_without_text"].tolist() == [False, False, False, False, False, True]


def test_every_flag_has_a_definition():
    out = quality.statement_flags(statements([AGREE] * 12), LIKERT)
    columns = list(out.columns)
    for category in quality.DRIVER_BARRIER_QUESTIONS:
        frame = drivers_frame().rename(columns=dict(zip(quality.DRIVER_BARRIER_QUESTIONS["drivers"]["lists"]
                                                         + ["reason_use_main", "reason_use_main_other"],
                                                         quality.DRIVER_BARRIER_QUESTIONS[category]["lists"]
                                                         + [quality.DRIVER_BARRIER_QUESTIONS[category]["main"],
                                                            quality.DRIVER_BARRIER_QUESTIONS[category]["other_text"]])))
        columns += list(quality.driver_barrier_flags(frame, category, n_options=6)[0].columns)
    flags = {c for c in columns if c.startswith("flag_")}
    assert flags <= set(quality.flag_definitions())


def test_missing_columns_are_skipped():
    out, _ = quality.driver_barrier_flags(pd.DataFrame({"reason_nonuse_mc": ["X"]}), "barriers")
    assert "flag_barriers_main_missing" not in out and "flag_barriers_many_selected" not in out
    assert not out["flag_barriers_none_recorded"].iloc[0]


def test_unknown_answers_counted_once_per_respondent():
    combined = pd.Series([["A", "Zz"], ["Zz", "Zz"], ["B "]])
    assert quality.unknown_answers(combined, ["A", "B"]) == {"Zz": 2}


# Choice labels as in the XLSForm, and the same labels after Excel's plain "CSV" save ("_" for lost letters)
FORM = [
    "Désir d'avoir des enfants : non spécifié/ Viwo ƒe lɔlɔ̃: Menyɔa nu si ta o.",
    "Ma femme / je suis ménopausée/ Nyɔnuŋ / nye ɣleti ƒe ɣeyiɣi va yi.",
    "Meilleure prise en charge des enfants : capacité à éduquer/former les enfants/ Ɖeviwo dzikpɔkpɔ",
    "Je préfère ne pas répondre/ Medi be magblɔ eŋu o",
]
SAVED = [
    "Désir d’avoir des enfants : non spécifié/ Viwo ƒe l_l__: Meny_a nu si ta o.",
    "Ma femme / je suis ménopausée/  Ny_nu__ / nye _keke ƒe _eyi_i va yi.",
    "Meilleure prise en charge des enfants : capacité à éduquer/former les enfants/ _eviwo dzikp_kp_",
    "Je préfère ne pas répondre/ Medi be magbl_ e_u o",
]


def test_link_key_is_the_french_part_even_after_a_bad_save():
    from src.config import link_key
    assert link_key(FORM[1]) == "ma femme / je suis ménopausée"
    assert link_key(FORM[2]).endswith("éduquer/former les enfants")
    assert [link_key(l) for l in SAVED] == [link_key(l) for l in FORM]
    assert link_key("Stigmatisation/Nǔnyanyawiwa") == "stigmatisation"     # no space after "/"
    assert link_key(np.nan) is None


def test_load_statement_links_reads_excel_mac_csv(tmp_path):
    from src.config import load_statement_links, link_key
    text = "list_name,label,linked_statements\n" + "".join(f'barriers,"{l}",statement_1\n' for l in SAVED)
    path = tmp_path / "links.csv"
    path.write_bytes(text.encode("mac_roman", errors="replace"))   # what Excel's plain "CSV" save does
    links = load_statement_links(path)
    assert links["key"].tolist() == [link_key(l) for l in FORM]


def test_link_problems():
    choices = pd.DataFrame({"list_name": ["barriers"] * 4, "label": FORM})
    links = pd.DataFrame({"list_name": ["barriers"] * 3,
                          "label": [SAVED[0], SAVED[3], "Distance trop grande/ Didi _ut_"],
                          "linked_statements": ["statement_1, statement_99", np.nan, np.nan]})
    problems = quality.link_problems(links, ["statement_1", "statement_2"], choices)
    assert len(problems) == 2
    assert any("statement_99" in p for p in problems) and any("Distance" in p for p in problems)
