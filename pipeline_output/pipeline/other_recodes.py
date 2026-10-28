"""
Recoding of "Autre / Other" write-ins and free-text answers.

The same file in Niger, Benin and Togo; each country's questions and paths
are in its config.py (OTHER_RECODE_QUESTIONS, FREE_TEXT_RECODE_QUESTIONS,
CODE_LISTS, OTHER_RECODES_PATH).

1. Refresh (doesn't change any output):
       python -m pipeline.other_recodes          # export only, to OTHER_WRITEINS_EXPORT
       python -m pipeline.other_recodes --fuzzy  # update OTHER_RECODES_PATH
   lists every distinct write-in for OTHER_RECODE_QUESTIONS ("Autre" +
   "<question>_other") and FREE_TEXT_RECODE_QUESTIONS (the answer itself is
   free text), with its count and a `recode_to` column. With --fuzzy, a
   write-in that clearly matches one of the question's own answer labels
   (CODE_LISTS) gets `recode_to` filled (fuzzy_suggest; runs locally, nothing
   is sent anywhere). Codes already in the file are never overwritten, and
   write-ins no longer in the data stay in the file with their codes.
   Prints aggregate counts only -- the text stays in the data folder.

2. Review the file: check the fuzzy codes and fill or correct `recode_to`
   (an existing answer label, or a new one). Blank = not reviewed yet;
   "-" = reviewed, keep the answer as written.

3. Apply, at the table_analysis stage: table_analysis/02z_recode_other.py
   runs recode_dataset() on the survey data files, so every later notebook,
   script, the app pipeline and the linkage read the recoded answers. Each
   recoded question keeps its answers as given in `_pre_recode__<question>`,
   so re-running (after new codes or new data) always starts from the
   original answers. run_pipeline.py runs the same step on the file it reads.

Keyed by question + write-in text, never by respondent. The Niger Phone
Pulse survey reuses the matching below: 2_phone pulse/etl_pipeline/other_recodes.py.
"""

import argparse
import csv
import difflib
import os
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.config import (
    DIR_MAPPED_DATA, OTHER_RECODE_QUESTIONS, FREE_TEXT_RECODE_QUESTIONS, CODE_LISTS,
    OTHER_RECODES_PATH, OTHER_WRITEINS_EXPORT, CLEAR_AI_CODES,
)
try:   # this country's keyword rules (pipeline/recode_rules.py), if it has them
    from pipeline.recode_rules import RULES, FALLBACK_RULES
except ImportError:
    RULES, FALLBACK_RULES = [], set()

RECODE_COLS = ["question", "other_text", "n", "recode_to", "coded_by"]

# Choice-list answers that aren't a reason, so never offered as a code.
NONSUBSTANTIVE_PREFIXES = ("je ne sais pas", "je préfère ne pas", "préfère ne pas",
                           "aucun", "aucune", "rien")
# recode_to = "-": reviewed, leave the answer as written (and don't suggest a code for it again)
KEEP_AS_WRITTEN = "-"
# Column holding a recoded question's answers as given (see recode_dataset)
PRE_RECODE_PREFIX = "_pre_recode__"


def _answers(df: pd.DataFrame, question: str) -> pd.Series:
    """The question's answers as given: its _pre_recode__ copy if the data was already recoded."""
    pre = PRE_RECODE_PREFIX + question
    return df[pre] if pre in df.columns else df[question]


def is_other_item(item) -> bool:
    """True for the "Autre"/"Other" catch-all choice. Labels are bilingual
    ('Autre : veuillez préciser/...', 'Wani / Autre'), so any '/'-segment
    starting with autre/other counts; '-22' is the form's code for it."""
    s = str(item).strip()
    if s == "-22" or s.startswith("-22 "):
        return True
    return any(seg.strip().casefold().startswith(("autre", "other")) for seg in s.split("/"))


def _key(text) -> str:
    """Match key for a write-in: trimmed, case-folded, whitespace collapsed."""
    return re.sub(r"\s+", " ", str(text)).strip().casefold()


def clean_writein(value):
    if pd.isna(value):
        return np.nan
    s = re.sub(r"\s+", " ", str(value)).strip()
    return s if s else np.nan


def _other_rows(df: pd.DataFrame, question: str) -> pd.Series:
    """Boolean mask: the answer as given contains the "Autre" choice."""
    return _answers(df, question).apply(
        lambda v: pd.notna(v) and any(is_other_item(p) for p in str(v).split("|"))
    )


def _questions(df: pd.DataFrame):
    """(question, write-in texts, rows to read) for every question present in the data."""
    for q in OTHER_RECODE_QUESTIONS:
        other_col = f"{q}_other"
        if q not in df.columns or other_col not in df.columns:
            print(f"  [other] {q}: skipped -- '{q}' or '{other_col}' not in data")
            continue
        yield q, df[other_col], _other_rows(df, q)
    for q in FREE_TEXT_RECODE_QUESTIONS:
        if q not in df.columns:
            print(f"  [other] {q}: skipped -- not in data")
            continue
        text = _answers(df, q)
        yield q, text, text.notna()


def code_list(df: pd.DataFrame, cols) -> list:
    """Distinct answer labels of the choice-list columns `cols` (e.g. the
    barriers list), as given, minus "Autre" and don't-know/none answers."""
    labels = set()
    for col in cols:
        if col not in df.columns:
            continue
        for v in _answers(df, col).dropna():
            for item in str(v).split("|"):
                item = item.strip()
                french = item.split("/")[0].strip().casefold()
                if item and not is_other_item(item) and not french.startswith(NONSUBSTANTIVE_PREFIXES):
                    labels.add(item)
    return sorted(labels)


def _load_recodes(path=OTHER_RECODES_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=RECODE_COLS)
    rec = pd.read_csv(path, dtype=str).fillna("")
    missing = {"question", "other_text", "recode_to"} - set(rec.columns)
    if missing:
        raise ValueError(f"{path} is missing column(s): {sorted(missing)}")
    if "coded_by" not in rec.columns:
        rec["coded_by"] = ""
    return rec


# ── Step 1: export ────────────────────────────────────────────────────────────

def export_writeins(df: pd.DataFrame, out_path=OTHER_WRITEINS_EXPORT,
                    recodes_path=OTHER_RECODES_PATH, fuzzy: bool = False) -> pd.DataFrame:
    existing = _load_recodes(recodes_path)
    # With CLEAR_AI_CODES (config.py), codes from the retired AI coding
    # (coded_by = "ai") are dropped, so those texts are re-coded by fuzzy
    # matching or left blank for hand-coding; otherwise they're kept as codes.
    if CLEAR_AI_CODES:
        n_ai = int((existing["coded_by"] == "ai").sum())
        if n_ai:
            print(f"  [other] {n_ai} AI-coded texts cleared -- re-coded by --fuzzy or left blank")
        existing = existing[existing["coded_by"] != "ai"]
    coded = {(q, _key(t)): (r, b) for q, t, r, b in
             zip(existing["question"], existing["other_text"],
                 existing["recode_to"], existing["coded_by"])}

    rows, seen = [], set()
    for q, texts, mask in _questions(df):
        text = texts[mask].apply(clean_writein).dropna()
        n_blank = int(mask.sum()) - len(text)

        counts = {}
        for t in text:
            k = _key(t)
            spelling, n = counts.get(k, (t, 0))
            counts[k] = (spelling, n + 1)
        for k, (spelling, n) in counts.items():
            recode_to, coded_by = coded.get((q, k), ("", ""))
            rows.append({"question": q, "other_text": spelling, "n": n,
                         "recode_to": recode_to, "coded_by": coded_by})
            seen.add((q, k))

        n_coded = sum(1 for k in counts if coded.get((q, k), ("",))[0])
        print(f"  [other] {q}: {int(mask.sum())} answers to code "
              f"({n_blank} blank), {len(counts)} distinct texts, {n_coded} already coded")

    # Write-ins not in this data (e.g. from respondents dropped later in
    # table_analysis) stay in the file with their codes.
    kept = existing[[(q, _key(t)) not in seen for q, t in
                     zip(existing["question"], existing["other_text"])]] \
        if len(existing) else existing
    out = pd.concat([pd.DataFrame(rows, columns=RECODE_COLS), kept[RECODE_COLS]], ignore_index=True)
    out["n"] = pd.to_numeric(out["n"], errors="coerce").fillna(0).astype(int)
    if fuzzy:
        codes_by_q = {q: code_list(df, cols) for q, cols in CODE_LISTS.items()}
        out = rule_suggest(out, codes_by_q)
        out = fuzzy_suggest(out, codes_by_q)
        export_choices(codes_by_q, os.path.join(os.path.dirname(out_path), "other_recode_choices.csv"))
    if "group" in out.columns:   # --fuzzy: similar texts next to each other
        out = out.sort_values(["question", "group", "n", "other_text"], ascending=[True, True, False, True])
    else:
        out = out.sort_values(["question", "n", "other_text"], ascending=[True, False, True])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  [other] wrote {len(out)} rows -> {out_path}")
    return out


# ── Step 1b: local suggestions (no API) ───────────────────────────────────────

def _fold(text) -> str:
    """Comparable form: no accents, lower case, punctuation as spaces."""
    s = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().casefold()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


# French words that carry no meaning for matching a reason to an answer label
_STOP = set("""le la les l de des du d un une et ou a au aux en je j ai as avait avons pas ne n
me m mon ma mes ton ta tes il elle on que qui qu quoi pour par sur sous avec sans ce cet cette
ces c est sont etait se s sa son ses y plus moins tres trop car mais donc lui leur leurs nous
vous ils elles suis es etre avoir fait faire dit dire veux veut peux peut encore deja aussi
tout tous toute toutes rien chose choses quand comme si cela ca ici la bas moi toi soi""".split())

# Auto-code only when the best answer clearly wins; everything else gets candidates to pick from
AUTO_SCORE = 0.55    # combined similarity (0-1) of the best answer
AUTO_MARGIN = 0.10   # and at least this much ahead of the second best


def _stem(word: str) -> str:
    """Light French stemming so 'secondaires' ~ 'secondaire', 'enceintes' ~ 'enceinte'."""
    for suffix in ("ements", "ement", "ations", "ation", "euses", "euse", "eurs", "eur",
                   "ees", "ee", "es", "e", "s", "x"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def _words(text) -> str:
    return " ".join(_stem(w) for w in _fold(text).split() if w not in _STOP and len(w) > 2)


def _label_text(label: str) -> str:
    """The French half of a bilingual 'French/Fon' label (the part people write in)."""
    return label.split("/")[0]


def _similarity(texts: list, labels: list):
    """texts x labels matrix of similarity (0-1): the mean of word-level (stemmed, no stop
    words, so a sentence matches a short label on its key words) and character-level
    (3-5 letter pieces, so typos and word forms still match) TF-IDF cosine similarity.
    Falls back to difflib ratios if scikit-learn isn't installed."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        return np.array([[difflib.SequenceMatcher(None, _fold(t), _fold(l)).ratio() for l in labels]
                         for t in texts])
    sims = []
    for analyzer, prep, ngrams in (("word", _words, (1, 2)), ("char_wb", _fold, (3, 5))):
        corpus = [prep(x) for x in list(labels) + list(texts)]
        if not any(corpus):
            continue
        vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngrams).fit(corpus)
        sims.append(cosine_similarity(vec.transform(corpus[len(labels):]),
                                      vec.transform(corpus[:len(labels)])))
    return np.mean(sims, axis=0) if sims else np.zeros((len(texts), len(labels)))


GROUP_SCORE = 0.5    # write-ins at least this similar to each other go in the same `group`


def _groups(texts: list) -> list:
    """Group number per text (texts linked by similarity >= GROUP_SCORE, chained); 0 = alone."""
    if len(texts) < 2:
        return [0] * len(texts)
    sim = _similarity(texts, texts)
    parent = list(range(len(texts)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if sim[i][j] >= GROUP_SCORE:
                parent[find(i)] = find(j)
    roots = [find(i) for i in range(len(texts))]
    sizes = pd.Series(roots).value_counts()
    numbers = {r: k for k, r in enumerate(sizes.index[sizes > 1], 1)}
    return [numbers.get(r, 0) for r in roots]


def fill_groups(out: pd.DataFrame) -> pd.DataFrame:
    """Within a question's `group`, copy the code when the coded members all agree to the
    uncoded ones (coded_by = "group"). Code one row per group by hand, re-run --fuzzy, done."""
    out = out.copy()
    for (q, g), grp in out[out["group"] > 0].groupby(["question", "group"]):
        codes = set(grp.loc[~grp["recode_to"].isin(["", KEEP_AS_WRITTEN]), "recode_to"])
        if len(codes) == 1:
            blank = grp.index[grp["recode_to"] == ""]
            out.loc[blank, "recode_to"] = codes.pop()
            out.loc[blank, "coded_by"] = "group"
    return out


def fuzzy_suggest(out: pd.DataFrame, codes_by_q: dict, label_text=_label_text) -> pd.DataFrame:
    """Local coding suggestions -- nothing is sent anywhere.

    For every uncoded write-in, scores each of the question's own answer labels (_similarity) and
    writes the 3 best to `candidates` ("label (0.62) | label (0.41) | ..."), to pick from when
    hand-coding. `recode_to` is filled (coded_by = "fuzzy <score>") only when the write-in is the
    label itself (ignoring case/accents/punctuation), or the best label scores >= AUTO_SCORE and
    leads the second by >= AUTO_MARGIN. Review those; anything less clear is left blank.
    `label_text` gives the part of a label that write-ins are compared with."""
    out = out.copy()
    out["candidates"] = ""
    out["group"] = 0
    for q in out["question"].unique():
        mine = list(out.index[out["question"] == q])
        out.loc[mine, "group"] = _groups([out.at[i, "other_text"] for i in mine])
    n_before = int((out["recode_to"] != "").sum())
    out = fill_groups(out)
    if int((out["recode_to"] != "").sum()) > n_before:
        print(f"  [other] {int((out['recode_to'] != '').sum()) - n_before} texts coded from a "
              f"coded text in the same group (coded_by = group)")
    for q, codes in codes_by_q.items():
        todo = list(out.index[(out["question"] == q) & (out["recode_to"] == "")])
        if not todo or not codes:
            continue
        exact = {}
        for label in codes:
            exact.setdefault(_fold(label_text(label)), label)
        sim = _similarity([out.at[i, "other_text"] for i in todo], [label_text(l) for l in codes])
        n_exact = n_auto = 0
        for row, idx in enumerate(todo):
            order = np.argsort(-sim[row])
            best = sim[row][order[0]]
            second = sim[row][order[1]] if len(order) > 1 else 0.0
            out.at[idx, "candidates"] = " | ".join(f"{codes[j]} ({sim[row][j]:.2f})"
                                                   for j in order[:3] if sim[row][j] > 0)
            hit = exact.get(_fold(out.at[idx, "other_text"]))
            if hit:
                out.at[idx, "recode_to"], out.at[idx, "coded_by"] = hit, "fuzzy exact"
                n_exact += 1
            elif best >= AUTO_SCORE and best - second >= AUTO_MARGIN:
                out.at[idx, "recode_to"] = codes[order[0]]
                out.at[idx, "coded_by"] = f"fuzzy {best:.2f}"
                n_auto += 1
        best_scores = sim.max(axis=1)
        n_groups = out.loc[todo, "group"].pipe(lambda g: g[g > 0].nunique())
        print(f"  [other] {q}: {len(todo)} uncoded texts -- {n_exact} exact, {n_auto} close "
              f"(coded_by = fuzzy <score>, review these); {len(todo) - n_exact - n_auto} left blank "
              f"with 3 candidates each; {n_groups} groups of similar texts")
        print(f"           best-candidate score: median {np.median(best_scores):.2f}; "
              + ", ".join(f">= {c}: {int((best_scores >= c).sum())}" for c in (0.25, 0.35, 0.45, 0.55)))
    out = out.sort_values(["question", "group", "n"], ascending=[True, True, False])
    return out


_LETTERS = str.maketrans({"ƙ": "k", "Ƙ": "k", "ɗ": "d", "Ɗ": "d", "ɓ": "b", "Ɓ": "b", "ƴ": "y", "Ƴ": "y",
                          "’": " ", "‘": " "})


def _rule_fold(text) -> str:
    """_fold, keeping the Hausa hooked letters as plain letters and curly apostrophes as
    spaces (like straight ones: "don’t" -> "don t"), padded with spaces."""
    return f" {_fold(str(text).translate(_LETTERS))} "


def rule_suggest(out: pd.DataFrame, codes_by_q: dict) -> pd.DataFrame:
    """Keyword rules (pipeline/recode_rules.py) for blank write-ins: each gets the answer option
    of the rule that matches earliest in its text (coded_by = "rule <name>"). Rule codes are
    worked out again on every run (so rule changes apply); other codes are never changed.
    No rules file -> nothing changes."""
    if not RULES:
        return out
    rules = [(name, re.compile(text), [re.compile(p) for p in labels]) for name, text, labels in RULES]
    out = out.copy()
    old = out["coded_by"].str.startswith("rule")
    out.loc[old, ["recode_to", "coded_by"]] = ""
    for q, codes in codes_by_q.items():
        todo = list(out.index[(out["question"] == q) & (out["recode_to"] == "")])
        if not todo or not codes:
            continue
        folded = {label: _rule_fold(label) for label in codes}
        target = {}                      # rule -> this question's answer option
        for name, _, patterns in rules:
            for p in patterns:
                hits = [label for label in codes if p.search(folded[label])]
                if hits:
                    target[name] = min(hits, key=len)
                    break
        used = {}
        for idx in todo:
            text = _rule_fold(out.at[idx, "other_text"])
            best = None   # (is fallback, position, rule): specific rules first, then earliest
            for name, pattern, _ in rules:
                if name in target and (m := pattern.search(text)):
                    key = (name in FALLBACK_RULES, m.start(), name)
                    if best is None or key[:2] < best[:2]:
                        best = key
            if best:
                name = best[2]
                out.at[idx, "recode_to"], out.at[idx, "coded_by"] = target[name], f"rule {name}"
                used[name] = used.get(name, 0) + 1
        if used:
            print(f"  [other] {q}: {sum(used.values())} of {len(todo)} uncoded texts coded by keyword rules "
                  f"(coded_by = rule <name>): " + ", ".join(f"{k} {v}" for k, v in
                                                            sorted(used.items(), key=lambda kv: -kv[1])))
    return out


def export_choices(codes_by_q: dict, path: str) -> None:
    """The valid answer labels per question (questionnaire labels, no respondent data), as a
    reference for hand-coding recode_to."""
    rows = [{"question": q, "answer_label": label} for q, codes in codes_by_q.items()
            for label in codes]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows, columns=["question", "answer_label"]).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  [other] answer labels for hand-coding -> {path}")


# ── Step 2: apply ─────────────────────────────────────────────────────────────

def apply_other_recodes(df: pd.DataFrame, recodes_path=OTHER_RECODES_PATH) -> pd.DataFrame:
    """Swap the "Autre" choice (or, for FREE_TEXT_RECODE_QUESTIONS, the whole
    free-text answer) for its reviewed recode, wherever the text is in the
    recodes file. For multi-select answers only the "Autre" item is replaced;
    the other items are kept. Starts from the answers as given, so applying
    to already-recoded data gives the same result as applying once."""
    if not os.path.exists(recodes_path):
        print(f"  [other] no recodes file ({recodes_path}) -- answers left as is")
        return df

    rec = _load_recodes(recodes_path)
    rec = rec[~rec["recode_to"].str.strip().isin(["", KEEP_AS_WRITTEN])]
    df = df.copy()
    for q in OTHER_RECODE_QUESTIONS + FREE_TEXT_RECODE_QUESTIONS:
        if q in df.columns:
            df[q] = _answers(df, q)
    for q, grp in rec.groupby("question"):
        free_text = q in FREE_TEXT_RECODE_QUESTIONS
        text_col = q if free_text else f"{q}_other"
        if q not in df.columns or text_col not in df.columns:
            print(f"  [other] WARNING: recodes given for '{q}' but '{q}'/'{text_col}' not in data")
            continue
        # recode_to can be typed in one language ("Artisan") while the data
        # holds the full bilingual label ("Artisan/..."); map it to that label
        # so it joins the existing category instead of making a new one.
        labels = {}
        for col in [q] + CODE_LISTS.get(q, []):
            if col in df.columns and not (free_text and col == q):
                for v in df[col].dropna():
                    for item in str(v).split("|"):
                        labels.setdefault(_key(item), item.strip())
                        for seg in item.split("/"):
                            labels.setdefault(_key(seg), item.strip())
        mapping = {_key(t): labels.get(_key(r), r.strip())
                   for t, r in zip(grp["other_text"], grp["recode_to"])}
        mask = df[q].notna() if free_text else _other_rows(df, q)
        target = df.loc[mask, text_col].apply(
            lambda t: mapping.get(_key(t)) if pd.notna(t) else None
        )
        target = target.dropna()

        def _swap(value, new):
            if free_text:
                return new
            items, out = str(value).split("|"), []
            for p in items:
                p = p.strip()
                p = new if is_other_item(p) else p
                if p not in out:
                    out.append(p)
            return "|".join(out)

        df.loc[target.index, q] = [_swap(df.at[i, q], new) for i, new in target.items()]
        print(f"  [other] {q}: recoded {len(target)} of {int(mask.sum())} answers")
    return df


# ── Step 3: recode the data files (table_analysis) ───────────────────────────

def recode_dataset(paths, recodes_path=OTHER_RECODES_PATH, refresh: bool = True) -> None:
    """Rewrite each survey data file in `paths` with the recoded answers, so
    every script that reads it gets them. Only the recoded questions change:
    each gets a `_pre_recode__<question>` column with its answers as given
    (added once, never changed), and the question column is rebuilt from it
    with the current recodes file -- so re-running picks up new codes and
    never recodes twice. Every other value is written back as read (all
    columns are read as text). With `refresh`, the recodes file is first
    updated from the first file (--fuzzy). Missing files are skipped."""
    paths = [Path(p) for p in paths]
    for p in [p for p in paths if not p.exists()]:
        print(f"  [other] {p.name}: not found -- skipped")
    paths = [p for p in paths if p.exists()]
    questions = OTHER_RECODE_QUESTIONS + FREE_TEXT_RECODE_QUESTIONS
    for i, path in enumerate(paths):
        with open(path, "rb") as f:
            encoding = "utf-8-sig" if f.read(3) == b"\xef\xbb\xbf" else "utf-8"
        with open(path, newline="", encoding=encoding) as f:
            header = next(csv.reader(f))      # written back as is (blank index name, repeated names)
        data = pd.read_csv(path, dtype=str, keep_default_na=False, encoding=encoding)
        new_cols = []
        for q in questions:
            if q in data.columns and PRE_RECODE_PREFIX + q not in data.columns:
                data[PRE_RECODE_PREFIX + q] = data[q]
                new_cols.append(PRE_RECODE_PREFIX + q)
        work = data.replace("", np.nan)
        print(f"\n  [other] {path.name}")
        if refresh and i == 0:
            export_writeins(work, out_path=recodes_path, recodes_path=recodes_path, fuzzy=True)
        recoded = apply_other_recodes(work, recodes_path)
        for q in questions:
            if q in data.columns:
                data[q] = recoded[q].fillna("")
        data.to_csv(path, index=False, header=header + new_cols, encoding=encoding)
        print(f"  [other] recoded answers written -> {path}")


if __name__ == "__main__":
    from pipeline.utils import load_raw

    parser = argparse.ArgumentParser(description="Export write-ins for review.")
    parser.add_argument("--fuzzy", action="store_true",
                        help="pre-fill recode_to where a write-in clearly matches an existing "
                             "answer (runs locally, no API), and write the answer labels per "
                             "question to other_recode_choices.csv for hand-coding the rest")
    args = parser.parse_args()
    # --fuzzy updates the recodes file that table_analysis/02z_recode_other.py
    # and run_pipeline apply (OTHER_RECODES_PATH), to be reviewed later.
    export_writeins(load_raw(DIR_MAPPED_DATA), fuzzy=args.fuzzy,
                    out_path=OTHER_RECODES_PATH if args.fuzzy else OTHER_WRITEINS_EXPORT)
