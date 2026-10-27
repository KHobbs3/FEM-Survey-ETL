"""Standardise free-text radio station answers (the station_*_other questions) to one entry per station.

Used by 01_mapping.ipynb. Kept here (not in the notebook) so it can be tested on synthetic answers.

Each answer is split into the stations it names, in this order:
  1. lookup  -- the researcher-reviewed CSV (DIR_STATIONS_OTHER_CLEANED): exact answer -> cleaned names
  2. rules   -- STATION_PATTERNS below find every known station anywhere in the answer, so
                "Radio Tokpa 96.7 Golfe FM 105.7" gives two stations
  3. fuzzy   -- what's left is split into pieces, and each piece is matched to the closest known
                name (STATION_PATTERNS names + the questionnaire's choice labels), e.g. "Tokpla" -> Radio Tokpa
  4. typed   -- a piece that matches nothing is kept as typed (tidied) and reported for review
Answers saying the respondent didn't listen become NO_LISTENING ("-99", the questionnaire's code).

Frequencies: the same station name can be heard on different frequencies, so each mention keeps the
FM frequency written next to it ("Radio Tokpa 96.7"). A mention without one gets the frequency of the
nearest respondent (by GPS) who named the same station with a frequency -- impute_frequencies().

In 01_mapping (before codes become labels): standardise_answers() for every station_*_other column,
then merge_other() puts each answer's stations in place of the -22 ("other") code; names matching a
questionnaire choice become that choice's code, so a station both ticked and written in counts once.
"""
import difflib
import re
import unicodedata

import numpy as np
import pandas as pd

NO_LISTENING = "-99"
UNCLEAR = "Other (unclear)"   # names a radio but no identifiable station (e.g. "Radio FM")

# (pattern on lower-case, accent-stripped text, standard name). Every match in an answer counts.
# Benin stations. Starting list = the stations modelled in CloudRF (benin_app's radio overlap table) plus
# national/international ones; extend from 01b_review_stations.py output ("typed"/"fuzzy" names).
STATION_PATTERNS = [
    (r"tokpa", "Radio Tokpa"),
    (r"benin culture", "Radio Bénin Culture"),
    (r"\bortb\b|radio (nationale|benin)\b(?! culture)", "ORTB (Radio Bénin)"),
    (r"radio parakou", "Radio Parakou"),
    (r"sedohoun", "Radio Sèdohoun"),
    (r"hemicycle", "Radio Hémicycle"),
    (r"arakou", "Radio Arakou"),
    (r"\bocean\b", "Océan FM"),
    (r"maranatha", "Radio Maranatha"),
    (r"immaculee", "Radio Immaculée Conception"),
    (r"\bbip\b", "BIP Radio"),
    (r"gerdes", "Radio Gerdès"),
    (r"frisson", "Frisson Radio"),
    (r"ekaaro|ejiire", "Radio Ekaaro Ejiire"),
    (r"deeman|\bdeman\b", "Deeman Radio"),
    (r"couffo", "Radio Couffo"),
    (r"\bcapp\b", "CAPP FM"),
    # Standard names that are questionnaire choices use the choice label, so station_key matches
    # and write-ins merge into the choice's code
    (r"\bkiff\b|atlantique ?fm|radio atlantique", "Kiff FM"),
    (r"adja ?ouere", "Radio Adja Ouèrè"),
    (r"vallee", "La Voix de la Vallée"),
    (r"ton+ass?e", "Radio Tonnasse"),
    (r"carrefour", "Radio Carrefour"),
    (r"tonignon", "Radio Tonignon"),
    (r"solidarite", "Solidarité FM"),
    (r"golfe", "Golfe FM"),
    (r"planete", "Radio Planète"),
    (r"soleil", "Soleil FM"),
    (r"fraternite", "Fraternité FM"),
    (r"\bmari[ao]\b|radiomaria", "Radio Maria"),
    (r"\brfi\b", "RFI"),
    (r"\bbbc\b", "BBC"),
    (r"\bvoa\b|voix de l.?amerique", "VOA"),
    # didn't listen ("n'ai pas écouté", "je n'ai écouté la radio", "n'a pas écouté radio")
    (r"\bn\W?ai?\b.{0,15}ecout|pas ecout", NO_LISTENING),
]

# Words that don't identify a station; removed from leftover pieces before fuzzy matching
_NOISE = re.compile(r"\b(radio|fm|station|parfois|aussi|egalement|quand|ca|vient|hier|je|la|le|les|de|du|des|"
                    r"et|ou|regionale?|en|plus|souvent)\b")
_SPLIT = re.compile(r"[,;/+&\n]|\bet\b|\bou\b")
_FREQ = re.compile(r"(?<![\d.,])(\d{2,3})(?:[.,](\d))?(?!\d)")
_FREQ_WINDOW = 15   # a frequency belongs to a station if written within this many characters of it


def fold(text):
    """Lower case, accents stripped, whitespace collapsed (newlines kept as separators)."""
    s = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().lower()
    return "\n".join(" ".join(line.split()) for line in s.splitlines()).strip()


def station_key(name):
    """Comparable station name: no accents, case, frequencies, brackets, 'radio' or 'fm'."""
    s = re.sub(r"\(.*?\)|\d+([.,]\d+)?", " ", fold(name))
    s = re.sub(r"\b(radio|fm)\b|[^a-z]+", " ", s)
    return " ".join(s.split())


def _freqs(text, start, end):
    """[(span, '99.5')] for FM frequencies (87-108.5 MHz) written in text[start:end]."""
    out = []
    for m in _FREQ.finditer(text, start, end):
        v = float(m.group(1) + ("." + m.group(2) if m.group(2) else ""))
        if 87 <= v <= 108.5:
            out.append((m.span(), f"{v:.1f}"))
    return out


def _tidy(piece):
    """A leftover piece as a display name: frequencies/brackets dropped, words capitalised."""
    s = re.sub(r"\(.*?\)|\d+([.,]\d+)?|\bfm\b", " ", piece, flags=re.I)
    return " ".join(s.split()).strip(" .-'").title()


def split_name_freq(entry):
    """'Radio Tokpa 96.7' -> ('Radio Tokpa', '96.7'); 'Radio Tokpa' -> ('Radio Tokpa', None)."""
    m = re.match(r"^(.*?)\s+(\d{2,3}\.\d)$", entry.strip())
    return (m.group(1), m.group(2)) if m else (entry.strip(), None)


def _standard_name(name):
    """A reviewed-lookup name as STATION_PATTERNS' standard name when exactly one station pattern
    matches it (so "Radio Tonassé" from an older review becomes "Radio Tonnasse"); else unchanged."""
    text = fold(name)
    hits = {std for pattern, std in STATION_PATTERNS if std != NO_LISTENING and re.search(pattern, text)}
    return hits.pop() if len(hits) == 1 else name


def format_entry(name, freq):
    return f"{name} {freq}" if freq else name


class StationStandardiser:
    """standardise(answer) -> ([(name, frequency or None), ...], method)."""

    def __init__(self, lookup=None, extra_names=(), cutoff=0.8):
        # lookup: {raw answer: "Name 99.5; Name B"} (reviewed CSV); extra_names: e.g. questionnaire choice labels
        self.lookup = {fold(k): v for k, v in (lookup or {}).items()}
        # Cleaned names in the reviewed lookup are known stations too, so new spellings of them fuzzy match
        reviewed = [split_name_freq(n)[0] for v in self.lookup.values() for n in v.split(";") if n.strip()]
        names = [n for _, n in STATION_PATTERNS] + list(extra_names) + reviewed
        names = [n for n in names if n not in (NO_LISTENING, UNCLEAR)]
        self.by_key = {}
        for n in names:
            self.by_key.setdefault(station_key(n), n)
        self.cutoff = cutoff

    def _fuzzy(self, piece):
        key = station_key(_NOISE.sub(" ", fold(piece)))
        if len(key.replace(" ", "")) < 3:
            return None
        if key in self.by_key:
            return self.by_key[key]
        hit = difflib.get_close_matches(key, list(self.by_key), n=1, cutoff=self.cutoff)
        return self.by_key[hit[0]] if hit else None

    def standardise(self, answer):
        if answer is None or pd.isna(answer) or not str(answer).strip() or fold(answer) == "0":
            return [], "empty"
        text = fold(answer)
        if text in self.lookup:
            entries = [split_name_freq(n) for n in self.lookup[text].split(";") if n.strip()]
            return [(_standard_name(n), f) for n, f in entries], "lookup"

        # 1. Known stations, in order of appearance (overlapping matches: first one wins)
        found = []
        for pattern, name in STATION_PATTERNS:
            for m in re.finditer(pattern, text):
                s, e = m.span()   # widen to whole words, so "firmam" covers all of "firmament"
                while s > 0 and text[s - 1].isalpha():
                    s -= 1
                while e < len(text) and text[e].isalpha():
                    e += 1
                found.append((s, e, name))
        found.sort()
        matches = []
        for s, e, name in found:
            if not matches or s >= matches[-1][1]:
                matches.append((s, e, name))

        # 2. Frequencies: the first one written just after a station, else just before it
        claimed, freq = set(), [None] * len(matches)
        for i, (s, e, _) in enumerate(matches):
            nxt = matches[i + 1][0] if i + 1 < len(matches) else len(text)
            for span, f in _freqs(text, e, min(e + _FREQ_WINDOW, nxt)):
                freq[i] = f
                claimed.add(span)
                break
        for i, (s, e, _) in enumerate(matches):
            if freq[i] is None:
                prev = matches[i - 1][1] if i else 0
                before = [(sp, f) for sp, f in _freqs(text, max(prev, s - _FREQ_WINDOW), s) if sp not in claimed]
                if before:
                    freq[i] = before[-1][1]
                    claimed.add(before[-1][0])
        entries = [(m[0], (m[2], f)) for m, f in zip(matches, freq)]
        method = "rules" if matches else None

        # 3. Parts of the answer (split on commas, new lines, "et", "ou", ...) that name no known station:
        # fuzzy match each one, else keep it as typed. A part that names a known station belongs to it --
        # its other words are qualifiers ("Vénus de Sokodé", "Fafa FM (du Ghana)", "Moisson finale").
        bounds = [0] + [m.end() for m in _SPLIT.finditer(text)]
        ends = [m.start() for m in _SPLIT.finditer(text)] + [len(text)]
        for a, b in zip(bounds, ends):
            if any(s < b and e > a for s, e, _ in matches):
                continue
            piece = text[a:b]
            if not re.search(r"[a-z]{3,}", _NOISE.sub(" ", piece)):
                continue
            piece_freqs = [(sp, f) for sp, f in _freqs(text, a, b) if sp not in claimed]
            f = piece_freqs[0][1] if piece_freqs else None
            match = self._fuzzy(piece)
            name = match or _tidy(piece)
            if not name:
                continue
            entries.append((a, (name, f)))
            method = "fuzzy" if match and method != "typed" else "typed" if not match else method

        # One entry per station: drop a frequency-less duplicate of a station named with a frequency
        out = []
        for _, (name, f) in sorted(entries, key=lambda x: x[0]):
            if (name, f) in out or (f is None and any(n == name for n, _ in out)):
                continue
            out = [(n, g) for n, g in out if not (n == name and g is None and f is not None)] + [(name, f)]
        if len(out) > 1:   # a named station outweighs "didn't listen" / "unclear"
            out = [e for e in out if e[0] not in (NO_LISTENING, UNCLEAR)] or out
        if not out:
            return [(UNCLEAR, None)], "unclear"
        return out, method


def impute_frequencies(mentions, name_col="name", freq_col="freq", lat_col="lat", lon_col="lon", max_km=None):
    """Fill a missing frequency from the nearest mention (great-circle distance between respondents' GPS)
    of the same station that states one, if it is within max_km. Mentions without GPS, of a station nobody
    gave a frequency for, or with no such respondent within max_km, stay missing. Adds 'freq_imputed' (bool)
    and 'freq_donor_km' (distance to the nearest respondent with a frequency). Returns a copy."""
    out = mentions.copy()
    out["freq_imputed"] = False
    out["freq_donor_km"] = np.nan
    for name, grp in out.groupby(name_col):
        if name in (NO_LISTENING, UNCLEAR):
            continue
        has_gps = grp[lat_col].notna() & grp[lon_col].notna()
        donors = grp[grp[freq_col].notna() & has_gps]
        need = grp[grp[freq_col].isna() & has_gps]
        if donors.empty or need.empty:
            continue
        lat1, lon1 = np.radians(need[lat_col].to_numpy(float))[:, None], np.radians(need[lon_col].to_numpy(float))[:, None]
        lat2, lon2 = np.radians(donors[lat_col].to_numpy(float))[None, :], np.radians(donors[lon_col].to_numpy(float))[None, :]
        h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
        km = 2 * 6371 * np.arcsin(np.sqrt(h))
        nearest = km.argmin(axis=1)
        dist = km[np.arange(len(need)), nearest]
        out.loc[need.index, "freq_donor_km"] = dist
        close = dist <= max_km if max_km is not None else np.ones(len(need), bool)
        out.loc[need.index[close], freq_col] = donors[freq_col].to_numpy()[nearest][close]
        out.loc[need.index[close], "freq_imputed"] = True
    return out


# ---------------------------------------------------------------------------
# Applying it to the survey (01_mapping.ipynb, 01b_review_stations.py)
# ---------------------------------------------------------------------------

def load_lookup(path):
    """Reviewed CSV -> {raw answer: "Name 99.5; Name B"} (first column = answer, second = cleaned names).
    Returns {} if the file doesn't exist yet."""
    from pathlib import Path
    if path is None:
        return {}
    if not Path(path).exists():
        print(f"WARNING: reviewed station lookup not found at {path} -- no reviewed answers applied")
        return {}
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return dict(zip(df.iloc[:, 0], df.iloc[:, 1]))


def station_choices(xlsform_path, question):
    """{choice label: code} for a select question's choice list in the XLSForm, plus the question's type
    (e.g. 'select_multiple'), as (choices, qtype). Bilingual "French/Fon" labels keep the French half."""
    survey = pd.read_excel(xlsform_path, sheet_name="survey", dtype=str)
    choices = pd.read_excel(xlsform_path, sheet_name="choices", dtype=str)
    row = survey.loc[survey["name"].str.strip() == question, "type"]
    if row.empty:
        raise KeyError(f"{question} not in the XLSForm survey sheet")
    parts = row.iloc[0].split()
    qtype, list_name = parts[0], (parts[1] if len(parts) > 1 else None)
    code_col = "name" if "name" in choices.columns else "value"
    label_col = next(c for c in choices.columns if str(c).startswith("label"))
    ch = choices[choices["list_name"].str.strip() == list_name]
    labels = ch[label_col].astype(str).str.split("/").str[0].str.strip()
    return dict(zip(labels, ch[code_col].str.strip())), qtype


def standardise_answers(data, other_cols, std, lat_col="gps-Latitude", lon_col="gps-Longitude", max_km=10):
    """One row per station mention in the station_*_other columns, frequencies imputed within max_km.
    Columns: row (index of data), column, answer, name, freq, method, freq_imputed, freq_donor_km."""
    rows = []
    for col in other_cols:
        for idx, answer in data[col].items():
            entries, method = std.standardise(answer)
            for name, freq in entries:
                rows.append({"row": idx, "column": col, "answer": answer, "name": name, "freq": freq,
                             "method": method,
                             "lat": data.at[idx, lat_col] if lat_col in data.columns else np.nan,
                             "lon": data.at[idx, lon_col] if lon_col in data.columns else np.nan})
    mentions = pd.DataFrame(rows, columns=["row", "column", "answer", "name", "freq", "method", "lat", "lon"])
    for c in ["lat", "lon"]:
        mentions[c] = pd.to_numeric(mentions[c], errors="coerce")
    return impute_frequencies(mentions, max_km=max_km).drop(columns=["lat", "lon"])


def entries_by_answer(mentions, column):
    """{row: ["Name 99.5", ...]} for one station_*_other column, after imputation."""
    m = mentions[mentions["column"] == column]
    return {row: [format_entry(n, f if pd.notna(f) else None) for n, f in zip(g["name"], g["freq"])]
            for row, g in m.groupby("row", sort=False)}


def merge_other(selected, entries, label_to_code, other_code="-22"):
    """Put an answer's stations in place of the 'other' code of a select_multiple value ("1 5 -22").

    Entries naming a questionnaire choice become that choice's code (once, so a station both ticked and
    written in counts once); "-99" stays a code; each other entry becomes one token with spaces as
    underscores, so the whitespace-split mapping keeps it whole and 01_mapping's '|' list has one item per
    station (turn "_" back into spaces after mapping). If the answer named nothing, the other code is kept."""
    codes = str(selected).split() if selected is not None and not pd.isna(selected) else []
    if not entries:
        return " ".join(codes)
    keyed = {station_key(label): code for label, code in label_to_code.items()}
    extra = []
    for entry in entries:
        name, _ = split_name_freq(entry)
        code = NO_LISTENING if name == NO_LISTENING else keyed.get(station_key(name))
        if code is not None:
            if code not in codes:
                codes.append(code)
        elif entry not in extra:
            extra.append(entry)
    codes = [c for c in codes if c != other_code]
    codes += [re.sub(r"\s+", "_", e.strip()) for e in extra]
    return " ".join(codes)


def _code_str(value):
    """A select value as read from CSV ('1 5 -22', 3, 3.0) -> '1 5 -22' / '3'; None if missing."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value).strip()


def drop_missing_codes(value, miss_codes=("-88", "-99")):
    """-88/-99 on their own are kept; mixed with stations they're dropped."""
    kept = [t for t in value.split() if t not in miss_codes]
    return " ".join(kept) if kept else value


def combine_station_columns(data, questions, std, choices_by_question, max_km=10, other_code="-22",
                            lat_col="gps-Latitude", lon_col="gps-Longitude"):
    """For each select_multiple station question q, put the standardised stations from q_other in place of
    the other code (merge_other), then drop -88/-99 mixed with stations. Unanswered questions stay missing.
    choices_by_question: {q: {choice label: code}}. Returns (data copy, mentions)."""
    data = data.copy()
    mentions = standardise_answers(data, [q + "_other" for q in questions], std, lat_col, lon_col, max_km)
    for q in questions:
        by_row = entries_by_answer(mentions, q + "_other")
        combined = []
        for idx, value in data[q].items():
            s = _code_str(value)
            if s is None:
                combined.append(value)
                continue
            merged = merge_other(s, by_row.get(idx, []), choices_by_question.get(q, {}), other_code)
            combined.append(drop_missing_codes(merged) if merged else merged)
        data[q] = combined
    return data, mentions


def export_review(mentions, choices_by_question, out_dir, lookup=None):
    """Write the station write-in review files to out_dir (station names, answers as typed and counts
    only -- no respondent rows) and print a summary. Called by 00_clean_data.ipynb after
    combine_station_columns, so the review is produced on every run:

      00_station_other_review.csv               every distinct answer per question -> the questionnaire
                                                choice(s) it was merged into, the station(s) kept as their
                                                own entry, how it was read, respondents, needs_review
      00_station_other_not_on_questionnaire.csv every standardised station that matched no questionnaire
                                                choice: how it was read, mentions, respondents
      stations_other_cleaned_DRAFT.csv          answer -> standardised entries for answers not yet in the
                                                reviewed lookup; correct it and save it as
                                                DIR_STATIONS_OTHER_CLEANED to apply it on the next run

    needs_review = read by fuzzy matching, kept as typed, or unclear (no identifiable station)."""
    from pathlib import Path
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lookup = lookup or {}
    m = mentions[mentions["method"].notna() & (mentions["method"] != "empty")].copy()
    m["question"] = m["column"].str.replace(r"_other$", "", regex=True)
    m["entry"] = [format_entry(n, f if pd.notna(f) else None) for n, f in zip(m["name"], m["freq"])]

    keyed = {q: {station_key(label): label for label in labels} for q, labels in choices_by_question.items()}

    def choice_for(question, name):
        if name == NO_LISTENING:
            return "Did not listen (-99)"
        return keyed.get(question, {}).get(station_key(name))

    m["questionnaire_choice"] = [choice_for(q, n) for q, n in zip(m["question"], m["name"])]
    review_methods = {"fuzzy", "typed", "unclear"}

    answers = (m.groupby(["question", "answer"], sort=False)
               .agg(method=("method", "first"),
                    merged_into_choice=("questionnaire_choice", lambda s: "; ".join(dict.fromkeys(s.dropna()))),
                    kept_as_own_station=("entry", lambda s: "; ".join(dict.fromkeys(
                        s[m.loc[s.index, "questionnaire_choice"].isna()]))),
                    standardised=("entry", lambda s: "; ".join(dict.fromkeys(s))),
                    respondents=("row", "nunique"))
               .reset_index())
    answers["in_reviewed_lookup"] = answers["answer"].isin(lookup)
    answers["needs_review"] = answers["method"].isin(review_methods) | (answers["standardised"] == UNCLEAR)
    answers = answers.sort_values(["needs_review", "respondents"], ascending=[False, False])

    own = m[m["questionnaire_choice"].isna()]
    not_on_q = (own.groupby("name")
                .agg(read_by=("method", lambda s: ", ".join(sorted(set(s)))),
                     mentions=("row", "size"), respondents=("row", "nunique"),
                     questions=("question", lambda s: ", ".join(sorted(set(s)))))
                .reset_index().rename(columns={"name": "station"})
                .sort_values("respondents", ascending=False))

    draft = (answers.loc[~answers["in_reviewed_lookup"], ["answer", "standardised"]]
             .drop_duplicates("answer").rename(columns={"standardised": "cleaned"}))

    answers.to_csv(out_dir / "00_station_other_review.csv", index=False, encoding="utf-8-sig")
    not_on_q.to_csv(out_dir / "00_station_other_not_on_questionnaire.csv", index=False, encoding="utf-8-sig")
    draft.to_csv(out_dir / "stations_other_cleaned_DRAFT.csv", index=False, encoding="utf-8-sig")

    n_ans = len(answers)
    print(f"\nStation write-ins: {n_ans} distinct answers ({int(answers['needs_review'].sum())} need review), "
          f"{len(m)} station mentions")
    print(f"  merged into a questionnaire choice: {int(m['questionnaire_choice'].notna().sum())} mentions")
    print(f"  kept as a station not on the questionnaire: {len(own)} mentions, {len(not_on_q)} stations "
          f"({', '.join(not_on_q['station'].head(8))}{'...' if len(not_on_q) > 8 else ''})")
    print(f"  reviewed lookup applied to {int(answers['in_reviewed_lookup'].sum())} answers; "
          f"{len(draft)} answers written to stations_other_cleaned_DRAFT.csv for review")
    print(f"  files: {out_dir}/00_station_other_review.csv, 00_station_other_not_on_questionnaire.csv, "
          f"stations_other_cleaned_DRAFT.csv")
    return answers, not_on_q


def review_tables(mentions):
    """(answers, stations) for review exports:
    answers  -- every distinct answer per column -> standardised entries ("; ") + method + respondents
    stations -- each station/frequency: mentions, how many frequencies were imputed, max donor distance"""
    m = mentions.copy()
    m["entry"] = [format_entry(n, f if pd.notna(f) else None) for n, f in zip(m["name"], m["freq"])]
    answers = (m.groupby(["column", "answer", "method"], dropna=False)
               .agg(standardised=("entry", lambda s: "; ".join(dict.fromkeys(s))),
                    respondents=("row", "nunique"))
               .reset_index().sort_values(["column", "respondents"], ascending=[True, False]))
    stations = (m.assign(freq=m["freq"].fillna("(none)"))
                .groupby(["name", "freq"])
                .agg(mentions=("row", "size"), imputed=("freq_imputed", "sum"),
                     max_imputed_km=("freq_donor_km", lambda s: s[m.loc[s.index, "freq_imputed"]].max()))
                .reset_index().sort_values(["name", "mentions"], ascending=[True, False]))
    return answers, stations
