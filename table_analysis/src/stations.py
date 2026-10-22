"""Standardise free-text radio station answers (the station_*_other questions) to one entry per station.

Used by 01_mapping.ipynb. Kept here (not in the notebook) so it can be tested on synthetic answers.

Each answer is split into the stations it names, in this order:
  1. lookup  -- the researcher-reviewed CSV (DIR_STATIONS_OTHER_CLEANED): exact answer -> cleaned names
  2. rules   -- STATION_PATTERNS below find every known station anywhere in the answer, so
                "Radio rfi 89.0 Radio Lomé 99.5" gives two stations
  3. fuzzy   -- what's left is split into pieces, and each piece is matched to the closest known
                name (STATION_PATTERNS names + the questionnaire's choice labels), e.g. "GVK" -> Radio VGK
  4. typed   -- a piece that matches nothing is kept as typed (tidied) and reported for review
Answers saying the respondent didn't listen become NO_LISTENING ("-99", the questionnaire's code).

Frequencies: the same station name can be heard on different frequencies, so each mention keeps the
FM frequency written next to it ("Radio Lomé 99.5"). A mention without one gets the frequency of the
nearest respondent (by GPS) who named the same station with a frequency -- impute_frequencies().
"""
import difflib
import re
import unicodedata

import numpy as np
import pandas as pd

NO_LISTENING = "-99"
UNCLEAR = "Other (unclear)"   # names a radio but no identifiable station (e.g. "Radio FM")

# (pattern on lower-case, accent-stripped text, standard name). Every match in an answer counts.
STATION_PATTERNS = [
    (r"v[e]nus", "Radio Vénus"),
    (r"\bbridi\b", "Bridi FM"),
    (r"\bbridge\b", "Bridge FM"),
    (r"\blome\b", "Radio Lomé"),
    (r"tchaoudjo", "Radio Tchaoudjo"),
    (r"tsetsi|tchaitchi", "Radio Tsétsi"),
    (r"elavagnon", "Radio Élavagnon"),
    (r"la paix", "Radio la Paix"),
    (r"\bmari[ao]\b|radiomaria", "Radio Maria"),
    (r"tabala", "Radio Tabala"),
    (r"binah", "Radio Binah"),
    (r"\bstade[rl]\b", "Stader FM"),
    (r"\bjva\b", "Radio JVA"),
    (r"\brcv\b", "RCV FM"),
    (r"bassar", "Radio Bassar"),
    (r"kamina", "Radio Kamina"),
    (r"sport ?fm", "Sport FM"),
    (r"\bwake\b|ouake", "Radio Ouaké (Bénin)"),
    (r"bryth|bright", "Bright FM"),
    (r"tchamba", "Radio Tchamba"),
    (r"dounegno", "Dunenyo"),
    (r"\bkara\b", "Radio Kara"),
    (r"sama ?fm", "Sama FM"),
    (r"bagah", "Radio Bagah"),
    (r"\brfi\b", "RFI"),
    (r"islamique", "Radio Islamique"),
    (r"etincelle", "Radio Étincelle"),
    (r"delta", "Delta FM"),
    (r"cosmos", "Cosmos Sotououa"),
    (r"\bkoz(ah)? ?fm", "Kozah FM"),
    (r"dankpen|danpken", "Radio Dankpen"),
    (r"reveil", "Radio Réveil"),
    (r"dawol", "Radio Dawol"),
    (r"\bvgk\b|\bgvk\b", "Radio VGK"),
    (r"evangile", "Radio Évangile"),
    (r"ephata", "Radio Éphata"),
    (r"\bfafa\b", "Radio Fafa"),
    (r"\brtds\b", "RTDS"),
    (r"voi[xe] d[eu] vo\b|\brvv\b", "Radio la Voix du Vo"),
    (r"firmam", "Radio Firmament"),
    (r"moisson|mission finale", "Radio Moisson Finale"),
    (r"novissi", "Fraterniténovissi"),
    (r"atlanti", "Radio Atlantic"),
    (r"agou", "Radio la Voix d'Agou"),
    (r"\bbbc\b", "BBC"),
    (r"radio arabe( regionale)?", UNCLEAR),
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
    """'Radio Lomé 99.5' -> ('Radio Lomé', '99.5'); 'Radio Lomé' -> ('Radio Lomé', None)."""
    m = re.match(r"^(.*?)\s+(\d{2,3}\.\d)$", entry.strip())
    return (m.group(1), m.group(2)) if m else (entry.strip(), None)


def format_entry(name, freq):
    return f"{name} {freq}" if freq else name


class StationStandardiser:
    """standardise(answer) -> ([(name, frequency or None), ...], method)."""

    def __init__(self, lookup=None, extra_names=(), cutoff=0.8):
        # lookup: {raw answer: "Name 99.5; Name B"} (reviewed CSV); extra_names: e.g. questionnaire choice labels
        self.lookup = {fold(k): v for k, v in (lookup or {}).items()}
        names = [n for _, n in STATION_PATTERNS if n not in (NO_LISTENING, UNCLEAR)] + list(extra_names)
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
            return entries, "lookup"

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
