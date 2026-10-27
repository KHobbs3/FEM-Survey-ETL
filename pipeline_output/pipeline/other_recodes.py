"""
Recoding of "Autre / Other" write-ins and free-text answers.

Same pattern as the radio station-name review:

1. Export (doesn't change any output):
       python -m pipeline.other_recodes          # export only
       python -m pipeline.other_recodes --ai     # export + AI-suggested codes
   writes every distinct write-in for OTHER_RECODE_QUESTIONS ("Autre" +
   "<question>_other") and FREE_TEXT_RECODE_QUESTIONS (the answer itself is
   free text), with its count and a `recode_to` column, to
   OTHER_WRITEINS_EXPORT. Rows already coded in OTHER_RECODES_PATH keep their
   `recode_to`, so re-running after new data only adds the new write-ins.
   Prints aggregate counts only -- the text stays in the data folder.

   With --ai, questions in AI_CODE_LISTS get `recode_to` suggested by Claude:
   the closest answer from that question's own choice list (e.g. the
   barriers list), or a short new theme when none fits. Only the distinct
   answer texts are sent -- no IDs or other columns. Suggestions are marked
   coded_by = "ai", and the file is written straight to OTHER_RECODES_PATH
   ("...-ai-gen.csv") so the pipeline uses them until they're reviewed.

2. Review the export, fill or correct `recode_to` (an existing answer label,
   or a new one), and save it as OTHER_RECODES_PATH. Blank `recode_to` =
   leave as is. run_pipeline.py then calls apply_other_recodes() right after
   loading, so every page sees the recoded answers. With no recodes file it
   does nothing.

Keyed by question + write-in text, never by respondent.
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

from pipeline.config import (
    DIR_MAPPED_DATA, OTHER_RECODE_QUESTIONS, FREE_TEXT_RECODE_QUESTIONS, AI_CODE_LISTS,
    AI_MODEL, OTHER_RECODES_PATH, OTHER_WRITEINS_EXPORT,
)

RECODE_COLS = ["question", "other_text", "n", "recode_to", "coded_by"]

# Choice-list answers that aren't a reason, so never offered as a code.
NONSUBSTANTIVE_PREFIXES = ("je ne sais pas", "je préfère ne pas", "préfère ne pas",
                           "aucun", "aucune", "rien")
AI_BATCH_SIZE = 50


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
    """Boolean mask: the answer contains the "Autre" choice."""
    return df[question].apply(
        lambda v: pd.notna(v) and any(is_other_item(p) for p in str(v).split("|"))
    )


def _questions(df: pd.DataFrame):
    """(question, text column, rows to read) for every question present in the data."""
    for q in OTHER_RECODE_QUESTIONS:
        other_col = f"{q}_other"
        if q not in df.columns or other_col not in df.columns:
            print(f"  [other] {q}: skipped -- '{q}' or '{other_col}' not in data")
            continue
        yield q, other_col, _other_rows(df, q)
    for q in FREE_TEXT_RECODE_QUESTIONS:
        if q not in df.columns:
            print(f"  [other] {q}: skipped -- not in data")
            continue
        yield q, q, df[q].notna()


def code_list(df: pd.DataFrame, cols) -> list:
    """Distinct answer labels of the choice-list columns `cols` (e.g. the
    barriers list), minus "Autre" and don't-know/none answers."""
    labels = set()
    for col in cols:
        if col not in df.columns:
            continue
        for v in df[col].dropna():
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
                    recodes_path=OTHER_RECODES_PATH, ai: bool = False) -> pd.DataFrame:
    existing = _load_recodes(recodes_path)
    coded = {(q, _key(t)): (r, b) for q, t, r, b in
             zip(existing["question"], existing["other_text"],
                 existing["recode_to"], existing["coded_by"])}

    rows = []
    for q, text_col, mask in _questions(df):
        text = df.loc[mask, text_col].apply(clean_writein).dropna()
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

        n_coded = sum(1 for k in counts if coded.get((q, k), ("",))[0])
        print(f"  [other] {q}: {int(mask.sum())} answers to code "
              f"({n_blank} blank), {len(counts)} distinct texts, {n_coded} already coded")

    out = pd.DataFrame(rows, columns=RECODE_COLS)
    if ai:
        out = ai_suggest(out, {q: (code_list(df, cfg["lists"]), cfg["about"])
                               for q, cfg in AI_CODE_LISTS.items()})
    out = out.sort_values(["question", "n", "other_text"], ascending=[True, False, True])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  [other] wrote {len(out)} rows -> {out_path}")
    return out


# ── Step 1b: AI suggestions ───────────────────────────────────────────────────

AI_SYSTEM = """You code open-ended answers from a family planning survey of women and men in \
West Africa (Togo, Benin, Niger). Answers are mostly in French, sometimes with local-language \
words or typos. Category labels may be bilingual ("French/local language").

The question being coded is described in <question>. For each answer, pick the ONE category from \
the numbered list that best matches it (for a list of reasons, the respondent's main reason). If no \
category fits, set category to 0 and give a short new theme in French (1-6 words) in new_theme -- \
reuse one of the existing new themes when it fits, so similar answers end up grouped under the \
same theme. If the answer doesn't answer the question (blank, "rien", unintelligible), set \
category to 0 and new_theme to "". Otherwise new_theme is ""."""

AI_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "category": {"type": "integer"},
                    "new_theme": {"type": "string"},
                },
                "required": ["id", "category", "new_theme"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["results"],
    "additionalProperties": False,
}


def ai_suggest(out: pd.DataFrame, codes_by_q: dict) -> pd.DataFrame:
    """Fill blank `recode_to` for the AI_CODE_LISTS questions with Claude's
    suggestion; existing codes (manual or earlier AI runs) are left alone.
    Stops at the first API problem (no key, bad key, network...) keeping the
    codes made so far -- re-running --ai continues with the blank ones."""
    try:
        import anthropic
    except ImportError:
        sys.exit("  [other] --ai needs the Anthropic SDK: pip install anthropic")

    client = anthropic.Anthropic()
    no_key = ("  [other] No working Anthropic API key -- set ANTHROPIC_API_KEY (or run "
              "`ant auth login`) and re-run --ai. Codes made so far are kept.")
    out = out.copy()
    for q, (codes, about) in codes_by_q.items():
        todo = out.index[(out["question"] == q) & (out["recode_to"] == "")]
        if len(todo) == 0:
            continue
        if not codes:
            print(f"  [other] {q}: no choice list found in data -- AI coding skipped")
            continue
        print(f"  [other] {q}: sending {len(todo)} distinct answer texts (no IDs, no other "
              f"columns) to the Anthropic API ({AI_MODEL}), {len(codes)} categories")
        categories = "\n".join(f"{i}. {label}" for i, label in enumerate(codes, 1))
        mine = out["question"] == q
        themes = sorted({r for r, b in zip(out.loc[mine, "recode_to"], out.loc[mine, "coded_by"])
                         if b == "ai" and r and r not in codes})
        n_coded = n_theme = 0

        for start in range(0, len(todo), AI_BATCH_SIZE):
            batch = list(todo[start:start + AI_BATCH_SIZE])
            answers = "\n".join(f"{i}. {out.at[idx, 'other_text']}" for i, idx in enumerate(batch, 1))
            prompt = (f"<question>{about}</question>\n\n"
                      f"<categories>\n{categories}\n</categories>\n\n"
                      f"<existing_new_themes>\n{chr(10).join(themes)}\n</existing_new_themes>\n\n"
                      f"<answers>\n{answers}\n</answers>")
            try:
                response = client.beta.messages.create(
                    model=AI_MODEL,
                    max_tokens=16000,
                    betas=["server-side-fallback-2026-07-01"],
                    fallbacks="default",
                    system=AI_SYSTEM,
                    output_config={"effort": "low",
                                   "format": {"type": "json_schema", "schema": AI_SCHEMA}},
                    messages=[{"role": "user", "content": prompt}],
                )
            except anthropic.AuthenticationError:
                print(no_key)
                return out
            except TypeError as e:   # the SDK's "Could not resolve authentication method"
                if "authentication" not in str(e).lower():
                    raise
                print(no_key)
                return out
            except anthropic.APIError as e:
                print(f"  [other] {q}: Anthropic API error ({type(e).__name__}) -- stopping. "
                      f"Codes made so far are kept; re-run --ai to continue.")
                return out
            if response.stop_reason in ("refusal", "max_tokens"):
                print(f"  [other] {q}: batch {start // AI_BATCH_SIZE + 1} stopped "
                      f"({response.stop_reason}) -- left blank")
                continue
            text = next(b.text for b in response.content if b.type == "text")
            for r in json.loads(text)["results"]:
                if not 1 <= r["id"] <= len(batch):
                    continue
                idx = batch[r["id"] - 1]
                if 1 <= r["category"] <= len(codes):
                    out.at[idx, "recode_to"] = codes[r["category"] - 1]
                    n_coded += 1
                elif r["new_theme"].strip():
                    theme = r["new_theme"].strip()
                    out.at[idx, "recode_to"] = theme
                    if theme not in themes:
                        themes.append(theme)
                    n_theme += 1
                else:
                    continue
                out.at[idx, "coded_by"] = "ai"
        print(f"  [other] {q}: AI coded {n_coded} to the list, {n_theme} to new themes "
              f"({len(themes)} themes), {len(todo) - n_coded - n_theme} left blank")
    return out


# ── Step 2: apply ─────────────────────────────────────────────────────────────

def apply_other_recodes(df: pd.DataFrame, recodes_path=OTHER_RECODES_PATH) -> pd.DataFrame:
    """Swap the "Autre" choice (or, for FREE_TEXT_RECODE_QUESTIONS, the whole
    free-text answer) for its reviewed recode, wherever the text is in the
    recodes file. For multi-select answers only the "Autre" item is replaced;
    the other items are kept."""
    if not os.path.exists(recodes_path):
        print(f"  [other] no recodes file ({recodes_path}) -- answers left as is")
        return df

    rec = _load_recodes(recodes_path)
    rec = rec[rec["recode_to"].str.strip() != ""]
    df = df.copy()
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
        for col in [q] + AI_CODE_LISTS.get(q, {}).get("lists", []):
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


if __name__ == "__main__":
    from pipeline.utils import load_raw

    parser = argparse.ArgumentParser(description="Export write-ins for review.")
    parser.add_argument("--ai", action="store_true",
                        help="suggest codes for AI_CODE_LISTS questions with Claude "
                             "(sends the distinct answer texts to the Anthropic API)")
    args = parser.parse_args()
    # --ai writes straight to the recodes file the pipeline applies
    # (OTHER_RECODES_PATH, "...-ai-gen.csv"), to be reviewed later.
    export_writeins(load_raw(DIR_MAPPED_DATA), ai=args.ai,
                    out_path=OTHER_RECODES_PATH if args.ai else OTHER_WRITEINS_EXPORT)
