#!/usr/bin/env python3
"""PreToolUse guard: keep participant data out of Claude's context.

Blocks Claude from:
  * reading/editing/grepping anything under the project's data folders, or files named
    like SurveyCTO/QA exports (*_WIDE*, *Full Data*, .dta, .sav);
  * reading notebooks that still have saved outputs (outputs can contain respondent rows);
  * shell commands that reference data paths, data-path config variables, or that execute
    the analysis notebooks/scripts (the researcher runs those and reports aggregates).

Wired up in .claude/settings.json with onFailure: "block", so a crash here fails closed.
Stdlib only. Exit 0 with a JSON "deny" decision to block; exit 0 silently to allow.
"""
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(os.environ.get("CLAUDE_PROJECT_DIR") or Path(__file__).resolve().parents[2]).resolve()
PROTECTED_DIRS = [
    ROOT / "data",
    ROOT / "table_analysis" / "data",
    ROOT / "table_analysis" / "radio_matching",  # free-text "other" answers
]
PROTECTED_NAME = re.compile(r"_WIDE|Full Data|\.dta$|\.sav$", re.I)

# Files in data/meta with no respondent data (the QA export and location overrides stay blocked)
META = ROOT / "table_analysis" / "data" / "meta"
ALLOWED_FILES = {META / n for n in (
    "[Ewe_Fr]+Questionnaire+de+Sondage.xlsx",
    "stations_other_cleaned.csv",
    "0_togo_surveybounds.geojson",
    "Togo QA Quantitative survey - Settlements chosen.csv",
    "driver_barrier_statement_links.csv",
)}

BASH_PATTERNS = [
    (r"(^|[\s'\"=(/.])data/", "references a data/ path"),
    (r"(^|[\s;&|(])(grep|rg|ag|find|cat|cp|mv|rsync|tar|zip|xargs|du)\b[^|;&]*\s['\"]?(\.\./)?data['\"]?(\s|$)",
     "passes the data folder to a command"),
    (r"_WIDE|Full Data", "references a raw/QA export"),
    (r"nbconvert[^|;&]*--execute|\bpapermill\b|jupyter\s+(execute|run)\b", "executes a notebook"),
    (r"\b(run_pipeline|validate_pipeline)\b", "runs the PII pipeline"),
    (r"\bpython[\d.]*\b[^|;&]*\b0\d[a-z]?_\w+\.py", "runs an analysis script"),
    (r"inspect_input_schemas", "runs a data-reading script (the researcher runs it)"),
]
# Config path variables are only a problem when the same command also loads/executes something.
DIR_VAR = re.compile(r"\bDIR_(RAW|CLEANED|MAPPED|LOCATED|WEIGHTED|QA|META|VALIDATE|FINAL|SETTLEMENTS|STATEMENT)")
LOADS = re.compile(r"read_(csv|excel|file|parquet|stata|json|table)|ExcelFile|open\(|exec\(|runpy|%run|from src|import src")
RECURSIVE = re.compile(r"\bgrep\b[^|;&]*\s-\w*[rR]|\b(rg|ag)\b|\bfind\b[^|;&]*-(exec|ok)|\bxargs\b")
RESTRICTED = re.compile(r"--include|--exclude-dir|\s-g\s|--glob|-name\s+['\"]?\*\.(py|md)['\"]?")
SAFE_LISTING = re.compile(r"^\s*(ls|find)\b(?!.*(\||;|&&|`|\$\(|-exec|-ok|-delete|-fprint))")


def deny(reason):
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": f"PII guard: {reason}. See CLAUDE.md -- write a script for the researcher to run instead.",
    }}))
    sys.exit(0)


def resolve(p, cwd):
    p = Path(os.path.expanduser(str(p)))
    return (p if p.is_absolute() else cwd / p).resolve()


def is_protected(path):
    return any(path == d or d in path.parents for d in PROTECTED_DIRS) or bool(PROTECTED_NAME.search(path.name))


def contains_protected(path):
    return any(path == d or path in d.parents for d in PROTECTED_DIRS)


def notebook_has_outputs(path):
    try:
        nb = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return any(c.get("outputs") for c in nb.get("cells", []) if c.get("cell_type") == "code")


def notebooks_with_outputs():
    return [p.relative_to(ROOT) for p in ROOT.rglob("*.ipynb")
            if ".ipynb_checkpoints" not in p.parts and not any(d in p.parents for d in PROTECTED_DIRS)
            and notebook_has_outputs(p)]


def main():
    inp = json.load(sys.stdin)
    tool = inp.get("tool_name", "")
    ti = inp.get("tool_input", {}) or {}
    cwd = Path(inp.get("cwd") or os.getcwd())

    if tool in ("Read", "Edit", "Write", "NotebookEdit"):
        raw = ti.get("file_path") or ti.get("notebook_path")
        if not raw:
            return
        path = resolve(raw, cwd)
        if is_protected(path):
            deny(f"{path.name} is participant data or sits in a protected data folder")
        if path.suffix == ".ipynb" and tool != "Write" and notebook_has_outputs(path):
            deny(f"{path.name} has saved outputs, which may contain respondent rows. Ask the researcher to "
                 f"clear them (jupyter nbconvert --clear-output --inplace \"{path.name}\")")

    elif tool == "Grep":
        path = resolve(ti.get("path") or cwd, cwd)
        if is_protected(path):
            deny("grep target is in a protected data folder")
        if path.suffix == ".ipynb" and notebook_has_outputs(path):
            deny(f"{path.name} has saved outputs")
        if path.is_dir():
            glob = ti.get("glob") or ""
            ftype = ti.get("type") or ""
            code_only = ftype in ("py", "python", "md", "markdown") or \
                (bool(glob) and not re.search(r"csv|xls|json|ipynb|geojson|gpkg|txt|^\*+$|^\*\*/\*$", glob, re.I))
            if contains_protected(path) and not code_only:
                deny("this search would cover a data folder -- narrow `path` (e.g. table_analysis/src) "
                     "or restrict with glob/type to code files like *.py")
            if not code_only and notebooks_with_outputs():
                deny("notebooks with saved outputs are in scope -- restrict with glob/type to *.py/*.md")

    elif tool == "Bash":
        cmd = ti.get("command", "")
        if not SAFE_LISTING.search(cmd):  # bare ls/find only list names
            for pattern, why in BASH_PATTERNS:
                if re.search(pattern, cmd):
                    deny(f"command {why}")
            if DIR_VAR.search(cmd) and LOADS.search(cmd):
                deny("command loads data through a config path variable")
        if RECURSIVE.search(cmd) and not RESTRICTED.search(cmd) and contains_protected(cwd.resolve()):
            deny("recursive search from here would read the data folders -- add --include='*.py' / "
                 "--exclude-dir=data, -g '*.py', or search a narrower folder")
        if ".ipynb" in cmd:
            dirty = notebooks_with_outputs()
            if dirty:
                deny(f"command touches notebooks and these have saved outputs: {[str(p) for p in dirty]}")


if __name__ == "__main__":
    main()
