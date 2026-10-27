"""Clear saved outputs from every notebook in this folder (in place). Stdlib only.

Run from anywhere: python clear_notebook_outputs.py
Outputs can contain respondent rows; clearing them lets Claude read and edit the notebooks' code.
"""
import json
from pathlib import Path

for path in sorted(Path(__file__).resolve().parent.glob("*.ipynb")):
    nb = json.loads(path.read_text(encoding="utf-8"))
    n = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            n += bool(cell.get("outputs"))
            cell["outputs"] = []
            cell["execution_count"] = None
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{path.name}: cleared {n} cell outputs")
