"""Map every top-level def/class in kaggle + colab cells to (cell_idx, offset, name)."""
import json, re

DEF_RE = re.compile(r'^(def |class |@)')

def map_defs(path, label):
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    cells = nb.get("cells", [])
    print(f"\n===== {label} =====")
    for i, c in enumerate(cells):
        if c.get("cell_type") != "code":
            continue
        src = c.get("source", [])
        lines = src if isinstance(src, list) else src.split("\n")
        defs = []
        for ln, txt in enumerate(lines):
            if DEF_RE.match(txt):
                name = txt.split("(")[0].split(" ")[1].split(":")[0]
                defs.append((ln, name))
        if defs:
            print(f"  cell[{i:3d}] ({len(lines):4d} ln)")
            for ln, name in defs:
                print(f"      L{ln:4d}  {name}")

map_defs("Road2AI_ApplePie/notebooks/kaggle-hybridrag-decomp-anchor-v2.ipynb", "KAGGLE")
map_defs("Road2AI_ApplePie/notebooks/retrieval_colab_decomp_submit.ipynb", "COLAB")
