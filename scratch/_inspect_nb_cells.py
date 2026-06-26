"""Inspect cell layout of the kaggle and colab notebooks."""
import json

def inspect(path, label):
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    cells = nb.get("cells", [])
    print(f"\n===== {label} : {len(cells)} cells =====")
    for i, c in enumerate(cells):
        src = c.get("source", [])
        if isinstance(src, str):
            lines = src.split("\n")
        else:
            lines = src
        first = ""
        for ln in lines:
            s = ln.strip()
            if s:
                first = s
                break
        n = len(lines)
        ct = c.get("cell_type", "?")
        print(f"  cell[{i:3d}] {ct:8s} lines={n:5d}  first={first[:90]!r}")

if __name__ == "__main__":
    inspect("Road2AI_ApplePie/notebooks/kaggle-hybridrag-decomp-anchor-v2.ipynb", "KAGGLE")
    inspect("Road2AI_ApplePie/notebooks/retrieval_colab_decomp_submit.ipynb", "COLAB")
