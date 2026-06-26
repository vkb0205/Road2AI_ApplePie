"""Idempotent patcher for retrieval_colab_decomp_submit.ipynb.

Adds a tqdm progress bar to the §8 batch-processing loop WITHOUT touching
retrieval/generation core logic:

  A. A small `_progress()` helper (with a plain-iterator fallback when tqdm is
     not installed) defined just above the batch-run section.
  B. Wrap the `for rec in records_in:` loop with that helper so a live
     progress bar (X/total, rate, ETA) renders over the query batch.

Re-running is safe: each insertion is guarded by a marker check, so the cell
is never patched twice.

tqdm ships preinstalled on Google Colab; the fallback keeps the cell runnable
everywhere else (incl. a bare local kernel).
"""
import json

NB = "Road2AI_ApplePie/notebooks/retrieval_colab_decomp_submit.ipynb"

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)


def _src(cell):
    s = cell.get("source", [])
    if isinstance(s, str):
        s = s.splitlines(keepends=True)
        cell["source"] = s
    return s


# ---------------------------------------------------------------------------
# Block A — the _progress() helper (tqdm with plain-iterator fallback)
# ---------------------------------------------------------------------------
progress_block = [
    "# --- progress bar for the batch loop -------------------------------------\n",
    "# Wraps the query list with a tqdm bar when tqdm is available (it ships with\n",
    "# Colab); falls back to a plain iterator otherwise so the cell runs unchanged.\n",
    "try:\n",
    "    from tqdm.auto import tqdm as _tqdm\n",
    "except Exception:\n",
    "    _tqdm = None\n",
    "\n",
    "\n",
    "def _progress(iterable, total=None, desc=\"batch\"):\n",
    "    \"\"\"Yield from *iterable* with a tqdm progress bar, or as-is if tqdm absent.\"\"\"\n",
    "    if _tqdm is None:\n",
    "        return iterable\n",
    "    return _tqdm(iterable, total=total, desc=desc, unit=\"q\",\n",
    "                 dynamic_ncols=True, leave=True)\n",
    "\n",
    "\n",
]

# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------
patched = False
for cell in nb["cells"]:
    if cell.get("cell_type") != "code":
        continue
    src = _src(cell)
    if not any("records_in = _json.loads" in s for s in src):
        continue
    # Locate the batch-run cell whether already patched (for rec in _batch_pbar)
    # or not yet patched (for rec in records_in).
    if not any(("for rec in records_in:" in s) or ("for rec in _batch_pbar:" in s)
               for s in src):
        continue

    # Block A: insert the _progress() helper just above the batch-run header.
    if not any("def _progress(" in s for s in src):
        for i, s in enumerate(src):
            if s.strip().startswith("# --- batch run:"):
                src[i:i] = progress_block
                break

    # Block B: wrap the loop iterable with the progress bar.
    if not any("_batch_pbar = _progress" in s for s in src):
        for i, s in enumerate(src):
            if s.strip() == "for rec in records_in:":
                src[i] = (
                    "_batch_pbar = _progress(records_in, total=len(records_in), desc=\"batch\")\n"
                )
                src.insert(i + 1, "for rec in _batch_pbar:\n")
                break

    patched = True
    break

assert patched, "batch-run cell not found — notebook structure changed?"

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
    f.write("\n")

print("patched OK")
