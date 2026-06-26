#!/usr/bin/env python3
"""Adapt retrieval_colab_decomp_submit.ipynb for Kaggle.

Kaggle differences vs Colab:
  - No Google Drive.  Data comes from a Kaggle Dataset attached to the notebook
    (auto-mounted at /kaggle/input/<dataset-name>/).
  - Output goes to /kaggle/working/ (the only persistent dir Kaggle keeps).
  - Secrets via kaggle_secrets.UserSecretsClient, not google.colab.userdata.
  - The repo can be git-cloned (internet ON) or bundled into the dataset.
  - HF cache should go to /kaggle/working/ or a dataset, not Drive.
  - Metadata: kernelspec display name "Kaggle", remove colab provenance block,
    keep accelerator GPU.

This patcher reads the Colab notebook and writes a NEW file
retrieval_kaggle_decomp_submit.ipynb with all Colab-specific references
replaced.  The original is untouched.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_NB = ROOT / "notebooks" / "retrieval_colab_decomp_submit.ipynb"
DST_NB = ROOT / "notebooks" / "retrieval_kaggle_decomp_submit.ipynb"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _src(cell):
    """Return the cell's source as a list of lines (each ending with \\n
    except possibly the last)."""
    s = cell.get("source", [])
    if isinstance(s, str):
        s = s.splitlines(keepends=True)
    return list(s)


def _set_src(cell, lines):
    cell["source"] = lines


def find_line(lines, needle, start=0):
    for i in range(start, len(lines)):
        if needle in lines[i]:
            return i
    return -1


def replace_in_lines(lines, old, new):
    """Replace every occurrence of *old* with *new* across all lines.
    Returns the number of lines changed."""
    changed = 0
    for i, ln in enumerate(lines):
        if old in ln:
            lines[i] = ln.replace(old, new)
            changed += 1
    return changed


def replace_block(lines, start_anchor, end_anchor, new_block):
    """Replace the inclusive range [start_anchor_line .. end_anchor_line] with
    new_block (a list of line strings).  Anchors are matched by substring."""
    s = find_line(lines, start_anchor)
    if s < 0:
        return False
    e = find_line(lines, end_anchor, s + 1)
    if e < 0:
        e = s  # single-line replacement
    lines[s : e + 1] = new_block
    return True


# ---------------------------------------------------------------------------
# Patches
# ---------------------------------------------------------------------------
def patch_metadata(nb):
    """Rewrite notebook metadata for Kaggle."""
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    }


def patch_title_cell(cell):
    """Intro markdown — replace Colab references with Kaggle."""
    lines = _src(cell)
    replace_in_lines(lines, "(Google Colab)", "(Kaggle)")
    replace_in_lines(lines, "on Google Colab", "on Kaggle")
    replace_in_lines(
        lines,
        "upload them to DATA_DIR on Drive.",
        "attach them as a Kaggle Dataset (set DATA_DIR in §1).",
    )
    _set_src(cell, lines)


def patch_markdown_cell(cell):
    """General markdown cleanup — Colab/Drive wording → Kaggle."""
    lines = _src(cell)
    replace_in_lines(lines, "clone repo, mount Drive, install deps, configure",
                     "clone repo, attach Dataset, install deps, configure")
    replace_in_lines(lines, "confirm the files you put on Drive are found",
                     "confirm the files in your Kaggle Dataset are found")
    replace_in_lines(lines, "on Drive are found", "in your Kaggle Dataset are found")
    _set_src(cell, lines)


def patch_setup_config_block(lines):
    """Replace the Configuration block at the top of §1.

    Old (Colab):
        from google.colab import drive
        drive.mount('/content/drive')
        # ===== Configuration (edit these) =====
        ...
        REPO_DIR    = "/content/Road2AI_ApplePie"
        DATA_DIR = "/content/drive/Shareddrives/R2AI/data/stage6_data"
        DEV_DIR  = "/content/Road2AI_ApplePie/dev_set"
        ...
        HF_CACHE_DIR = "/content/drive/Shareddrives/R2AI/data/cache_decomp"
        INPUT_QUERIES_PATH  = ""
        OUTPUT_RESULTS_PATH = "/content/results_batch_decomposition.json"

    New (Kaggle):
        # ===== Configuration (edit these) =====
        ...
        REPO_DIR    = "/kaggle/working/Road2AI_ApplePie"
        # Stage-6 data: attach a Kaggle Dataset, then point DATA_DIR at it.
        # e.g.  KAGGLE_DATASET = "your-username/stage6-data"
        KAGGLE_DATASET = "vkb0205/stage6-data"          # <-- edit to YOUR dataset slug
        DATA_DIR = f"/kaggle/input/{KAGGLE_DATASET.split('/')[-1]}"
        DEV_DIR  = "/kaggle/working/Road2AI_ApplePie/dev_set"
        ...
        HF_CACHE_DIR = "/kaggle/working/hf_cache"
        INPUT_QUERIES_PATH  = ""
        OUTPUT_RESULTS_PATH = "/kaggle/working/results_batch_decomposition.json"
    """
    # 1. Remove the first two lines (drive.mount).
    drive_start = find_line(lines, "from google.colab import drive")
    if drive_start >= 0:
        # delete the drive.mount line(s) too
        mount_line = find_line(lines, "drive.mount('/content/drive')", drive_start)
        if mount_line >= 0:
            del lines[mount_line : mount_line + 1]
        del lines[drive_start : drive_start + 1]

    # 2. REPO_DIR
    replace_in_lines(
        lines,
        'REPO_DIR    = "/content/Road2AI_ApplePie"',
        'REPO_DIR    = "/kaggle/working/Road2AI_ApplePie"',
    )

    # 3. DATA_DIR + DEV_DIR  — replace the two lines and insert a KAGGLE_DATASET knob.
    #    Also fix the stale "Folder on your Google Drive" comment above.
    replace_in_lines(
        lines,
        "# Folder on your Google Drive that holds the Stage-6 artifacts + dev_set.",
        "# Kaggle Dataset that holds the Stage-6 artifacts + dev_set.",
    )
    data_line = find_line(lines, 'DATA_DIR = "/content/drive/Shareddrives/R2AI/data/stage6_data"')
    if data_line >= 0:
        new_block = [
            "# Stage-6 data: attach a Kaggle Dataset to this notebook, then set\n",
            "# KAGGLE_DATASET to the dataset slug.  Kaggle auto-mounts it under\n",
            "# /kaggle/input/<dataset-name>/  — DATA_DIR is derived from the slug.\n",
            "KAGGLE_DATASET = \"vkb0205/stage6-data\"          # <-- edit to YOUR dataset slug\n",
            "DATA_DIR = f\"/kaggle/input/{KAGGLE_DATASET.split('/')[-1]}\"\n",
            'DEV_DIR  = "/kaggle/working/Road2AI_ApplePie/dev_set"\n',
        ]
        lines[data_line : data_line + 2] = new_block  # replace DATA_DIR + DEV_DIR lines

    # 4. HF cache
    replace_in_lines(
        lines,
        'HF_CACHE_DIR = "/content/drive/Shareddrives/R2AI/data/cache_decomp"  # if HF_CACHE_ON_DRIVE',
        'HF_CACHE_DIR = "/kaggle/working/hf_cache"  # if HF_CACHE_ON_DRIVE',
    )

    # 5. Input / output paths
    replace_in_lines(
        lines,
        'INPUT_QUERIES_PATH  = ""                         # e.g. "/content/drive/.../my_queries.json"',
        'INPUT_QUERIES_PATH  = ""                         # e.g. "/kaggle/input/<dataset>/my_queries.json"',
    )
    replace_in_lines(
        lines,
        'OUTPUT_RESULTS_PATH = "/content/results_batch_decomposition.json"',
        'OUTPUT_RESULTS_PATH = "/kaggle/working/results_batch_decomposition.json"',
    )


def patch_setup_body(lines):
    """Replace the §1 body: Drive mount block + google.colab.userdata."""
    # --- 1b. Mount Google Drive block → Kaggle data-dir check
    drive_mount_start = find_line(lines, "# --- 1b. Mount Google Drive")
    if drive_mount_start >= 0:
        # find the end of the drive.mount block (the blank line after DATA print)
        data_print = find_line(lines, 'print("[DATA_DIR exists]"', drive_mount_start)
        end = find_line(lines, 'print("[DEV_DIR  exists]"', data_print)
        if end >= 0:
            end += 1  # include the DEV_DIR print line
            new_block = [
                "# --- 1b. Confirm the Kaggle Dataset is attached ----------------------\n",
                "# Kaggle auto-mounts attached datasets under /kaggle/input/.  We just\n",
                "# verify the paths are accessible — no Drive mount needed.\n",
                "DATA = Path(DATA_DIR)\n",
                "DEV  = Path(DEV_DIR)\n",
                'print("[DATA_DIR exists]", DATA.exists(), DATA)\n',
                'print("[DEV_DIR  exists]", DEV.exists(),  DEV)\n',
            ]
            lines[drive_mount_start:end] = new_block

    # --- google.colab.userdata → kaggle_secrets
    ud_start = find_line(lines, 'from google.colab import userdata')
    if ud_start >= 0:
        # The original block is:
        #   try:                              <- try_line (one above ud_start)
        #       from google.colab import userdata
        #       _tok = userdata.get("HF_TOKEN")
        #       if _tok: os.environ["HF_TOKEN"] = _tok
        #   except Exception:
        #       if HF_TOKEN: os.environ["HF_TOKEN"] = HF_TOKEN  <- last line
        # Back up to include the 'try:' line so we don't leave it orphaned.
        try_line = ud_start - 1
        if try_line >= 0 and "try:" not in lines[try_line]:
            try_line = ud_start  # safety: no try: above, start at import
        except_line = find_line(lines, "except Exception:", ud_start)
        if except_line < 0:
            end = ud_start + 4  # fallback: replace 4 lines
        else:
            end = except_line + 2  # include except + the if HF_TOKEN fallback line
        new_block = [
            "try:\n",
            "    from kaggle_secrets import UserSecretsClient\n",
            "    _tok = UserSecretsClient().get_secret(\"HF_TOKEN\")\n",
            "    if _tok: os.environ[\"HF_TOKEN\"] = _tok\n",
            "except Exception:\n",
            "    if HF_TOKEN: os.environ[\"HF_TOKEN\"] = HF_TOKEN\n",
        ]
        lines[try_line:end] = new_block


def patch_verify_cell(lines):
    """§2 verify cell — 'upload them to DATA_DIR on Drive' → Kaggle wording."""
    replace_in_lines(
        lines,
        "❌ Missing required input files — upload them to DATA_DIR on Drive.",
        "❌ Missing required input files — attach the Kaggle Dataset and set DATA_DIR.",
    )


def patch_section8_paths(lines):
    """§8 submission output paths → /kaggle/working/."""
    replace_in_lines(
        lines,
        'RESULTS_PATH        = "/content/results.json"',
        'RESULTS_PATH        = "/kaggle/working/results.json"',
    )
    replace_in_lines(
        lines,
        'SUBMISSION_ZIP_PATH = "/content/submission.zip"',
        'SUBMISSION_ZIP_PATH = "/kaggle/working/submission.zip"',
    )


def patch_cleanup_cell(lines):
    """§10 cleanup — 'decomposition notebook's §9' wording is fine, but the
    nvidia-smi block at the end references nothing Colab-specific.  No-op
    unless there are stray /content/ references."""
    # safety net: replace any remaining /content/ → /kaggle/working/
    for i, ln in enumerate(lines):
        if "/content/" in ln:
            lines[i] = ln.replace("/content/", "/kaggle/working/")
    return lines


def patch_all_cells(nb):
    """Walk every cell and apply the relevant patches."""
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            # markdown cells: global Colab→Kaggle wording for the intro
            if cell is nb["cells"][0]:
                patch_title_cell(cell)
            else:
                patch_markdown_cell(cell)
                lines = _src(cell)
                replace_in_lines(lines, "on Google Colab", "on Kaggle")
                replace_in_lines(lines, "(Google Colab)", "(Kaggle)")
                _set_src(cell, lines)
            continue

        lines = _src(cell)

        # Detect which cell this is by its content and apply the right patch.
        has_config = find_line(lines, "# ===== Configuration (edit these) =====") >= 0
        has_drive_mount = find_line(lines, "# --- 1b. Mount Google Drive") >= 0
        has_userdata = find_line(lines, "from google.colab import userdata") >= 0
        has_results_path = find_line(lines, 'RESULTS_PATH        = "/content/results.json"') >= 0
        has_verify_msg = find_line(lines, "upload them to DATA_DIR on Drive") >= 0
        is_section1 = has_config or has_drive_mount or has_userdata

        if is_section1:
            if has_config:
                patch_setup_config_block(lines)
            if has_drive_mount:
                patch_setup_body(lines)
            if has_userdata and find_line(lines, "from google.colab import userdata") >= 0:
                # may still be present if the drive-mount block was removed
                # but the userdata block is separate
                patch_setup_body(lines)

        if has_verify_msg:
            patch_verify_cell(lines)

        if has_results_path:
            patch_section8_paths(lines)

        # Global safety net: catch any remaining Colab-only strings.
        replace_in_lines(lines, "from google.colab import drive", "")
        replace_in_lines(lines, "drive.mount('/content/drive')", "")
        replace_in_lines(lines, "from google.colab import userdata", "from kaggle_secrets import UserSecretsClient")
        replace_in_lines(lines, "userdata.get(", "UserSecretsClient().get_secret(")

        # Human-readable "Colab" wording → "Kaggle" (comments + strings).
        # NOTE: filenames like retrieval_colab_*.ipynb are left unchanged — they
        # are real notebook names, not environment references.
        replace_in_lines(lines, "the Colab secret", "the Kaggle secret")
        replace_in_lines(lines, "In Colab: Runtime → Change runtime type → T4 GPU",
                         "In Kaggle: Settings → Accelerator → GPU T4 x2, then restart and run all.")
        replace_in_lines(lines, "Make sure the Colab runtime",
                         "Make sure the Kaggle runtime")
        replace_in_lines(lines, "(Runtime → Change runtime type → T4 GPU).",
                         "(Settings → Accelerator → GPU T4 x2).")
        replace_in_lines(lines, "# Colab); falls back", "# Kaggle); falls back")
        replace_in_lines(lines, "the colab DecomposingHybridRetriever",
                         "the Kaggle DecomposingHybridRetriever")
        replace_in_lines(lines, "The colab retriever already fuses",
                         "The Kaggle retriever already fuses")
        replace_in_lines(lines, "identical to retrieval_colab.ipynb",
                         "identical to retrieval_kaggle.ipynb")

        # Replace any lingering /content/ paths.
        for i, ln in enumerate(lines):
            if "/content/drive" in ln:
                lines[i] = ln.replace("/content/drive/Shareddrives/R2AI/data", "/kaggle/input")
            elif "/content/" in ln and "Road2AI_ApplePie" in ln:
                lines[i] = ln.replace("/content/", "/kaggle/working/")
            elif "/content/" in ln and "results" in ln:
                lines[i] = ln.replace("/content/", "/kaggle/working/")
            elif "/content/" in ln and "submission" in ln:
                lines[i] = ln.replace("/content/", "/kaggle/working/")

        # Strip now-empty lines that were drive.mount (avoid leaving blank shells).
        lines = [ln for ln in lines if ln.strip() != "" or ln == "\n"]

        _set_src(cell, lines)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def verify(nb):
    """Check no Colab-specific strings remain."""
    problems = []
    for ci, cell in enumerate(nb["cells"]):
        lines = _src(cell)
        for li, ln in enumerate(lines):
            low = ln.lower()
            if "google.colab" in low:
                problems.append(f"cell[{ci}] L{li}: google.colab → {ln.rstrip()}")
            if "drive.mount" in low:
                problems.append(f"cell[{ci}] L{li}: drive.mount → {ln.rstrip()}")
            if "/content/drive" in low:
                problems.append(f"cell[{ci}] L{li}: /content/drive → {ln.rstrip()}")
            if 'userdata.get' in low:
                problems.append(f"cell[{ci}] L{li}: userdata.get → {ln.rstrip()}")
            # 'Colab' wording — but skip .ipynb filename references (real names).
            if "colab" in low and "kaggle" not in low and ".ipynb" not in low:
                problems.append(f"cell[{ci}] L{li}: 'Colab' wording → {ln.rstrip()}")
    return problems


def compile_check(nb):
    """Compile every code cell to catch syntax errors."""
    import ast

    import re as _re

    errors = []
    _magic_re = _re.compile(r"^\s*[!%]")
    for ci, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        lines = _src(cell)
        # Strip IPython shell escapes (!cmd) and magics (%cmd) — valid in
        # notebooks but not parseable by ast.
        clean = [ln for ln in lines if not _magic_re.match(ln)]
        src = "".join(clean)
        if not src.strip():
            continue
        try:
            ast.parse(src)
        except SyntaxError as e:
            errors.append(f"cell[{ci}] SyntaxError: {e.msg} (line {e.lineno})")
    return errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    assert SRC_NB.exists(), f"source notebook not found: {SRC_NB}"
    print(f"[patch] reading  {SRC_NB}")
    nb = json.loads(SRC_NB.read_text(encoding="utf-8"))

    patch_metadata(nb)
    patch_all_cells(nb)

    # Write
    DST_NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"[patch] wrote    {DST_NB}")

    # Verify
    problems = verify(nb)
    if problems:
        print("\n⚠️  Colab remnants found:")
        for p in problems:
            print(f"  {p}")
    else:
        print("[verify] ✓ no Colab-specific strings remain")

    errors = compile_check(nb)
    if errors:
        print("\n❌ Compile errors:")
        for e in errors:
            print(f"  {e}")
    else:
        print("[compile] ✓ all code cells parse")

    n_cells = len(nb["cells"])
    n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    print(f"[stats] {n_cells} cells ({n_code} code)")

    if problems or errors:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
