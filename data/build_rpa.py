"""
Build dataset_rpa.pckl by deep-copying cutoff IPA graphs and swapping y for RPA labels.
"""

import copy
import json
import os
import pickle
import numpy as np
import pandas as pd
import torch
from pymatgen.entries.computed_entries import ComputedStructureEntry

DATA_DIR = "data"
RPA_DB_DIR = f"{DATA_DIR}/database_RPA"
IPA_CUTOFF = f"{DATA_DIR}/dataset_cutoff_ipa.pckl"
OUT_RPA = f"{DATA_DIR}/dataset_cutoff_rpa.pckl"

# ---------------------------------------------------------------------------
# Step 1: load cutoff IPA graphs
# ---------------------------------------------------------------------------

print("Loading cutoff IPA graphs ...")
with open(IPA_CUTOFF, "rb") as f:
    ipa_graphs = pickle.load(f)
print(f"  {len(ipa_graphs)} IPA graphs loaded.")

# ---------------------------------------------------------------------------
# Step 2: load and filter rpa_df
# ---------------------------------------------------------------------------

print("\nLoading RPA database ...")
rpa_data = []
for filename in sorted(os.listdir(RPA_DB_DIR)):
    if ".json" not in filename:
        continue
    filepath = os.path.join(RPA_DB_DIR, filename)
    with open(filepath, "r") as f:
        try:
            entry = json.load(f)
        except Exception as e:
            print(f"  Could not parse {filename}: {e}")
            continue
    cse = ComputedStructureEntry.from_dict(entry)
    if "rpa_kppa" in cse.parameters:
        rpa_data.append({"structure": cse.structure} | cse.data | cse.parameters)

rpa_df = pd.DataFrame(rpa_data)
rpa_df = rpa_df.drop(rpa_df[rpa_df["ipa_indirect_gap"] < 0.1].index)
rpa_df = rpa_df.drop(rpa_df[rpa_df["ipa_direct_gap"] > 10].index)
rpa_df = rpa_df.drop(rpa_df[rpa_df["mat_id"] == "agm005546161"].index)
print(f"  {len(rpa_df)} RPA materials after filtering.")

# Unpack directional spectra with xx fallback for non-orthorhombic materials
rpa_epsI_xx, rpa_epsI_yy, rpa_epsI_zz = [], [], []
rpa_epsR_xx, rpa_epsR_yy, rpa_epsR_zz = [], [], []

for _, row in rpa_df.iterrows():
    if isinstance(row["rpa_epsI"], float):
        for lst in [rpa_epsI_xx, rpa_epsI_yy, rpa_epsI_zz,
                    rpa_epsR_xx, rpa_epsR_yy, rpa_epsR_zz]:
            lst.append(np.nan)
        continue
    for (out_xx, out_yy, out_zz), src in [
        ((rpa_epsI_xx, rpa_epsI_yy, rpa_epsI_zz), row["rpa_epsI"]),
        ((rpa_epsR_xx, rpa_epsR_yy, rpa_epsR_zz), row["rpa_epsR"]),
    ]:
        out_xx.append(src["xx"])
        out_yy.append(src["yy"] if "yy" in src else src["xx"])
        out_zz.append(src["zz"] if "zz" in src else src["xx"])

rpa_df["rpa_epsI_xx"] = rpa_epsI_xx
rpa_df["rpa_epsI_yy"] = rpa_epsI_yy
rpa_df["rpa_epsI_zz"] = rpa_epsI_zz
rpa_df["rpa_epsR_xx"] = rpa_epsR_xx
rpa_df["rpa_epsR_yy"] = rpa_epsR_yy
rpa_df["rpa_epsR_zz"] = rpa_epsR_zz

rpa_lookup = {row["mat_id"]: row for _, row in rpa_df.iterrows()}

# ---------------------------------------------------------------------------
# Step 3: build RPA graphs by deep-copying IPA graphs and swapping y
# ---------------------------------------------------------------------------

print("\nBuilding RPA graphs ...")
rpa_graphs = []

for graph in ipa_graphs:
    mat_id = graph.mat_id
    if mat_id not in rpa_lookup:
        continue
    row = rpa_lookup[mat_id]
    spec_keys = ["rpa_epsI_xx", "rpa_epsI_yy", "rpa_epsI_zz",
                 "rpa_epsR_xx", "rpa_epsR_yy", "rpa_epsR_zz"]
    if any(not isinstance(row[k], np.ndarray) for k in spec_keys):
        continue

    rpa_epsi = (row["rpa_epsI_xx"] + row["rpa_epsI_yy"] + row["rpa_epsI_zz"]) / 3.0
    rpa_epsr = (row["rpa_epsR_xx"] + row["rpa_epsR_yy"] + row["rpa_epsR_zz"]) / 3.0

    rpa_graph = copy.deepcopy(graph)
    rpa_graph.y = torch.hstack([
        torch.tensor(rpa_epsi, dtype=torch.float32),
        torch.tensor(rpa_epsr, dtype=torch.float32),
    ])
    rpa_graphs.append(rpa_graph)

print(f"  Done. {len(rpa_graphs)} RPA graphs built.")

print(f"Saving RPA dataset to {OUT_RPA} ...")
with open(OUT_RPA, "wb") as f:
    pickle.dump(rpa_graphs, f)
print("  Saved.")

print("\nAll done.")
print(f"  RPA: {len(rpa_graphs)} graphs  ->  {OUT_RPA}")