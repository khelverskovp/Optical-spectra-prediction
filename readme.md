# Data Setup

This project requires an external data file that is **not included in the repository** due to the size. You must download it manually before running any scripts.

---

## Required File

### `data/data.pckl`
A pickled pandas DataFrame containing crystal structure entries and their computed dielectric properties.

Each row represents one material and must contain the following fields:

| Field | Type | Description |
|---|---|---|
| `mat_id` | `str` | Material identifier |
| `structure` | `pymatgen.Structure` | Crystal structure object |
| `ipa_indirect_gap` | `float` | Indirect band gap (eV) |
| `ipa_direct_gap` | `float` | Direct band gap (eV) |
| `ipa_epsI_0/1/2` | `float` or `np.ndarray` | Imaginary dielectric tensor components |
| `ipa_epsR_0/1/2` | `float` or `np.ndarray` | Real dielectric tensor components |

---

## Where to Download

> [Download from Figshare](https://figshare.com/articles/dataset/Data_for_evaluating_OptiMate3B/30257551?file=58427512)

Once downloaded, place the file at:

```
data/data.pckl
```

---

## After Downloading

Run the graph construction script to preprocess the data:

```bash
python data/build_ipa.py
```

This will read `data/data.pckl`, filter materials by band gap, build cutoff-radius graphs (cutoff = 6.0 Å), and save the processed dataset to `data/dataset_cutoff_ipa.pckl`.

Then run the RPA script to build the RPA dataset:

```bash
python data/build_rpa.py
```

---

## Filtering Applied

The preprocessing script automatically drops:
- Materials with `ipa_indirect_gap < 0.1 eV` (metals / near-zero gap)
- Materials with `ipa_direct_gap > 10 eV` (very wide band gap insulators)