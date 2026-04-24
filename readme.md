# Data Setup
This project requires external data files that are **not included in the repository** due to their size. You must download them manually before running any scripts.
---
## Required Files
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

### `data/database_RPA/`
A directory of JSON files, each representing a `ComputedStructureEntry` with RPA-level dielectric spectra. Each file must contain the fields `rpa_epsI`, `rpa_epsR`, `rpa_kppa`, `ipa_indirect_gap`, `ipa_direct_gap`, and `mat_id`. Used by `build_rpa.py` to swap IPA labels for RPA labels on the pre-built graphs.

---
## Where to Download
> [Download data.pckl from Figshare](__https://figshare.com/articles/dataset/Data_for_evaluating_OptiMate3B/30257551?file=58427512__)

Once downloaded, place the file at:

data/data.pckl

> [Download database_RPA from Figshare](https://figshare.com/articles/dataset/Dielectric_Function_of_Semiconductors_and_Insulators_under_the_Independent_Particle_Approximation_and_the_Random_Phase_Approximation/30257689?file=58427713)

Once downloaded, place the directory at:

data/database_RPA/

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
This will read `data/database_RPA/`, filter materials by band gap, and build the RPA dataset by deep-copying the IPA graphs and replacing their labels with RPA dielectric spectra. The result is saved to `data/dataset_cutoff_rpa.pckl`.

---
## Filtering Applied
The preprocessing scripts automatically drop:
- Materials with `ipa_indirect_gap < 0.1 eV` (metals / near-zero gap)
- Materials with `ipa_direct_gap > 10 eV` (very wide band gap insulators)
- The material with `mat_id == "agm005546161"` (RPA only)