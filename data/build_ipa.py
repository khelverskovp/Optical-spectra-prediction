"""
Build cutoff-radius graphs for all entries in data_df.
"""

import pickle
import numpy as np
import pandas as pd
import torch
from collections import defaultdict, Counter
from torch_geometric.data import Data

DATA_DIR = "data"
IPA_PCKL = f"{DATA_DIR}/data.pckl"
OUT = f"{DATA_DIR}/dataset_cutoff_ipa.pckl"

CUTOFF = 6.0  # Angstroms


def get_neighbors_cutoff(structure, cutoff=CUTOFF):
    all_neighbors = structure.get_all_neighbors(cutoff, include_index=True)
    src_idx, dst_idx, edg_vec_list = [], [], []
    for i, neighbors in enumerate(all_neighbors):
        for nbr in neighbors:
            src_idx.append(i)
            dst_idx.append(nbr.index)
            edg_vec_list.append(nbr.coords - structure.sites[i].coords)
    return src_idx, dst_idx, edg_vec_list


def graph_from_entry(entry, label_epsi, label_epsr):
    structure = entry["structure"]

    pos = np.array([site.coords for site in structure.sites])
    pos = torch.tensor(pos, dtype=torch.float32)

    atomic_numbers = torch.tensor(
        [site.specie.Z for site in structure.sites], dtype=torch.long
    )

    self_fea = []
    for site in structure.species:
        group = torch.nn.functional.one_hot(
            torch.tensor(site.group - 1, dtype=torch.int64), num_classes=18
        )
        row = torch.nn.functional.one_hot(
            torch.tensor(site.row - 1, dtype=torch.int64), num_classes=6
        )
        self_fea.append(torch.hstack([group, row]))
    self_fea = torch.vstack(self_fea).to(torch.float32)

    src_idx, dst_idx, edg_vec_list = get_neighbors_cutoff(structure)

    # Sanity check: no isolated nodes
    if len(set(src_idx)) < len(structure.sites):
        raise ValueError("Isolated node detected — consider increasing CUTOFF")

    edg_vec = torch.tensor(np.array(edg_vec_list), dtype=torch.float32)
    edge_lengths = torch.linalg.norm(edg_vec, dim=1)
    edge_vec_norm = edg_vec / edge_lengths.unsqueeze(1)

    src_idx = torch.tensor(src_idx, dtype=torch.long)
    dst_idx = torch.tensor(dst_idx, dtype=torch.long)
    edge_index = torch.vstack([src_idx, dst_idx])

    centers = np.arange(0, 5 + 0.1, 0.1)
    sigma2 = 0.05
    edge_embeds = np.exp(
        -((edge_lengths.numpy()[:, None] - centers) ** 2) / (2 * sigma2)
    )
    edge_embeds /= np.sqrt(2 * np.pi * sigma2)
    edge_embeds = torch.tensor(edge_embeds, dtype=torch.float32)

    edges_from = defaultdict(list)
    for i, src in enumerate(src_idx.numpy()):
        edges_from[src].append(i)

    bond_pairs = []
    for edges in edges_from.values():
        n = len(edges)
        idx = np.arange(n)
        ii, jj = np.meshgrid(idx, idx, indexing="ij")
        mask = ii != jj
        bond_pairs.extend(
            zip(np.array(edges)[ii[mask]], np.array(edges)[jj[mask]])
        )

    threebody_index = torch.tensor(
        np.asarray(bond_pairs, dtype=np.int64).T, dtype=torch.long
    )

    vij = edg_vec[threebody_index[0]]
    vik = edg_vec[threebody_index[1]]
    cos_jik = torch.cosine_similarity(vij, vik)
    angle_attr = torch.stack([cos_jik, 3 * cos_jik ** 2 - 1]).T.to(torch.float32)

    y = torch.hstack([
        torch.tensor(label_epsi, dtype=torch.float32),
        torch.tensor(label_epsr, dtype=torch.float32),
    ])

    return Data(
        pos=pos,
        z=atomic_numbers,
        node_features=self_fea,
        edge_index=edge_index,
        edge_vec=edg_vec,
        edge_vec_norm=edge_vec_norm,
        edge_lengths=edge_lengths,
        edge_attr=edge_embeds,
        threebody_index=threebody_index,
        angle_attr=angle_attr,
        y=y,
        mat_id=entry["mat_id"],
    )


def isotropic_average(entry, key0, key1, key2):
    c0 = entry[key0]
    c1 = entry[key1] if isinstance(entry[key1], np.ndarray) else c0
    c2 = entry[key2] if isinstance(entry[key2], np.ndarray) else c0
    return (c0 + c1 + c2) / 3.0


print("Loading data.pckl ...")
data_df = pd.read_pickle(IPA_PCKL)
data_df = data_df.reset_index()
data_df = data_df.drop(data_df[data_df["ipa_indirect_gap"] < 0.1].index)
data_df = data_df.drop(data_df[data_df["ipa_direct_gap"] > 10].index)
print(f"  {len(data_df)} materials after filtering.")

graphs = []
skipped = []

for i, entry in data_df.iterrows():
    if i % 100 == 0:
        print(f"  [{i}] {len(graphs)} graphs built so far ...")
    try:
        epsi = isotropic_average(entry, "ipa_epsI_0", "ipa_epsI_1", "ipa_epsI_2")
        epsr = isotropic_average(entry, "ipa_epsR_0", "ipa_epsR_1", "ipa_epsR_2")
        graphs.append(graph_from_entry(entry, epsi, epsr))
    except Exception as e:
        print(f"  Skipping {i} ({entry.get('mat_id', '?')}): {e}")
        skipped.append(i)

print(f"\nDone. {len(graphs)} graphs built, {len(skipped)} skipped.")
print(f"Saving to {OUT} ...")
with open(OUT, "wb") as f:
    pickle.dump(graphs, f)
print("Saved.")