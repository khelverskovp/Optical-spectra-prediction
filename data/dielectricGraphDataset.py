import torch
from torch.utils.data import Dataset
from pymatgen.core import Element
import torch.nn.functional as F


class DielectricGraphDataset(Dataset):
    """
    Dataset wrapper for precomputed crystal graph data with dielectric spectra targets.

    Each sample is expected to be a dict-like or object with:
        pos           : float32 [N_atoms, 3]   - Cartesian coordinates (Å)
        z             : int64   [N_atoms]       - atomic numbers
        edge_index    : int64   [2, N_edges]    - COO adjacency (source, dest)
        edge_vec_norm : float32 [N_edges, 3]    - unit bond vectors
        edge_lengths  : float32 [N_edges]       - bond lengths (Å)
        y             : float32 [2·L]           - concatenated [epsI_avg | epsR_avg]
    """

    def __init__(self, raw_data, transform=None):
        """
        Args:
            raw_data  : Iterable of graph samples (list, HDF5 wrapper, PyG Dataset, etc.)
            transform : Optional callable applied to the output dict.
        """
        self.data = raw_data
        self.transform = transform

    # ------------------------------------------------------------------
    # Standard Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        sample = self.data[idx]

        # ── unpack ────────────────────────────────────────────────────
        if isinstance(sample, dict):
            pos           = sample["pos"]           # [N, 3]
            z             = sample["z"]             # [N]
            edge_index    = sample["edge_index"].T    # [N_edges, 2]
            edge_vec_norm = sample["edge_vec_norm"] # [N_edges, 3]
            edge_lengths  = sample["edge_lengths"]  # [N_edges]
            y             = sample["y"]             # [2·L]
        else:                                        # PyG Data / namedtuple
            pos           = sample.pos
            z             = sample.z
            edge_index    = sample.edge_index.T
            edge_vec_norm = sample.edge_vec_norm
            edge_lengths  = sample.edge_lengths
            y             = sample.y

        # ── periodic table one-hot node features ──────────────────────

        groups  = torch.tensor([Element.from_Z(zi.item()).group - 1 for zi in z], dtype=torch.long)
        periods = torch.tensor([Element.from_Z(zi.item()).row   - 1 for zi in z], dtype=torch.long)
        node_features = torch.cat([
            F.one_hot(groups,  num_classes=18).float(),  # [N, 18]
            F.one_hot(periods, num_classes=7).float(),   # [N,  7]
        ], dim=-1)                                       # [N, 25]

        out = {
            "node_features":    node_features,  # [N, 25] float32
            "node_coordinates": pos,
            "edge_index":       edge_index,
            "edge_vectors":     edge_vec_norm,
            "edge_lengths":     edge_lengths,
            "label":            y,
        }

        if self.transform is not None:
            out = self.transform(out)

        return out