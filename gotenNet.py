import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from output_layers import DielectricReadout

# -----------------------------
# small utilities
# -----------------------------
def scatter_add(src, index, dim=0, dim_size=None):
    if dim_size is None:
        dim_size = int(index.max().item()) + 1 if index.numel() > 0 else 0
    out = src.new_zeros((dim_size,) + src.shape[1:])
    out.index_add_(0, index, src)
    return out


def scatter_max(src, index, dim_size):
    """Per-segment maximum, returns [dim_size, ...] filled with -inf where empty."""
    out = src.new_full((dim_size,) + src.shape[1:], fill_value=float('-inf'))
    # index_reduce_ with 'amax' available in PyTorch >= 2.0
    out.index_reduce_(0, index, src, reduce='amax', include_self=True)
    return out


def segment_softmax(scores, dst, num_nodes):
    if scores.dim() == 1:
        max_vals = scatter_max(scores, dst, num_nodes).clamp(min=-1e9)  # [N]
        scores = scores - max_vals[dst]   # subtract per-segment max, not global max
        exp = torch.exp(scores)
        denom = scatter_add(exp, dst, dim_size=num_nodes).clamp_min(1e-12)
        return exp / denom[dst]

    if scores.dim() == 2:
        max_vals = scatter_max(scores, dst, num_nodes).clamp(min=-1e9)  # [N, H]
        scores = scores - max_vals[dst]   # [E, H]
        exp = torch.exp(scores)
        denom = scatter_add(exp, dst, dim_size=num_nodes).clamp_min(1e-12)
        return exp / denom[dst]

    raise ValueError(f"segment_softmax expects scores [E] or [E,H], got {tuple(scores.shape)}")

def vector_rejection(x, r, eps=1e-8):
    """
    Remove the component of x parallel to r.

    Args:
        x: [E, m, d]   steerable features
        r: [E, m]      edge representation for the same degree
        eps: float     numerical stability constant

    Returns:
        x_rej: [E, m, d]
    """
    r = r.unsqueeze(-1)                              # [E, m, 1]
    dot = (x * r).sum(dim=1, keepdim=True)          # [E, 1, d]
    #denom = (r * r).sum(dim=1, keepdim=True) + eps  # [E, 1, 1]
    #proj = (dot / denom) * r                        # [E, m, d]
    proj = dot * r
    return x - proj


# -----------------------------
# real spherical harmonics (Lmax <= 2)
# -----------------------------

def sh_l2(u):
    """
    Degree-2 real spherical harmonics basis, output dim 5.
    u: [E, 3] with components x,y,z (normalized edge vector)
    returns: [E, 5]
    """
    x, y, z = u[:, 0], u[:,1], u[:,2]
    
    s3 = math.sqrt(3.0)

    Y1 = s3 * x * y
    Y2 = s3 * y * z
    Y3 = 0.5 * (3.0 * z * z - 1.0)
    Y4 = s3 * z * x
    Y5 = 0.5 * s3 * (x * x - y * y)

    return torch.stack([Y1, Y2,Y3,Y4,Y5], dim=-1)

def edge_tensor_repr(r, u, Lmax):
    """
    Build edge tensor representations r~_ij^(l) for l=0..Lmax:
      l=0: distance scalar [E, 1]
      l=1: direction  [E, 3]
      l=2: SH basis   [E, 5]
    Returns list indexed by l, where tensors are [E, 2l+1].
    """
    out = []
    out.append(r.unsqueeze(-1))  # l=0
    if Lmax >= 1:
        out.append(u)    # l=1
    if Lmax >= 2:
        out.append(sh_l2(u))     # l=2
    if Lmax > 2:
        raise NotImplementedError("This implementation only supports Lmax <= 2.")
    return out

# -----------------------------
# RBF + cutoff
# ----------------------------
class RBFExpansion(nn.Module):
    def __init__(self, num_rbf=50, r_cut=5.0):
        super().__init__()
        self.r_cut = r_cut
        self.num_rbf = num_rbf
        self.alpha = 5.0 / r_cut
        means, betas = self._initial_params()
        self.register_buffer("means", means)
        self.register_buffer("betas", betas)

    def _initial_params(self):
        start_value = torch.exp(torch.scalar_tensor(-self.r_cut))
        means = torch.linspace(start_value, 1, self.num_rbf)
        betas = torch.tensor([(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf)
        return means, betas

    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        return torch.exp(-self.betas * (torch.exp(self.alpha * (-dist)) - self.means) ** 2)

# cosine cutoff function 
def cosine_cutoff(r, r_cut):
    cutoff = 0.5 * (torch.cos(math.pi * r / r_cut) + 1.0)
    cutoff = cutoff * (r < r_cut).float()
    return cutoff.unsqueeze(-1)  # [E,1]


# -----------------------------
# MLP and Linear Layer util
# -----------------------------
def mlp(
    d_in,
    d_hidden,
    d_out,
    dropout,
    weight_init=nn.init.xavier_uniform_,
    bias_init=nn.init.zeros_,
):
    l1 = nn.Linear(d_in, d_hidden)
    l2 = nn.Linear(d_hidden, d_out)

    weight_init(l1.weight)
    bias_init(l1.bias)

    weight_init(l2.weight)
    bias_init(l2.bias)

    return nn.Sequential(
        l1,
        nn.SiLU(),
        nn.Dropout(dropout),
        l2,
    )

def linear(in_features, out_features, bias=True):
    layer = nn.Linear(in_features, out_features, bias=bias)
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer

# -----------------------------
# core modules: Structure Embedding, GATA, EQFF, HTR
# -----------------------------
class StructureEmbedding(nn.Module):
    """
    Implements Eq. (1)-(3) in batched form.

    Returns:
      h: [N, dne]
      t: [E, ded]
      rtilde_list: list over l=0..Lmax
      cutoff: [E,1]
    """

    def __init__(self, dne, ded, num_rbfs, cutoff, Lmax):
        super().__init__()
        self.dne = dne
        self.ded = ded
        self.Lmax = Lmax
        self.cutoff = cutoff

        self.rbf = RBFExpansion(num_rbf=num_rbfs, r_cut=cutoff)

        self.A_na  = nn.Linear(25, dne)
        self.A_nbr = nn.Linear(25, dne)

        self.W_ndp = linear(num_rbfs, dne)
        self.W_nrd = linear(2 * dne, dne)
        self.W_nru = linear(dne, dne)
        self.W_erp = linear(num_rbfs, ded)

        self.ln = nn.LayerNorm(dne)

    def forward(self, z, edge, r, u):
        i = edge[:, 0]
        j = edge[:, 1]
        N = z.size(0)

        rtilde = edge_tensor_repr(r, u, self.Lmax)
        rbf = self.rbf(r)
        c = cosine_cutoff(r, self.cutoff)

        # Eq (1)
        nbr = F.silu(self.A_nbr(z))             # [N, dne]
        gate = self.W_ndp(rbf) * c              # [E, dne]
        msg = nbr[j] * gate                     # [E, dne]
        m = scatter_add(msg, i, dim=0, dim_size=N)  # [N, dne]

        # Eq (2)
        na = F.silu(self.A_na(z))               # [N, dne]
        h = torch.cat([na, m], dim=-1)          # [N, 2dne]
        h = self.W_nrd(h)
        h = self.ln(h)
        h = F.silu(h)
        h = self.W_nru(h)                       # [N, dne]

        # Eq (3)
        t = (h[i] + h[j]) * self.W_erp(rbf)    # [E, ded]

        return h, t, rtilde, c
    
class GATA(nn.Module):
    """
    Geometry-aware tensor attention (GATA), with multi-head attention.

    Inputs:
      h:      [N, dne]
      t:      [E, ded]          (edge scalar features, updated by HTR first)
      X_list: list l=1..Lmax of [N, 2l+1, dne]
      edge:   [E, 2] with (i,j) meaning message j -> i
      rtilde: list l=0..Lmax; use rtilde[l]: [E, 2l+1] for l>=1
      c:      [E, 1] cosine cutoff

    Outputs:
      h_out: updated [N, dne]
      t_out: updated [E, ded] (after HTR; used in GATA)
      X_out_list: updated steerable features list
    """

    def __init__(self, dne, ded, Lmax, htr, num_heads=8, dropout=0.0, sigma_k=F.silu):
        super().__init__()
        self.dne = dne
        self.ded = ded
        self.Lmax = Lmax
        self.htr = htr
        self.num_heads = num_heads
        self.sigma_k = sigma_k
        self.dropout = dropout

        self.d_head = dne // num_heads
        self.S = 1 + 2 * Lmax
        self.coeff_dim = self.S * dne

        self.coeff_head = self.coeff_dim // num_heads  # per-head coefficient width

        # LN on node scalars
        self.ln_h = nn.LayerNorm(dne)

        # Attention projections (produce dne, then reshape into heads)
        self.W_q = linear(dne, dne)
        self.W_k = linear(dne, dne)

        # Geometric encoding in attention: sigma(t_ij W_re + b_re)
        # produce dne, then reshape into heads
        self.W_re = linear(ded, dne)

        # Value MLP for attention: v_j = gamma_v(h_j) in R^{S*dne}
        self.gamma_v = mlp(dne, dne, self.coeff_dim, dropout=dropout)

        # (t_ij W_rs + b_rs) * gamma_s(h_j) * cutoff
        self.W_rs = linear(ded, self.coeff_dim)
        self.gamma_s = mlp(dne, dne, self.coeff_dim, dropout=dropout)

    def forward(self, h, t, X_list, edge, rtilde, c, last_layer=False):
        i = edge[:, 0]
        j = edge[:, 1]
        N = h.size(0)
        E = edge.size(0)
        H = self.num_heads

        # 1) LN on node scalars
        #h_ln = self.ln_h(h)            # [N, dne]

        # 2) Multi-head attention coefficients alpha_ij^h with head-splitting of dne
        q_i = self.W_q(h[i]).view(E, H, self.d_head)  # [E,H,d_head]
        k_j = self.W_k(h[j]).view(E, H, self.d_head)  # [E,H,d_head]
        geom = self.sigma_k(self.W_re(t)).view(E, H, self.d_head)  # [E,H,d_head]

        degree = torch.bincount(i, minlength=N)  # [N]
        n_edges_per_edge = degree[i].float()             # [E]

        alpha = (q_i * (k_j * geom)).sum(dim=-1)         # [E,H]
        attn = segment_softmax(alpha, i, N)              # [E,H]

        # Apply normalization
        norm = (n_edges_per_edge ** 0.5) / (self.dne ** 0.5)  # [E]
        attn = attn * norm.unsqueeze(-1)                  # [E, H]

        attn = attn.unsqueeze(-1)                         # [E, H, 1]

        # Apply random dropout
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        # 3) Values per head: split coeff_dim across heads (concat later)
        v_j = self.gamma_v(h[j]).view(E, H, self.coeff_head)  # [E,H,coeff_head]
        sea = attn * v_j                                         # [E,H,coeff_head]

        # 4) term per head (same head-splitting of coeff_dim)
        t_proj = self.W_rs(t).view(E, H, self.coeff_head)        # [E,H,coeff_head]
        s_j = self.gamma_s(h[j]).view(E, H, self.coeff_head)  # [E,H,coeff_head]
        spatial = t_proj * s_j * c.unsqueeze(1)                  # [E,H,coeff_head]

        # 5) Concatenate heads back -> [E, coeff_dim]
        coeff = (sea + spatial).reshape(E, self.coeff_dim)       # [E, S*dne]
        coeff = coeff.view(E, self.S, self.dne)                  # [E, S, dne]

        # 6) Split into [c^s, {c^d(l)}, {c^t(l)}]
        c_s = coeff[:, 0, :]                                      # [E, dne]
        c_d = coeff[:, 1:1 + self.Lmax, :]                        # [E, Lmax, dne]
        c_t = coeff[:, 1 + self.Lmax:1 + 2*self.Lmax, :]          # [E, Lmax, dne]

        # 7) Eq. 8
        dh = scatter_add(c_s, i, dim=0, dim_size=N)               # [N, dne]
        h_out = h + dh

        # 8) Eq. 8
        X_out_list = []
        for l in range(1, self.Lmax + 1):
            X_l = X_list[l - 1]                                   # [N, 2l+1, dne]
            rl = rtilde[l]                                        # [E, 2l+1]

            o_d_l = c_d[:, l - 1, :]                               # [E, dne]
            o_t_l = c_t[:, l - 1, :]                               # [E, dne]

            msg_dir = rl.unsqueeze(-1) * o_d_l.unsqueeze(1)        # [E, 2l+1, dne]
            msg_tens = X_l[j] * o_t_l.unsqueeze(1)                 # [E, 2l+1, dne]

            #dX_l = scatter_add(msg_dir + msg_tens, i, dim=0, dim_size=N)
            dX_l = scatter_add(msg_dir, i, dim=0, dim_size=N)
            dX_l += scatter_add(msg_tens, i, dim=0, dim_size=N)
            X_out_list.append(X_l + dX_l)
        
        # 9) Update edge features using HTR 
        if not last_layer:
            t = self.htr(t, X_out_list, edge, rtilde)  # [E, ded]

        return h_out, t, X_out_list

class EQFF(nn.Module):
    """
    Equivariant feed-forward (EQFF) block.
    """

    def __init__(self, dne: int, dropout: float, eps: float = 1e-8):
        super().__init__()

        # Linear proj of steerable features
        self.W_vu = linear(dne, dne, bias=False)

        # network producing vectors (m1, m2)
        self.gamma_m = mlp(2 * dne, 2 * dne, 2 * dne, dropout)

        # Numerical stability constant used in norm computation
        self.eps = eps

    def forward(self, h, X_list):
        """
        Args:
            h:       [N, dne] scalar node features
            X_list:  list of tensors X^(l), each [N, 2l+1, dne]

        Returns:
            Updated (h, X_list) with residual EQFF updates applied.
        """

        # Project steerable features once using W_vu
        X_proj_list = [self.W_vu(X) for X in X_list]

        # Compute ||X W_vu||_2 across all steerable channels
        # Sum squared magnitudes over the (2l+1) dimension
        norm_sq = 0.0
        for Xp in X_proj_list:
            norm_sq = norm_sq + (Xp * Xp).sum(dim=1)  # [N, dne]

        xnorm = torch.sqrt(norm_sq + self.eps)        # [N, dne]

        # Form vector and compute coefficients
        ctx = torch.cat([h, xnorm], dim=-1)           # [N, 2*dne]
        m = self.gamma_m(ctx)                         # [N, 2*dne]
        m1, m2 = m.chunk(2, dim=-1)                   # each [N, dne]

        # Scalar residual update
        h_out = h + m1

        # Steerable residual updates per degree
        X_out_list = []
        for X, Xp in zip(X_list, X_proj_list):
            delta = m2.unsqueeze(1) * Xp              # broadcast over (2l+1)
            X_out_list.append(X + delta)

        return h_out, X_out_list


class HTR(nn.Module):
    """
    Hierarchical Tensor Refinement, Eq. (10)-(12)
    Updates edge scalars t_ij based on products of high-degree steerable features.
    """
    def __init__(self, dne, ded, dxpd, Lmax, dropout):
        super().__init__()
        self.Lmax = Lmax
        self.W_vq = linear(dne, dxpd, bias=False)  # shared across degrees (Eq.10)
        self.W_vk = nn.ModuleList([linear(dne, dxpd, bias=False) for _ in range(Lmax)])  # degree-specific (Eq.10)

        self.gamma_w = mlp(dxpd, dxpd, ded, dropout=dropout)  # γ_w (Eq.12)
        self.gamma_t = mlp(ded, ded, ded, dropout=dropout)    # γ_t (Eq.12)

    def forward(self, t, X_list, edge, rtilde):
        i = edge[:, 0]
        j = edge[:, 1]

        X_cat = torch.cat(X_list, dim=1)                                        # [N, sum(2l+1), dne]
        Q_cat = self.W_vq(X_cat)[i]                                             # [E, sum(2l+1), dxpd]
        K_cat = torch.cat([self.W_vk[l](X_list[l]) for l in range(self.Lmax)], dim=1)[j]  # [E, sum(2l+1), dxpd]

        Q_cat = torch.cat([
            vector_rejection(Q_cat[:, :3, :], rtilde[1]),
            vector_rejection(Q_cat[:, 3:, :], rtilde[2]),
        ], dim=1)

        K_cat = torch.cat([
            vector_rejection(K_cat[:, :3, :], -rtilde[1]),
            vector_rejection(K_cat[:, 3:, :], -rtilde[2]),
        ], dim=1)

        w = (Q_cat * K_cat).sum(dim=1)                                          # [E, dxpd]

        dt = self.gamma_w(w) * self.gamma_t(t)
        return t + dt
# -----------------------------
# Full GotenNet
# -----------------------------
@dataclass
class GotenNetConfig:
    dne: int = 256
    ded: int = 256
    dxpd: int = 256
    num_layers: int = 4
    num_rbfs: int = 64
    cutoff: float = 5.0
    Lmax: int = 2
    dropout: float = 0.1
    num_heads: int = 8
    # readout options
    readout_depth: int = 2
    readout_n_hidden: int = 256
    plain_last: bool = True       


class GotenNet(nn.Module):
    """
    Structure Embedding -> (GATA+EQFF)xN -> DielectricReadout.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.embed = StructureEmbedding(
            dne=cfg.dne,
            ded=cfg.ded,
            num_rbfs=cfg.num_rbfs,
            cutoff=cfg.cutoff,
            Lmax=cfg.Lmax,
        )

        self.gata = nn.ModuleList([
            GATA(
                dne=cfg.dne,
                ded=cfg.ded,
                Lmax=cfg.Lmax,
                htr=HTR(cfg.dne, cfg.ded, cfg.dxpd, cfg.Lmax, cfg.dropout),
                num_heads=cfg.num_heads,
                dropout=cfg.dropout,
            )
            for _ in range(cfg.num_layers)
        ])

        self.eqff = nn.ModuleList([
            EQFF(cfg.dne, cfg.dropout) for _ in range(cfg.num_layers)
        ])

        self.head = DielectricReadout(
            d_node=cfg.dne,
            n_hidden=cfg.readout_n_hidden,
            depth=cfg.readout_depth,
            plain_last=cfg.plain_last,
        )

    def forward(self, batch):
        """
        Expects batch dict with:
        node_features:    [N, 25]  float32  (group + period one-hot)
        node_coordinates: [N, 3]   float32
        edge_index:       [E, 2]   int64
        edge_lengths:     [E]      float32
        edge_vectors:     [E, 3]   float32
        node_graph_index: [N]      int64
        """
        z    = batch["node_features"]       # [N]
        edge = batch["edge_index"]           # [2, E]
        r    = batch["edge_lengths"]         # [E]
        u    = batch["edge_vectors"]         # [E, 3]
        gidx = batch["node_graph_index"]     # [N]

        N    = z.size(0)
        dne  = self.cfg.dne
        Lmax = self.cfg.Lmax

        h, t, rtilde, c = self.embed(z, edge, r, u)

        X_list = [h.new_zeros((N, 2 * l + 1, dne)) for l in range(1, Lmax + 1)]

        for layer in range(self.cfg.num_layers):
            h, t, X_list = self.gata[layer](
                h, t, X_list, edge, rtilde, c,
                last_layer=(layer == self.cfg.num_layers - 1),
            )
            h, X_list = self.eqff[layer](h, X_list)

        out, eps_imag, eps_real = self.head(h, gidx)
        return out, eps_imag, eps_real