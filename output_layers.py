import torch
import torch.nn as nn

class DielectricReadout(nn.Module):
    """
    Readout head matching OptiMate3B exactly.
    Attention pooling:
        a_i  = softmax_i( SiLU( W_pool * h_i ) )    [N, d_node], softmax over nodes per graph
        h_G  = sum_i( a_i * h_i )                   [G, d_node]
    Output MLP:
        out  = mlp_out(h_G)                          [G, 4002]
    Split:
        eps_imag = softplus( out[:, :2001] )
        eps_real = out[:, 2001:]
    Args:
        d_node:     node feature dimension (= hidden_dim in OptiMate3B)
        n_hidden:   hidden dim of output MLP (= spectra_dim in OptiMate3B)
        depth:      number of n_hidden layers in output MLP (= depth in OptiMate3B)
        n_out:      total output size (4002 = 2 x 2001)
        L:          split point between imag and real (2001)
        plain_last: if True, no activation on the final layer of mlp_out.
    """

    def __init__(
        self,
        d_node: int,
        n_hidden: int,
        depth: int = 2,
        n_out: int = 4002,
        L: int = 2001,
        plain_last: bool = True,
    ):
        super().__init__()
        self.L = L

        # pool_mlp: MLP([hidden_dim, hidden_dim], act=silu)
        # = single linear layer followed by SiLU
        self.pool_mlp = nn.Sequential(
            nn.Linear(d_node, d_node),
            nn.SiLU(),
        )

        # mlp_out: MLP([hidden_dim] + [spectra_dim]*depth + [4002], act=silu)
        # = depth linear+SiLU layers followed by a final linear,
        #   with optional SiLU on the last layer controlled by plain_last
        layers = []
        in_dim = d_node
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, n_hidden))
            layers.append(nn.SiLU())
            in_dim = n_hidden
        layers.append(nn.Linear(in_dim, n_out))
        if not plain_last:
            layers.append(nn.SiLU())
        self.mlp_out = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor, node_graph_index: torch.Tensor):
        """
        Args:
            h:                [N, d_node]  node features
            node_graph_index: [N]          graph index for each node
        Returns:
            out:      [G, 4002]
            eps_imag: [G, 2001]
            eps_real: [G, 2001]
        """
        G = int(node_graph_index.max().item()) + 1
        N, d = h.shape

        # --- vector attention pooling ---
        att = self.pool_mlp(h)                                             # [N, d_node]
        # stable segment softmax over nodes within each graph
        att_max = h.new_full((G, d), float('-inf'))
        att_max.index_reduce_(0, node_graph_index, att, reduce='amax', include_self=True)
        att_exp = torch.exp(att - att_max[node_graph_index])               # [N, d_node]
        att_sum = h.new_zeros(G, d).index_add(0, node_graph_index, att_exp)
        att_norm = att_exp / att_sum[node_graph_index].clamp(min=1e-12)    # [N, d_node]

        # weighted sum -> graph-level vector
        h_G = h.new_zeros(G, d).index_add(0, node_graph_index, h * att_norm)  # [G, d_node]

        # --- output MLP ---
        out = self.mlp_out(h_G)                                            # [G, 4002]
        eps_imag = out[:, :self.L]                                         # [G, 2001] 
        eps_real = out[:, self.L:]                                         # [G, 2001]
        #out = torch.cat([eps_imag, eps_real], dim=-1)                      # [G, 4002]

        return out, eps_imag, eps_real