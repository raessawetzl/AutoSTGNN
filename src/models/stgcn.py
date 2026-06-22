import torch
import torch.nn as nn


def compute_normalized_laplacian(adjacency):
    """
    STGCN's spectral graph convolution needs the adjacency matrix in a
    specific form: the symmetric normalized Laplacian, scaled to the
    range required by Chebyshev polynomials.

    This only needs to be computed once per dataset, so it lives here
    rather than being recomputed every forward pass.
    """
    num_nodes = adjacency.shape[0]
    identity = torch.eye(num_nodes)

    degree = torch.sum(adjacency, dim=1)
    degree_inv_sqrt = torch.pow(degree, -0.5)
    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
    degree_matrix_inv_sqrt = torch.diag(degree_inv_sqrt)

    # symmetric normalized Laplacian: L = I - D^-1/2 * A * D^-1/2
    laplacian = identity - degree_matrix_inv_sqrt @ adjacency @ degree_matrix_inv_sqrt

    # Chebyshev polynomials are defined on eigenvalues in [-1, 1], so we
    # rescale using the largest eigenvalue (approximated as 2 for a
    # normalized Laplacian, which is the standard STGCN simplification)
    scaled_laplacian = laplacian - identity

    return scaled_laplacian


def chebyshev_polynomials(scaled_laplacian, K):
    """
    Builds the list [T_0, T_1, ..., T_{K-1}] of Chebyshev polynomials
    of the scaled Laplacian. Each T_k is a (num_nodes, num_nodes) matrix.

    These let the model aggregate information from up to K-1 hops away
    in the graph, which is what K_cheb controls in your search space.
    """
    num_nodes = scaled_laplacian.shape[0]
    identity = torch.eye(num_nodes, device=scaled_laplacian.device)

    polynomials = [identity, scaled_laplacian]
    k = 2
    while k < K:
        next_poly = 2 * scaled_laplacian @ polynomials[-1] - polynomials[-2]
        polynomials.append(next_poly)
        k = k + 1

    return polynomials[:K]


class ChebGraphConv(nn.Module):
    """
    A single Chebyshev graph convolution layer. Aggregates each node's
    features with its neighbours' features (up to K hops away) and
    projects them into a new feature space.
    """

    def __init__(self, in_channels, out_channels, K):
        super().__init__()
        self.K = K
        self.weight = nn.Parameter(torch.randn(K, in_channels, out_channels) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x, chebyshev_polys):
        # x: (batch, num_nodes, in_channels)
        out = 0
        for k in range(self.K):
            # propagate features across the graph using the k-th
            # Chebyshev polynomial, then project to out_channels
            propagated = torch.einsum("nm,bmf->bnf", chebyshev_polys[k], x)
            out = out + torch.matmul(propagated, self.weight[k])
        out = out + self.bias
        return out


class TemporalConv(nn.Module):
    """
    A gated 1D convolution over the time dimension, applied per node.
    This captures temporal patterns (e.g. rush hour build-up) the same
    way STGCN's original temporal-spatial-temporal block does, using a
    simplified GLU (gated linear unit) gating mechanism.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        # "same" padding along the time axis only, so the time
        # dimension is preserved after this layer instead of shrinking.
        # Spatial axis (num_nodes) is never padded since kernel width
        # there is always 1.
        padding = (0, (kernel_size - 1) // 2)
        self.conv = nn.Conv2d(
            in_channels, out_channels * 2, kernel_size=(1, kernel_size), padding=padding
        )

    def forward(self, x):
        # x: (batch, channels, num_nodes, time)
        out = self.conv(x)
        out_a, out_b = out.chunk(2, dim=1)
        return out_a * torch.sigmoid(out_b)  # GLU gating


class STGCNBlock(nn.Module):
    """
    One temporal-spatial-temporal block: temporal conv -> graph conv ->
    temporal conv. This is STGCN's core building unit, stacked
    num_layers times in the full model.
    """

    def __init__(self, channels, K_cheb, dropout):
        super().__init__()
        self.temporal1 = TemporalConv(channels, channels)
        self.graph_conv = ChebGraphConv(channels, channels, K_cheb)
        self.temporal2 = TemporalConv(channels, channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x, chebyshev_polys):
        # x: (batch, channels, num_nodes, time)
        x = self.temporal1(x)

        # graph conv operates per timestep, so we move time into the
        # batch dimension temporarily
        batch, channels, num_nodes, time = x.shape
        x_reshaped = x.permute(0, 3, 2, 1).reshape(batch * time, num_nodes, channels)
        x_graph = self.graph_conv(x_reshaped, chebyshev_polys)
        x_graph = self.relu(x_graph)
        x = x_graph.reshape(batch, time, num_nodes, channels).permute(0, 3, 2, 1)

        x = self.temporal2(x)
        x = self.dropout(x)
        return x


class STGCN(nn.Module):
    """
    Full STGCN model. Constructor arguments map directly onto your
    ConfigSpace hyperparameters, so a sampled configuration from BOHB
    can be used to build a model with a single line:

        model = STGCN(num_nodes, hidden_units=cfg['hidden_units'], ...)
    """

    def __init__(
        self,
        num_nodes,
        input_length=12,
        hidden_units=64,
        num_layers=2,
        dropout=0.2,
        K_cheb=3,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.K_cheb = K_cheb

        # project the single input feature (traffic flow) into
        # hidden_units channels
        self.input_proj = nn.Conv2d(1, hidden_units, kernel_size=(1, 1))

        self.blocks = nn.ModuleList(
            [STGCNBlock(hidden_units, K_cheb, dropout) for _ in range(num_layers)]
        )

        # collapses the remaining time dimension and produces a single
        # predicted value per node
        self.output_layer = nn.Conv2d(hidden_units, 1, kernel_size=(1, input_length))

    def forward(self, x, chebyshev_polys):
        # x: (batch, input_length, num_nodes) -> rearrange for conv layers
        x = x.unsqueeze(1).permute(0, 1, 3, 2)  # (batch, 1, num_nodes, time)
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x, chebyshev_polys)

        out = self.output_layer(x)  # (batch, 1, num_nodes, 1)
        out = out.squeeze(1).squeeze(-1)  # (batch, num_nodes)
        return out