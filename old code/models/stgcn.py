import torch
import torch.nn as nn # neural network layers 
import torch.nn.functional as F # activation functions 

# temporal relationships
class TemporalConv(nn.Module):
    """
    1D convolution along the time axis.
    uses a gated activation — splits channels in half,
    one half goes through tanh, the other through sigmoid,
    then they are multiplied together.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        '''
        parameters: 
            in_channels: input features per time step
            out_channels: output features
            kernel_size: how many time steps it looks at
        '''

        super(TemporalConv, self).__init__()
        # 2D convolution moving along time only, not sensors 
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * 2,
            kernel_size=(kernel_size, 1),
            padding=(kernel_size // 2, 0)
        )
        self.out_channels = out_channels #output size 

    # forward pass
    def forward(self, x):
        # x: (batch, in_channels, time, sensors)
        x = self.conv(x) # applies temporal convolution
        # split channels in half 
        x1 = x[:, :self.out_channels, :, :]
        x2 = x[:, self.out_channels:, :, :]
        return torch.tanh(x1) * torch.sigmoid(x2) #gating 

# spatial relationships
class GraphConv(nn.Module):
    """
    chebyshev spectral graph convolution.
    Lk: precomputed Chebyshev polynomials (graph hops), shape (K, num_sensors, num_sensors)
    """
    def __init__(self, in_channels, out_channels, K):
        super(GraphConv, self).__init__()
        self.K = K
        self.linear = nn.Linear(K * in_channels, out_channels)

    def forward(self, x, Lk):
        # x:  (batch, in_channels, time, sensors)
        # Lk: (K, sensors, sensors)
        batch, channels, time, sensors = x.shape

        # rearrange to (batch * time, sensors, channels)
        x = x.permute(0, 2, 3, 1).reshape(batch * time, sensors, channels)

        # apply graph convolution
        chunks = []
        for k in range(self.K):
            Lk_single = Lk[k]  # (sensors, sensors)
            chunk = torch.matmul(Lk_single, x)  # (batch*time, sensors, channels)
            chunks.append(chunk)

        x = torch.cat(chunks, dim=-1)  # (batch*time, sensors, K*channels)
        x = self.linear(x)             # (batch*time, sensors, out_channels)

        # back to (batch, out_channels, time, sensors)
        x = x.reshape(batch, time, sensors, -1).permute(0, 3, 1, 2)
        return F.relu(x)


class STGCNBlock(nn.Module):
    """
    One STGCN block: temporal conv -> graph conv -> temporal conv -> layer norm.
    """
    def __init__(self, in_channels, hidden_units, K, dropout):
        super(STGCNBlock, self).__init__()
        self.temp_conv1 = TemporalConv(in_channels, hidden_units)
        self.graph_conv = GraphConv(hidden_units, hidden_units, K)
        self.temp_conv2 = TemporalConv(hidden_units, hidden_units)
        self.norm = nn.LayerNorm(hidden_units)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, Lk):
        # x: (batch, in_channels, time, sensors)
        x = self.temp_conv1(x)
        x = self.graph_conv(x, Lk)
        x = self.temp_conv2(x)
        # norm expects (batch, time, sensors, channels)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.dropout(x)
        # back to (batch, channels, time, sensors)
        x = x.permute(0, 3, 1, 2)
        return x


class STGCN(nn.Module):
    def __init__(self, num_sensors, num_layers, hidden_units, K, dropout, horizon=12):
        super(STGCN, self).__init__()

        self.blocks = nn.ModuleList()
        # first block takes 1 input channel (flow only)
        self.blocks.append(STGCNBlock(1, hidden_units, K, dropout))
        # remaining blocks
        for _ in range(num_layers - 1):
            self.blocks.append(STGCNBlock(hidden_units, hidden_units, K, dropout))

        self.output_layer = nn.Linear(hidden_units, horizon)

    def forward(self, x, Lk):
        # x: (batch, window, sensors, 1) — from your dataloader
        # rearrange to (batch, channels, time, sensors) for conv layers
        x = x.permute(0, 3, 1, 2)

        for block in self.blocks:
            x = block(x, Lk)

        # x: (batch, channels, time, sensors)
        # take last timestep
        x = x[:, :, -1, :]          # (batch, channels, sensors)
        x = x.permute(0, 2, 1)      # (batch, sensors, channels)
        x = self.output_layer(x)    # (batch, sensors, horizon)
        x = x.permute(0, 2, 1).unsqueeze(-1)  # (batch, horizon, sensors, 1)
        return x