


from torch import nn

import torch
from torch_geometric.nn import GCNConv, GATConv
import torch_geometric.nn as gnn


class GNN(torch.nn.Module):
    def __init__(self, n_in_dim, hidden_dim, output_dim, num_layers):
        super(GNN, self).__init__()
        self.n_in_dim = n_in_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        self.convs = torch.nn.ModuleList()
        self.attentions = torch.nn.ModuleList()
        self.convs.append(GCNConv(n_in_dim, hidden_dim))
        self.attentions.append(GATConv(hidden_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.attentions.append(GATConv(hidden_dim * 2, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, output_dim))
        self.pool = gnn.global_mean_pool

    def forward(self, x, edge_index):
        # Perform the forward pass of the GNN
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = torch.relu(x)
            x = self.attentions[i](x, edge_index)
            x = torch.cat([x, x], dim=-1)
            if i != self.num_layers - 1:
                x = self.pool(x, edge_index)
        return x


class GATConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.att = torch.nn.Linear(2 * out_channels, 1)

    def forward(self, x, edge_index):
        # Compute attention coefficients.
        row, col = edge_index
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Compute attention coefficients.
        x_i = self.lin(x[row])
        x_j = self.lin(x[col])
        alpha = (x_i * x_j).sum(dim=-1, keepdim=True)
        alpha = F.leaky_relu(alpha, 0.2)

        # Sample attention coefficients stochastically.
        alpha = softmax(alpha, edge_index[0], num_nodes=x.size(0))

        # Sample attention coefficients stochastically.
        alpha = dropout(alpha, p=0.6, training=self.training)

        # Linearly transform node feature vectors and compute attention-head outputs.
        out = torch.zeros(x.size(0), self.out_channels).to(x.device)
        out = out.scatter_add_(dim=0, index=row.view(-1, 1).expand(-1, self.out_channels), src=(x_j * alpha).view(-1, self.out_channels))
        return out