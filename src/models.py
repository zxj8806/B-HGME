import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np
import pandas as pd
import traceback


class SubgraphEncoder(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_vertices, num_subvertices, dropout=0.2, is_training=False):
        super(SubgraphEncoder, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.linear = nn.Linear(num_subvertices * hidden_dim, hidden_dim)
        self.is_training = is_training
        self.dropout = 0.2
        self.num_vertices = num_vertices
        self.num_subvertices = num_subvertices

    def embed(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        return x

    def forward(self, x, edge_index):
        embeddings = self.embed(x, edge_index)
        x = embeddings
        x = x.view(x.shape[0] // self.num_subvertices, x.shape[
            1] * self.num_subvertices)
        x = self.linear(x)
        return x, embeddings


class GraphEncoder(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, dropout=0.2, is_training=False):
        super(GraphEncoder, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.is_training = is_training
        self.dropout = 0.2

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        return x


class InnerProductDecoder(torch.nn.Module):

    def forward(self, z, edge_index, sigmoid=True):
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value


class MultiviewEncoder(torch.nn.Module):
    def __init__(self, SubgraphConcentrations, GraphConcentrations, SubgraphEncoders, GraphEncoders, is_training=False, n_experts=100):
        super(MultiviewEncoder, self).__init__()
        self.concentration_g = SubgraphConcentrations  # encoder for gene level graph
        self.concentration_c = GraphConcentrations  # encoder for cell level graph

        self.encoders_g = SubgraphEncoders  # encoder for gene level graph
        self.encoders_c = GraphEncoders  # encoder for cell level graph
        self.is_training = is_training

    def forward(self, x_c, x_g, edge_index_c, edge_index_g, time_step, n_experts):
        Z_g = []  # Mean of gene embeddings
        Z_c = []  # Mean of cell embeddings
        gene_embeddings = []  # Mean of p_z

        Z_concentration = []  # Variance of combined embeddings
        Z_g_concentration = []  # Variance of gene embeddings
        Z_c_concentration = []  # Variance of cell embeddings
        gene_embeddings_concentration = []  # Variance of p_z

        Z = []

        for t in range(1, n_experts + 1):
            # Mean
            Z_g_, gene_embeddings_ = self.encoders_g[t - 1](x_g, edge_index_g)
            Z_c_ = self.encoders_c[t - 1](x_c, edge_index_c)
            Z_ = torch.cat((Z_c_, Z_g_), dim=1)

            # Variance
            Z_g_var, gene_embeddings_var = self.concentration_g[t - 1](x_g, edge_index_g)
            Z_c_var = self.concentration_c[t - 1](x_c, edge_index_c)
            Z_var = torch.cat((Z_c_var, Z_g_var), dim=1)

            # Store values
            Z_g.append(Z_g_)
            Z_c.append(Z_c_)
            gene_embeddings.append(gene_embeddings_)
            Z.append(Z_)

            Z_g_concentration.append(Z_g_var)
            Z_c_concentration.append(Z_c_var)
            gene_embeddings_concentration.append(gene_embeddings_var)
            Z_concentration.append(Z_var)

        return Z, gene_embeddings, Z_concentration, gene_embeddings_concentration

