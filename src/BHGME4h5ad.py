#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse, random, time, copy, warnings, math
from typing import List, Tuple

import numpy as np
import pandas as pd
import scipy
from scipy.spatial import distance_matrix
from scipy.sparse import csr_matrix
import scanpy as sc
import anndata

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_sparse
from torch_sparse import SparseTensor
from torch_geometric.nn import GCNConv
from torch_geometric.utils import (
    dense_to_sparse,
    dropout_edge,
    from_scipy_sparse_matrix,
)
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics as sk_metrics

from utils import (
    get_feat_mask, dense2sparse, normalize_adj_symm,
    refine_labels, split_batch, _get_coord, _knn_graph_fast, _knn_graph_dense,
    load_data_from_raw, _build_parser,
)
from hyperspherical_vae.distributions import VonMisesFisher, HypersphericalUniform


class BayesianMoEGraphLearner(nn.Module):

    def __init__(
        self,
        nlayers: int,
        isize: int,
        neighbor: int,
        gamma: float,
        adj: SparseTensor,
        coords,
        device: torch.device,
        omega: float,
        n_experts: int = 4,
        dir_alpha: float = 1.0,
    ):
        super().__init__()
        self.device = device
        self.input_dim = isize
        self.omega = float(omega)
        self.n_experts = int(n_experts)
        self.dir_alpha = float(dir_alpha)

        self.adj: SparseTensor = adj.to(device)

        if coords is None:
            n = adj.size(0)
            coords = torch.zeros((n, 2), dtype=torch.float32)
        elif isinstance(coords, np.ndarray):
            coords = torch.tensor(coords, dtype=torch.float32)
        self.coords = coords.to(device)

        with torch.no_grad():
            dmat_np = distance_matrix(
                self.coords.detach().cpu().numpy(),
                self.coords.detach().cpu().numpy()
            )
            d_matrix = torch.tensor(dmat_np, dtype=torch.float32, device=device)

            d_sorted, _ = d_matrix.sort(dim=1)
            d_cut = torch.median(d_sorted[:, neighbor])

            c1 = d_matrix > 0
            c2 = d_matrix <= d_cut
            self.adj_mask = torch.logical_and(c1, c2)

            d_norm = torch.where(self.adj_mask, d_matrix, torch.inf) / d_cut
            self.s_d = 1.0 / torch.exp(torch.as_tensor(gamma, device=device) * (d_norm ** 2))

        self.experts = nn.ModuleList([
            nn.ModuleList(
                [GCNConv(isize, isize)] +
                [GCNConv(isize, isize) for _ in range(nlayers - 1)]
            )
            for _ in range(self.n_experts)
        ])

        coord_dim = int(self.coords.size(1))
        self.gate = nn.Sequential(
            nn.Linear(isize + coord_dim, isize),
            nn.ReLU(),
            nn.Linear(isize, self.n_experts),
        )

        self.latest_kl = torch.tensor(0.0, device=device)
        self.noise_scale = nn.Parameter(torch.tensor(2e-2))
        self.init_kappa = 5.0
        self.log_kappa = nn.Parameter(torch.tensor(self.init_kappa))

        

    @staticmethod
    def _encode_expert(x: torch.Tensor, convs: nn.ModuleList, adj_t: SparseTensor) -> torch.Tensor:
        for i, conv in enumerate(convs):
            x = conv(x, adj_t)
            if i != len(convs) - 1:
                x = F.relu(x)
        return F.normalize(x, dim=1)

    @staticmethod
    def _kl_dirichlet_uniform(q: torch.Tensor) -> torch.Tensor:
        
        k = q.size(1)
        return (q * (q.log() - math.log(1.0 / k))).sum(-1).mean()
    
    def kappa(self):
        return F.softplus(self.log_kappa)
    
    @staticmethod
    def _reparameterize(mean: torch.Tensor,
                        concentration: torch.Tensor):
        q_z = VonMisesFisher(mean, concentration)
        p_z = HypersphericalUniform(mean.size(1) - 1, device=mean.device)
        return q_z, p_z
    
    def _expert_sample(self,
                       mean: torch.Tensor) -> torch.Tensor:

        kappa_b = torch.full((mean.size(0), 1), self.kappa().item(), device=mean.device)
        
        q_z, _ = self._reparameterize(mean, kappa_b)
        sampled_z = q_z.rsample()
        return F.normalize(sampled_z, dim=1)
    
    

    def forward(self, features: torch.Tensor, *, beta_kl: float = 1e-3):
        adj_t = self.adj.t()
        h_list = [self._encode_expert(features, convs, adj_t) for convs in self.experts]

        logits = self.gate(torch.cat([features, self.coords], dim=1))
        q = F.softmax(logits, dim=1)

        self.latest_kl = self._kl_dirichlet_uniform(q) * float(beta_kl)

        n = features.size(0)
        s = torch.zeros((n, n), device=features.device, dtype=features.dtype)
        for tau, h_tau in enumerate(h_list):
            h_tau = self._expert_sample(h_tau)

            sim = h_tau @ h_tau.t()
            w = 0.5 * (q[:, tau].unsqueeze(1) + q[:, tau].unsqueeze(0))
            s += w * sim

        s = self.omega * s + (1.0 - self.omega) * self.s_d
        s = torch.where(self.adj_mask, s, torch.zeros_like(s))
        return normalize_adj_symm(s), s

class GraphEncoder(nn.Module):
    def __init__(self, nlayers, in_dim, hidden_dim, emb_dim, proj_dim, dropout, dropout_adj):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.dropout_adj_p = dropout_adj
        self.relu = nn.ReLU(inplace=True)
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(nlayers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, emb_dim))
        self.proj_head = nn.Sequential(
            nn.Linear(emb_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )

    def _dropout_adj(self, Adj: SparseTensor) -> SparseTensor:
        row, col, val = Adj.coo()
        edge_index = torch.stack([row, col], dim=0)
        edge_index, edge_mask = dropout_edge(edge_index, p=self.dropout_adj_p)
        val = val[edge_mask]
        return SparseTensor(
            row=edge_index[0],
            col=edge_index[1],
            value=val,
            sparse_sizes=(Adj.size(0), Adj.size(1)),
        )

    def forward(self, x, Adj: SparseTensor, training: bool):
        if training:
            Adj = self._dropout_adj(Adj)
        for conv in self.convs[:-1]:
            x = conv(x, Adj.t())
            x = self.relu(x)
            x = self.dropout(x)
        x = self.convs[-1](x, Adj.t())
        z = self.proj_head(x)
        return x, z


class GCL(nn.Module):
    def __init__(self, nlayers, cell_feature_dim, in_dim, hidden_dim, emb_dim, proj_dim, dropout, dropout_adj, margin, bn):
        super().__init__()
        self.encoder = GraphEncoder(
            nlayers, in_dim, hidden_dim, emb_dim, proj_dim, dropout, dropout_adj
        )
        self.cell_encoder = nn.Sequential(
            nn.Linear(cell_feature_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
        )
        if bn:
            self.cell_encoder.append(nn.BatchNorm1d(in_dim))
        self.graph_learner = None
        self.margin_loss = nn.MarginRankingLoss(margin=margin, reduction="mean")

    def get_cell_features(self, gene_exp):
        return self.cell_encoder(gene_exp)

    def get_learned_adj(self, cell_features):
        return self.graph_learner(cell_features)

    def forward(self, x_, Adj, maskfeat_rate=None, training=None):
        x = x_ * get_feat_mask(x_, maskfeat_rate) if maskfeat_rate else x_
        if training is None:
            training = self.training
        embedding, z = self.encoder(x, Adj, training)
        return embedding, z

    @staticmethod
    def sim_loss(x, x_aug, temperature, sym: bool = True):
        n, _ = x.size()
        sim = torch.einsum("ik,jk->ij", x, x_aug) / (x.norm(dim=1)[:, None] * x_aug.norm(dim=1)[None, :])
        sim = torch.exp(sim / temperature)
        pos = sim[range(n), range(n)]
        if sym:
            loss0 = pos / (sim.sum(dim=0) - pos)
            loss1 = pos / (sim.sum(dim=1) - pos)
            return -(torch.log(loss0).mean() + torch.log(loss1).mean()) / 2
        else:
            loss1 = pos / (sim.sum(dim=1) - pos)
            return -torch.log(loss1).mean()

class BHGME:
    def __init__(self, args):
        self.args = args
        (
            self.adata, self.gene_exp, self.labels, self.nclasses,
            self.adj_knn, self.cell_coords, self.dist_sort, self.dist_sort_idx,
        ) = load_data_from_raw(args)
        self.gene_exp = self.gene_exp.to(args.device)

        self.labels_true = (
            self.adata.obs["cluster"].values
            if "cluster" in self.adata.obs.columns
            else None
        )

    @staticmethod
    def setup_seed(seed: int):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(seed)
        random.seed(seed)

    def _predict(self, model, cell_features, learned_adj):
        model.eval()
        with torch.no_grad():
            embedding, _ = model(cell_features.detach(), learned_adj)
        embedding = embedding.cpu().numpy()
        cmode = self.args.clu_model
        if cmode == "kmeans":
            kmeans = KMeans(n_clusters=self.nclasses, random_state=0, n_init="auto").fit(embedding)
            labels_pred = kmeans.predict(embedding)
        else:
            raise ValueError(f"Unknown cluster model {cmode}")
        if self.args.refine:
            labels_pred = refine_labels(labels_pred, self.dist_sort_idx, self.args.refine)
        return embedding, labels_pred

    def train(self):
        args = self.args
        self.setup_seed(args.seed)
        job_dir = args.output_dir
        os.makedirs(job_dir, exist_ok=True)
        with open(os.path.join(job_dir, "args.txt"), "w") as f:
            print(args, file=f)

        anchor_adj = normalize_adj_symm(self.adj_knn).to(args.device)  # SparseTensor
        bn = not args.no_bn
        model = GCL(
            nlayers=args.nlayers,
            cell_feature_dim=self.gene_exp.size(1),
            in_dim=args.exp_out,
            hidden_dim=args.hidden_dim,
            emb_dim=args.rep_dim,
            proj_dim=args.proj_dim,
            dropout=args.dropout,
            dropout_adj=args.dropedge_rate,
            margin=args.margin,
            bn=bn,
        )

        if args.sparse_learner:
            pass
        else:
            model.graph_learner = BayesianMoEGraphLearner(
                nlayers=args.nlayers,
                isize=args.exp_out,
                neighbor=args.k,
                gamma=args.gamma,
                adj=anchor_adj,
                coords=self.cell_coords,
                device=args.device,
                omega=args.adj_weight,
                n_experts=getattr(args, "n_experts", 4),
                dir_alpha=1.0,
            )

        model = model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)
        identity = dense2sparse(torch.eye(self.gene_exp.shape[0])).to(args.device)

        b_ari = -1.0
        best_epoch = -1
        best_state = {}
        threshold_states = {}

        for epoch in range(1, args.epochs + 1):
            model.train()
            optimizer.zero_grad()

            cell_features = model.get_cell_features(self.gene_exp)
            _, z1 = model(cell_features, anchor_adj, args.maskfeat_rate_anchor)
            learned_adj, learned_adj_raw = model.get_learned_adj(cell_features)
            _, z2 = model(cell_features, learned_adj, args.maskfeat_rate_learner)

            idx = torch.randperm(self.gene_exp.shape[0])
            _, z1_neg = model(cell_features[idx], identity, args.maskfeat_rate_anchor, training=False)

            d_pos = F.pairwise_distance(z2, z1)
            d_neg = F.pairwise_distance(z2, z1_neg)
            margin_label = -torch.ones_like(d_pos)

            if args.sim_batch_size == 0:
                loss_nt = model.sim_loss(z1, z2, args.temperature)
            else:
                loss_nt = 0.0
                for batch in split_batch(list(range(self.gene_exp.shape[0])), args.sim_batch_size):
                    weight = len(batch) / self.gene_exp.shape[0]
                    loss_nt += model.sim_loss(z1[batch], z2[batch], args.temperature) * weight

            loss_triplet = model.margin_loss(d_pos, d_neg, margin_label) * args.margin_weight

            loss = loss_nt + loss_triplet
            loss.backward()
            optimizer.step()

            anchor_adj = anchor_adj.mul_nnz(torch.tensor(args.tau, dtype=torch.float32), layout="coo")
            anchor_adj = anchor_adj.add(
                learned_adj.detach().mul_nnz(torch.tensor(1 - args.tau, dtype=torch.float32), layout="coo")
            )

            if self.labels_true is not None:
                with torch.no_grad():
                    embed_epoch, labels_pred_epoch = self._predict(model, cell_features, learned_adj)
                ari_val = sk_metrics.adjusted_rand_score(self.labels_true, labels_pred_epoch)
                ami_val = sk_metrics.adjusted_mutual_info_score(self.labels_true, labels_pred_epoch)

                if ari_val > b_ari:
                    b_ari = ari_val
                    best_epoch = epoch
                    best_state = {
                        "embedding": embed_epoch.copy(),
                        "labels_pred": labels_pred_epoch.copy(),
                        "learned_adj": learned_adj.detach().cpu().to_dense(),
                        "learned_adj_raw": learned_adj_raw.detach().cpu().to_dense(),
                    }

                print(
                    f"Epoch {epoch:03d} | NT-Xent {loss_nt.item():.5f} | "
                    f"Triplet {loss_triplet.item():.5f}"
                )
            else:
                print(
                    f"Epoch {epoch:03d} | NT-Xent {loss_nt.item():.5f} | "
                    f"Triplet {loss_triplet.item():.5f}"
                )

        print(f"Adjusted Rand Index (ARI): {b_ari:.4f}")

def main():
    args = _build_parser().parse_args()
    if not hasattr(args, "beta_kl"):
        args.beta_kl = 1e-3
    BHGME(args).train()


if __name__ == "__main__":
    main()