# utils.py
# -*- coding: utf-8 -*-
import os, argparse, random, time, copy, warnings

import numpy as np
import scipy
from scipy.spatial import distance_matrix
from scipy.sparse import csr_matrix
import scanpy as sc
import anndata

import torch
import torch_sparse
from torch_sparse import SparseTensor
from torch_geometric.utils import dense_to_sparse
from sklearn.neighbors import NearestNeighbors
from typing import List, Tuple
from torch_geometric.utils import (
    dense_to_sparse,
    dropout_edge,
    from_scipy_sparse_matrix,
)

def get_feat_mask(features: torch.Tensor, rate: float) -> torch.Tensor:
    feat_size = features.shape[1]
    mask = torch.ones_like(features)
    samples = np.random.choice(feat_size, size=int(feat_size * rate), replace=False)
    mask[:, samples] = 0
    return mask


def dense2sparse(adj: torch.Tensor) -> SparseTensor:
    (row, col), val = dense_to_sparse(adj)
    num_nodes = adj.size(0)
    return SparseTensor(row=row, col=col, value=val, sparse_sizes=(num_nodes, num_nodes))


def normalize_adj_symm(adj) -> SparseTensor:
    assert adj.size(0) == adj.size(1)
    if not isinstance(adj, SparseTensor):
        adj = dense2sparse(adj)
    if not adj.has_value():
        adj = adj.fill_value(1.0, dtype=torch.float32)
    deg = torch_sparse.sum(adj, dim=1)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0.0)
    adj = torch_sparse.mul(adj, deg_inv_sqrt.view(-1, 1))
    adj = torch_sparse.mul(adj, deg_inv_sqrt.view(1, -1))
    return adj


def refine_labels(raw_labels, dist_sort_idx, n_neigh):
    n_cell = len(raw_labels)
    raw_labels = np.tile(raw_labels, (n_cell, 1))
    idx = dist_sort_idx[:, 1 : n_neigh + 1]
    new_labels = raw_labels[np.arange(n_cell)[:, None], idx]
    new_labels = scipy.stats.mode(new_labels, axis=1).mode
    return new_labels


def split_batch(init_list: List[int], batch_size: int) -> List[List[int]]:
    groups = zip(*(iter(init_list),) * batch_size)
    end_list = [list(i) for i in groups]
    count = len(init_list) % batch_size
    if count:
        end_list.append(init_list[-count:])
    return end_list


def _get_coord(adata: anndata.AnnData) -> np.ndarray:
    obs_columns = adata.obs.columns
    if "spatial" in adata.obsm.keys():
        return adata.obsm["spatial"]
    if {"x", "y"}.issubset(obs_columns):
        return adata.obs[["x", "y"]].values
    if {"st_x", "st_y"}.issubset(obs_columns):
        return adata.obs[["st_x", "st_y"]].values
    raise RuntimeError("Cannot find x–y coordinates in adata.")


def _knn_graph_fast(
    adata: anndata.AnnData,
    num_neighbors: int = 5,
    is_undirected: bool = True,
    n: int = 1000,
):
    coords = _get_coord(adata)
    k = num_neighbors
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(coords)
    k_idx = nbrs.kneighbors(coords, return_distance=False)

    rows = np.repeat(np.arange(adata.n_obs), k)
    cols = k_idx[:, 1:].flatten()
    data = np.ones(k * adata.n_obs)
    adj = csr_matrix((data, (rows, cols)), shape=(adata.n_obs, adata.n_obs))
    if is_undirected:
        adj = adj + adj.T
        adj.data = np.ones_like(adj.data)

    adata.obsp["knn_adj"] = adj

    dist_n = min(adata.n_obs, n + 1)
    dist_sort, dist_sort_idx = nbrs.kneighbors(coords, n_neighbors=dist_n)
    return adata, dist_sort, dist_sort_idx


def _knn_graph_dense(
    adata: anndata.AnnData, num_neighbors: int = 5, is_undirected: bool = True
):
    coords = _get_coord(adata)
    dist = distance_matrix(coords, coords)

    dist_sort_idx = np.argsort(dist, axis=1)
    dist_sort = np.take_along_axis(dist, dist_sort_idx, axis=1)

    adjacency = np.zeros((adata.n_obs, adata.n_obs), dtype=int)
    for i, n_idx in enumerate(dist_sort_idx):
        n_idx = n_idx[n_idx != i][:num_neighbors]
        adjacency[i, n_idx] = 1
    if is_undirected:
        adjacency = ((adjacency + adjacency.T) > 0).astype(int)

    adata.obsp["knn_adj"] = csr_matrix(adjacency)
    return adata, dist_sort, dist_sort_idx


def load_data_from_raw(args):
    adata = sc.read_h5ad(args.adata_file)
    if args.hvg:
        sc.pp.filter_genes(adata, min_cells=args.filter_cell)
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    args.norm_target = float(args.norm_target) if args.norm_target else None
    sc.pp.normalize_total(adata, target_sum=args.norm_target)
    sc.pp.log1p(adata)
    if "highly_variable" in adata.var.keys():
        adata = adata[:, adata.var["highly_variable"]].copy()
    counts = adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X
    gene_exp = torch.tensor(counts, dtype=torch.float32)
    if args.n_clusters is not None:
        labels, nclasses = None, args.n_clusters
    else:
        for key in ("cluster", "domain", "ground_truth"):
            if key in adata.obs.columns:
                cat = adata.obs[key].astype("category").values
                labels = torch.tensor(cat.codes, dtype=torch.long)
                nclasses = len(cat.categories)
                break
        else:
            raise RuntimeError("No cluster annotations found in adata.")
    if args.sparse_learner:
        adata, dist_sort, dist_sort_idx = _knn_graph_fast(
            adata, num_neighbors=args.a_k
        )
    else:
        adata, dist_sort, dist_sort_idx = _knn_graph_dense(
            adata, num_neighbors=args.a_k
        )
    (row, col), val = from_scipy_sparse_matrix(adata.obsp["knn_adj"])
    num_nodes = adata.obsp["knn_adj"].shape[0]
    adj_knn = SparseTensor(
        row=row, col=col, value=val.to(torch.float32), sparse_sizes=(num_nodes, num_nodes)
    )
    cell_coords = torch.tensor(_get_coord(adata), dtype=torch.float32)
    return (
        adata, gene_exp, labels, nclasses,
        adj_knn, cell_coords, dist_sort, dist_sort_idx,
    )

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BHGME")
    p.add_argument("--adata_file", type=str, required=True, help="Input AnnData .h5ad")
    p.add_argument("--output_dir", type=str, required=True, help="Directory for outputs")
    #p.add_argument("--n_clusters", type=int, required=True, help="Number of clusters")
    p.add_argument("--n_clusters", type=int, default=None)
    
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--clu_model", type=str, default="kmeans", choices=["kmeans"])
    p.add_argument("--sparse_learner", action="store_true")
    p.add_argument("--epochs", type=int, default=350)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--w_decay", type=float, default=1e-3)
    p.add_argument("--sim_batch_size", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--nlayers", type=int, default=2)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--exp_out", type=int, default=512)
    p.add_argument("--rep_dim", type=int, default=64)
    p.add_argument("--proj_dim", type=int, default=64)
    p.add_argument("--k", type=int, default=15)
    p.add_argument("--a_k", type=int, default=5)
    p.add_argument("--gamma", type=float, default=2)
    p.add_argument("--adj_weight", type=float, default=0.5)
    p.add_argument("--maskfeat_rate_anchor", type=float, default=0.9)
    p.add_argument("--maskfeat_rate_learner", type=float, default=0.6)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--dropedge_rate", type=float, default=0.5)
    p.add_argument("--margin", type=float, default=0.5)
    p.add_argument("--margin_weight", type=float, default=2)
    p.add_argument("--tau", type=float, default=0.999)
    p.add_argument("--hvg", action="store_true")
    p.add_argument("--filter_cell", type=int, default=100)
    p.add_argument("--norm_target", default=None)
    p.add_argument("--refine", type=int, default=0)
    p.add_argument("--no_bn", action="store_true")
    p.add_argument("--n_experts", type=int, default=4, help="number of experts")
    p.add_argument("--beta_kl", type=float, default=1e-3, help="KL coefficient β")
    
    p.add_argument("--ntrials", type=int, default=1, help="Number of repeated trials")
    p.add_argument("--eval_repeats", type=int, default=1, help="Repeated KMeans for ARI/AMI averaging")
    p.add_argument("--save_model", action="store_true", help="Save the best model per trial")

    return p