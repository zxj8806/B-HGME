#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import copy
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn import metrics as sk_metrics

from BHGME4h5ad import (
    BHGME, GCL, BayesianMoEGraphLearner,
    normalize_adj_symm, dense2sparse, refine_labels,
)


from utils import (
    _build_parser as _build_base_parser,
)


class Experiment(BHGME):
    def __init__(self, args):
        super().__init__(args)

    def predict(self, model, cell_features, learned_adj):

        model.eval()
        with torch.no_grad():
            embedding, _ = model(cell_features.detach(), learned_adj)

        embedding = embedding.cpu().detach().numpy()

        ari_ls, ami_ls = [], []
        for clu_trial in range(self.args.eval_repeats):
            kmeans = KMeans(
                n_clusters=self.nclasses,
                random_state=clu_trial,
                n_init="auto"
            ).fit(embedding)
            pred = kmeans.predict(embedding)
            if self.args.refine:
                pred = refine_labels(pred, self.dist_sort_idx, self.args.refine)

            if self.labels_true is not None:
                ari = sk_metrics.adjusted_rand_score(self.labels_true, pred)
                ami = sk_metrics.adjusted_mutual_info_score(self.labels_true, pred)
                ari_ls.append(ari)
                ami_ls.append(ami)

        if self.labels_true is None or len(ari_ls) == 0:
            ari, ami = np.nan, np.nan
        else:
            ari, ami = float(np.mean(ari_ls)), float(np.mean(ami_ls))

        return embedding, ari, ami

    def train(self):
        args = self.args

        # Prepare output directory & args record
        job_dir = args.output_dir
        os.makedirs(job_dir, exist_ok=True)
        with open(os.path.join(job_dir, 'args.txt'), 'w') as f:
            print(args, file=f)

        record_ls = []

        for trial in range(args.ntrials):
            self.setup_seed(trial)

            anchor_adj = normalize_adj_symm(self.adj_knn).to(args.device)

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

            best_ari = -np.inf
            best_ami = None
            best_embedding = None
            best_model = None

            ari_records, ami_records = [], []

            print(model)

            for epoch in range(1, 1 + args.epochs):
                optimizer.zero_grad()
                model.train()

                cell_features = model.get_cell_features(self.gene_exp)

                _, z1 = model(cell_features, anchor_adj, args.maskfeat_rate_anchor)

                learned_adj, _ = model.get_learned_adj(cell_features)

                _, z2 = model(cell_features, learned_adj, args.maskfeat_rate_learner)

                idx = torch.randperm(self.gene_exp.shape[0])
                _, z1_neg = model(cell_features[idx], identity, args.maskfeat_rate_anchor, training=False)

                d_pos = F.pairwise_distance(z2, z1)
                d_neg = F.pairwise_distance(z2, z1_neg)
                margin_label = -1 * torch.ones_like(d_pos)

                if args.sim_batch_size == 0:
                    loss_nt = model.sim_loss(z1, z2, args.temperature)
                else:
                    loss_nt = 0.0
                    n = self.gene_exp.shape[0]
                    step = args.sim_batch_size
                    for start in range(0, n, step):
                        end = min(start + step, n)
                        weight = (end - start) / n
                        loss_nt = loss_nt + model.sim_loss(
                            z1[start:end], z2[start:end], args.temperature
                        ) * weight

                loss_triplet = model.margin_loss(d_pos, d_neg, margin_label) * args.margin_weight

                loss = loss_nt + loss_triplet

                loss.backward()
                optimizer.step()

                anchor_adj = anchor_adj.mul_nnz(
                    torch.tensor(args.tau, dtype=torch.float32, device=args.device), layout="coo"
                )
                anchor_adj = anchor_adj.add(
                    learned_adj.detach().mul_nnz(
                        torch.tensor(1 - args.tau, dtype=torch.float32, device=args.device), layout="coo"
                    )
                )

                embedding, ari, ami = self.predict(model, cell_features, learned_adj)
                ari_records.append(ari)
                ami_records.append(ami)

                kl_val = None
                if hasattr(model.graph_learner, "latest_kl"):
                    kl_t = model.graph_learner.latest_kl
                    try:
                        kv = float(kl_t.detach().cpu().item()) if torch.is_tensor(kl_t) else float(kl_t)
                        if np.isfinite(kv):
                            kl_val = kv
                    except Exception:
                        kl_val = None

                if kl_val is None:
                    print(
                        f"Trial {trial} | Epoch {epoch:05d} | "
                        f"NT-Xent {float(loss_nt):.5f} | Triplet {float(loss_triplet):.5f} | "
                    )
                else:
                    print(
                        f"Trial {trial} | Epoch {epoch:05d} | "
                        f"NT-Xent {float(loss_nt):.5f} | Triplet {float(loss_triplet):.5f} | "
                    )

                if not np.isnan(ari) and ari > best_ari:
                    best_ari = ari
                    best_ami = ami
                    best_embedding = embedding
                    if args.save_model:
                        best_model = copy.deepcopy(model)

            print(f"Adjusted Rand Index (ARI): {best_ari:.4f}")
                        


def _build_parser() -> argparse.ArgumentParser:
    p = _build_base_parser()

    return p


def main():
    args = _build_parser().parse_args()
    print(args)
    experiment = Experiment(args)
    experiment.train()


if __name__ == "__main__":
    main()
