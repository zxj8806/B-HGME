import math
from typing import Optional, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from torch_geometric.nn.inits import reset

from hyperspherical_vae.distributions import VonMisesFisher, HypersphericalUniform


def _as_int(x: Union[int, torch.Tensor]) -> int:
    if isinstance(x, torch.Tensor):
        if x.numel() != 1:
            raise ValueError(f"Expected scalar tensor, got shape {tuple(x.shape)}.")
        return int(x.item())
    return int(x)


def _clamp_level(idx: int, hi: int) -> int:
    if idx < 1:
        return 1
    if idx > hi:
        return hi
    return idx


def _linear_blend_schedule(num_experts: int) -> torch.Tensor:
    beta_start, beta_end = 1e-4, 2e-2
    return torch.linspace(beta_start, beta_end, num_experts)


def _scalar_at(tensor: torch.Tensor, idx: Union[int, torch.Tensor]) -> float:
    idx = _as_int(idx) - 1  # switch to 0-index
    if idx < 0 or idx >= tensor.numel():
        raise IndexError("Index out of bounds")
    return tensor[idx].item()


def _precompute_blend_params(num_experts: int, device: torch.device
                             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    betas = _linear_blend_schedule(num_experts).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    cumulative_sum_reversed = torch.flip(sqrt_one_minus_alphas_cumprod, dims=[0]).cumsum(dim=0)
    cumulative_blend = torch.flip(cumulative_sum_reversed, dims=[0])

    return sqrt_one_minus_alphas_cumprod, sqrt_one_minus_alphas_cumprod, cumulative_blend


class B_HGME(nn.Module):

    def __init__(self,
                 encoder: nn.Module,
                 num_experts: int,
                 gating_hidden: int = 64,
                 gating_beta: float = 1e-2,
                 *,
                 gating_mode: str = "default"):
        super().__init__()
        self.encoder = encoder
        self.num_experts = int(num_experts)
        self.gating_hidden = gating_hidden
        self.gating_beta = gating_beta

        self.gating_mode = gating_mode.lower()
        if self.gating_mode not in {"scalar", "full", "default"}:
            raise ValueError(f"Unsupported gating_mode: {gating_mode}")

        (self.sqrt_one_minus_alphas_cumprod,
         _,
         self.cumulative_blend) = _precompute_blend_params(
            self.num_experts, torch.device("cpu")
        )

        self.noise_scale = nn.Parameter(torch.tensor(2e-2))

        self._gating_net: Optional[nn.Module] = None

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.encoder)
        if self._gating_net is not None:
            for m in self._gating_net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.)

    def encode(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    @staticmethod
    def _reparameterize(mean: torch.Tensor,
                        concentration: torch.Tensor):
        q_z = VonMisesFisher(mean, concentration)
        p_z = HypersphericalUniform(mean.size(1) - 1, device=mean.device)
        return q_z, p_z

    def _build_gating_net(self, in_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(in_dim, self.gating_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.gating_hidden, self.num_experts),
        )

    def _compute_gating(self,
                        emb: torch.Tensor,
                        coords: Optional[torch.Tensor] = None
                        ) -> torch.Tensor:
        if coords is not None:
            x = torch.cat([emb, coords], dim=1)
        else:
            x = emb

        in_dim = x.size(1)
        if (self._gating_net is None or
                getattr(self._gating_net[0], "in_features", None) != in_dim):
            self._gating_net = self._build_gating_net(in_dim).to(x.device)

        logits = self._gating_net(x)
        return F.softmax(logits, dim=1)

    def _expert_sample(self,
                       mean: torch.Tensor,
                       concentration: torch.Tensor) -> torch.Tensor:
        mean = F.normalize(mean, p=2, dim=1)
        concentration = torch.nan_to_num(concentration, nan=0.0, posinf=0.0, neginf=0.0)
        concentration = concentration.sum(dim=1, keepdim=True)
        concentration = F.softplus(concentration) + 1e-3
        q_z, _ = self._reparameterize(mean, concentration)
        sampled_z = q_z.rsample()
        return F.normalize(mean + self.noise_scale * sampled_z, p=2, dim=1)

    def _edge_gating_weight(self,
                            pi: torch.Tensor,
                            edge_index: torch.Tensor,
                            expert_id: int) -> torch.Tensor:
        gi = pi[edge_index[0], expert_id]
        gj = pi[edge_index[1], expert_id]
        return 0.5 * (gi + gj)

    def decode(self,
               z: List[torch.Tensor],
               z_conc: List[torch.Tensor],
               edge_index: torch.Tensor,
               τ: Union[int, torch.Tensor],
               num_experts: Union[int, torch.Tensor],
               *,
               sigmoid: bool = True,
               coords: Optional[torch.Tensor] = None) -> torch.Tensor:

        device = z[0].device
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.cumulative_blend = self.cumulative_blend.to(device)

        τ = _clamp_level(_as_int(τ), self.num_experts)
        num_experts = _clamp_level(_as_int(num_experts), self.num_experts)

        if self.gating_mode != "default":
            pi = self._compute_gating(z[0].detach(), coords)

        norm_factor = _scalar_at(self.cumulative_blend, τ)
        value = torch.zeros(edge_index.size(1), device=device)

        for ℓ in range(τ, num_experts + 1):
            k = ℓ - 1
            factor = _scalar_at(self.sqrt_one_minus_alphas_cumprod, ℓ)
            mean, conc = z[k], z_conc[k]
            emb = self._expert_sample(mean, conc)
            sim_k = (emb[edge_index[0]] * emb[edge_index[1]]).sum(dim=1)

            if self.gating_mode == "full":
                w_k = self._edge_gating_weight(pi, edge_index, k)
                value += factor * w_k * sim_k
            elif self.gating_mode == "scalar":
                w_k_scalar = pi[:, k].mean()
                value += factor * w_k_scalar * sim_k
            else:
                value += factor * sim_k

        value = value / norm_factor
        return torch.clamp(value, 0, 1) if sigmoid else value

    def decode_all(self,
                   z: List[torch.Tensor],
                   z_conc: List[torch.Tensor],
                   τ: Union[int, torch.Tensor],
                   num_experts: Union[int, torch.Tensor],
                   *,
                   sigmoid: bool = True,
                   coords: Optional[torch.Tensor] = None
                   ) -> torch.Tensor:

        device = z[0].device
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.cumulative_blend = self.cumulative_blend.to(device)

        τ = _clamp_level(_as_int(τ), self.num_experts)
        num_experts = _clamp_level(_as_int(num_experts), self.num_experts)

        if self.gating_mode != "default":
            pi = self._compute_gating(z[0].detach(), coords)

        n_nodes = z[0].size(0)
        norm_factor = _scalar_at(self.cumulative_blend, τ)
        pred_adj = torch.zeros(n_nodes, n_nodes, device=device)

        for ℓ in range(τ, num_experts + 1):
            k = ℓ - 1
            factor = _scalar_at(self.sqrt_one_minus_alphas_cumprod, ℓ)
            mean, conc = z[k], z_conc[k]
            emb = self._expert_sample(mean, conc)
            sim_k = torch.mm(emb, emb.t())  # n × n

            if self.gating_mode == "full":
                w_k = 0.5 * (pi[:, k].unsqueeze(1) + pi[:, k].unsqueeze(0))
                pred_adj += factor * w_k * sim_k
            elif self.gating_mode == "scalar":
                w_k_scalar = pi[:, k].mean()
                pred_adj += factor * w_k_scalar * sim_k
            else:
                pred_adj += factor * sim_k

        pred_adj = pred_adj / norm_factor
        return torch.clamp(pred_adj, 0, 1) if sigmoid else pred_adj

    def forward_all(self, z, z_conc, τ, num_experts, *, sigmoid=True, **kwargs):
        return self.decode_all(z, z_conc, τ, num_experts, sigmoid=sigmoid, **kwargs)

    def recon_loss(self,
                   z: List[torch.Tensor],
                   z_conc: List[torch.Tensor],
                   pos_edge_idx: torch.Tensor,
                   τ: Union[int, torch.Tensor],
                   num_experts: Union[int, torch.Tensor],
                   coords: Optional[torch.Tensor] = None) -> torch.Tensor:

        eps = 1e-10
        device = z[0].device

        τ = _clamp_level(_as_int(τ), self.num_experts)
        num_experts = _clamp_level(_as_int(num_experts), self.num_experts)

        if num_experts < τ:
            raise ValueError(f"num_experts < τ ({num_experts} < {τ})")

        kl_z = 0.0
        for k in range(τ - 1, num_experts):
            mean, conc = z[k], z_conc[k]
            conc = torch.nan_to_num(conc, nan=0.0, posinf=0.0, neginf=0.0).sum(dim=1, keepdim=True)
            conc = F.softplus(conc) + 1.0
            mean = F.normalize(mean, p=2, dim=1)
            q, p = self._reparameterize(mean, conc)
            kl_z += torch.distributions.kl.kl_divergence(q, p).mean()
        kl_z = kl_z / (num_experts - τ + 1)

        pos_loss = -torch.log(
            self.decode(z, z_conc, pos_edge_idx, τ, num_experts,
                        coords=coords, sigmoid=True) + eps).mean()

        n_nodes = z[0].size(0)
        neg_edge_idx = negative_sampling(pos_edge_idx, num_nodes=n_nodes)
        neg_loss = -torch.log(
            1.0 - self.decode(z, z_conc, neg_edge_idx, τ, num_experts,
                              coords=coords, sigmoid=True) + eps).mean()

        # -------- Dirichlet-style KL (gating) --------
        pi = self._compute_gating(z[0].detach(), coords)
        uniform = 1.0 / self.num_experts
        kl_gate = (pi * (pi.clamp_min(eps).log() - math.log(uniform))).sum(1).mean()

        return pos_loss + neg_loss + self.noise_scale * kl_z + self.gating_beta * kl_gate

    def test(self,
             z: List[torch.Tensor],
             z_conc: List[torch.Tensor],
             pos_edge_idx: torch.Tensor,
             neg_edge_idx: torch.Tensor,
             τ: Union[int, torch.Tensor],
             num_experts: Union[int, torch.Tensor],
             coords: Optional[torch.Tensor] = None) -> Tuple[float, float]:

        from sklearn.metrics import roc_auc_score, average_precision_score

        with torch.no_grad():
            τ = _clamp_level(_as_int(τ), self.num_experts)
            num_experts = _clamp_level(_as_int(num_experts), self.num_experts)

            y_pos = torch.ones(pos_edge_idx.size(1), device=z[0].device)
            y_neg = torch.zeros(neg_edge_idx.size(1), device=z[0].device)
            y_true = torch.cat([y_pos, y_neg], dim=0)

            p_pos = self.decode(z, z_conc, pos_edge_idx, τ, num_experts,
                                coords=coords, sigmoid=True)
            p_neg = self.decode(z, z_conc, neg_edge_idx, τ, num_experts,
                                coords=coords, sigmoid=True)
            y_pred = torch.cat([p_pos, p_neg], dim=0)

            auc = roc_auc_score(y_true.cpu().numpy(), y_pred.cpu().detach().numpy())
            ap = average_precision_score(y_true.cpu().numpy(), y_pred.cpu().detach().numpy())
        return auc, ap

    @property
    def decoder(self):
        return self
