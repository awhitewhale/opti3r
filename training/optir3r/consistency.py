"""Small torch-only consistency components for Opti3R.

The renderer is intentionally conservative: it changes attenuation, scattering
and background light while keeping the input geometry fixed.  It is used only
to create a second training view, never as an RGB reconstruction target.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _depth01(depth: torch.Tensor) -> torch.Tensor:
    if depth.ndim == 5 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    finite = torch.isfinite(depth) & (depth > 0)
    safe = torch.where(finite, depth, torch.zeros_like(depth))
    count = finite.flatten(-2).sum(-1, keepdim=True).clamp_min(1)
    lo = torch.where(finite, depth, torch.full_like(depth, float("inf"))).flatten(-2).amin(-1, keepdim=True)
    hi = torch.where(finite, depth, torch.full_like(depth, float("-inf"))).flatten(-2).amax(-1, keepdim=True)
    lo = torch.where(torch.isfinite(lo), lo, torch.zeros_like(lo)).unsqueeze(-1)
    hi = torch.where(torch.isfinite(hi), hi, torch.ones_like(hi)).unsqueeze(-1)
    del safe, count
    return ((depth - lo) / (hi - lo).clamp_min(1e-4)).clamp(0, 1) * finite


def make_counterfactual_images(images: torch.Tensor, depths: torch.Tensor, severity: float = 0.65) -> torch.Tensor:
    """Render a bounded optical intervention with fixed scene geometry."""
    if images.ndim != 5 or images.shape[2] != 3:
        raise ValueError(f"Expected images [B,S,3,H,W], got {tuple(images.shape)}")
    b, s, _, h, w = images.shape
    d = _depth01(depths.detach()).to(device=images.device, dtype=images.dtype)
    z = 0.5 + (2.5 + 2.5 * float(severity)) * d[:, :, None]

    low_h, low_w = max(4, h // 32), max(4, w // 32)
    noise = torch.rand((b, 1, low_h, low_w), device=images.device, dtype=images.dtype)
    field = F.interpolate(noise, size=(h, w), mode="bilinear", align_corners=False).unsqueeze(2)
    field = 1.0 + (0.08 + 0.16 * float(severity)) * (field - 0.5)

    base = torch.empty((b, 1, 1, 1, 1), device=images.device, dtype=images.dtype).uniform_(0.18, 0.55)
    rgb = torch.tensor([1.45, 0.95, 0.62], device=images.device, dtype=images.dtype).view(1, 1, 3, 1, 1)
    beta_d = base * rgb
    beta_b = beta_d * torch.empty((b, 1, 3, 1, 1), device=images.device, dtype=images.dtype).uniform_(0.60, 0.90)
    background = torch.empty((b, 1, 3, 1, 1), device=images.device, dtype=images.dtype).uniform_(0.04, 0.32)

    optical_depth = z * field
    direct = torch.exp(-beta_d * optical_depth)
    backscatter = torch.exp(-beta_b * optical_depth)
    return (images.detach().clamp(0, 1) * direct + background * (1.0 - backscatter)).clamp(0, 1)


def _scale_invariant_depth(depth: torch.Tensor) -> torch.Tensor:
    if depth.ndim == 5 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    depth = depth.clamp_min(1e-5)
    scale = depth.flatten(-2).median(dim=-1).values.clamp_min(1e-5).unsqueeze(-1).unsqueeze(-1)
    return torch.log(depth / scale)


def _scale_invariant_points(points: torch.Tensor) -> torch.Tensor:
    norm = points.norm(dim=-1).clamp_min(1e-5)
    scale = norm.flatten(-2).median(dim=-1).values.clamp_min(1e-5).unsqueeze(-1).unsqueeze(-1)
    return points / scale.unsqueeze(-1)


def intervention_consistency_loss(prediction: dict, counterfactual: dict) -> dict[str, torch.Tensor]:
    """Compare geometry outputs for two optical versions of one sample."""
    terms = {}
    if "depth" in prediction and "depth" in counterfactual:
        d1 = _scale_invariant_depth(prediction["depth"])
        d2 = _scale_invariant_depth(counterfactual["depth"])
        terms["loss_intervene_depth"] = F.smooth_l1_loss(d1, d2)
    if "pose_enc" in prediction and "pose_enc" in counterfactual:
        terms["loss_intervene_pose"] = F.smooth_l1_loss(prediction["pose_enc"], counterfactual["pose_enc"])
    if "world_points" in prediction and "world_points" in counterfactual:
        p1 = _scale_invariant_points(prediction["world_points"])
        p2 = _scale_invariant_points(counterfactual["world_points"])
        terms["loss_intervene_point"] = F.smooth_l1_loss(p1, p2)
    if not terms:
        raise ValueError("No shared geometry outputs for intervention consistency")
    terms["loss_intervene"] = sum(terms.values()) / len(terms)
    return terms
