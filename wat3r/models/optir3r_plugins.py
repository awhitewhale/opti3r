"""Dependency-light plug-in blocks for Opti3R dense prediction heads.

The blocks preserve ``[B, C, H, W]`` shapes and are intentionally wrapped by a
zero-initialized residual gate in :class:`DPTHead`.  This makes a plug-in
checkpoint start exactly from the released Wat3R solution.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class LSKBlock(nn.Module):
    """Large selective kernels, adapted from the supplied LSK module."""

    def __init__(self, dim: int):
        super().__init__()
        hidden = max(dim // 2, 1)
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim, dim, 7, padding=9, groups=dim, dilation=3
        )
        self.conv1 = nn.Conv2d(dim, hidden, 1)
        self.conv2 = nn.Conv2d(dim, hidden, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(hidden, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)
        attn1, attn2 = self.conv1(attn1), self.conv2(attn2)
        attn = torch.cat([attn1, attn2], dim=1)
        agg = torch.cat(
            [attn.mean(dim=1, keepdim=True), attn.amax(dim=1, keepdim=True)],
            dim=1,
        )
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0:1] + attn2 * sig[:, 1:2]
        return x * self.conv(attn)


class SCSABlock(nn.Module):
    """Spatial-and-channel squeeze attention from the supplied SCSA module."""

    def __init__(self, dim: int, head_num: int = 8, window_size: int = 7):
        super().__init__()
        if dim % 4 != 0 or dim % head_num != 0:
            raise ValueError(f"SCSA requires dim divisible by 4 and heads: {dim}")
        group = dim // 4
        self.dim, self.head_num = dim, head_num
        self.head_dim = dim // head_num
        self.scale = self.head_dim ** -0.5
        self.local = nn.Conv1d(group, group, 3, padding=1, groups=group)
        self.small = nn.Conv1d(group, group, 5, padding=2, groups=group)
        self.medium = nn.Conv1d(group, group, 7, padding=3, groups=group)
        self.large = nn.Conv1d(group, group, 9, padding=4, groups=group)
        self.norm_h = nn.GroupNorm(4, dim)
        self.norm_w = nn.GroupNorm(4, dim)
        self.norm = nn.GroupNorm(1, dim)
        self.q = nn.Conv2d(dim, dim, 1, groups=dim)
        self.k = nn.Conv2d(dim, dim, 1, groups=dim)
        self.v = nn.Conv2d(dim, dim, 1, groups=dim)
        self.window_size = window_size
        self.ca_gate = nn.Sigmoid()

    def _axis(self, z: torch.Tensor, norm: nn.Module) -> torch.Tensor:
        a, b, c, d = torch.chunk(z, 4, dim=1)
        return norm(torch.cat(
            [self.local(a), self.small(b), self.medium(c), self.large(d)], dim=1
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        ah = self._axis(x.mean(dim=3), self.norm_h).view(b, c, h, 1)
        # Transpose the spatial axes so the same 1-D filters model the width.
        aw = self._axis(x.mean(dim=2), self.norm_w).view(b, c, 1, w)
        gated = x * ah.sigmoid() * aw.sigmoid()
        if self.window_size > 1 and h >= self.window_size and w >= self.window_size:
            pooled = F.avg_pool2d(
                gated, self.window_size, self.window_size, ceil_mode=True
            )
        else:
            pooled = gated
        y = self.norm(pooled)
        q, k, v = self.q(y), self.k(y), self.v(y)
        q = q.reshape(b, self.head_num, self.head_dim, -1)
        k = k.reshape(b, self.head_num, self.head_dim, -1)
        v = v.reshape(b, self.head_num, self.head_dim, -1)
        attn = (q @ k.transpose(-2, -1) * self.scale).softmax(dim=-1)
        attn = attn @ v
        ph, pw = pooled.shape[-2:]
        channel_gate = attn.reshape(b, c, ph, pw).mean((2, 3), keepdim=True)
        return gated * channel_gate.sigmoid()


class MogaBlock(nn.Module):
    """Multi-order gated aggregation from the supplied MogaNet implementation."""

    def __init__(self, dim: int):
        super().__init__()
        if dim % 8 != 0:
            raise ValueError(f"Moga requires channels divisible by 8: {dim}")
        c1, c2 = dim * 3 // 8, dim // 2
        c0 = dim - c1 - c2
        self.c0, self.c1, self.c2 = c0, c1, c2
        self.proj1 = nn.Conv2d(dim, dim, 1)
        self.gate = nn.Conv2d(dim, dim, 1)
        self.dw0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.dw1 = nn.Conv2d(c1, c1, 5, padding=4, dilation=2, groups=c1)
        self.dw2 = nn.Conv2d(c2, c2, 7, padding=9, dilation=3, groups=c2)
        self.pw = nn.Conv2d(dim, dim, 1)
        self.proj2 = nn.Conv2d(dim, dim, 1)
        self.sigma = nn.Parameter(torch.full((1, dim, 1, 1), 1e-5))
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        z = self.proj1(x)
        z = z + self.sigma * (z - z.mean((2, 3), keepdim=True))
        z = self.act(z)
        base = self.dw0(z)
        z1 = self.dw1(base[:, self.c0:self.c0 + self.c1])
        z2 = self.dw2(base[:, self.c0 + self.c1:])
        value = self.pw(torch.cat([base[:, :self.c0], z1, z2], dim=1))
        return shortcut + self.proj2(self.act(self.gate(z)) * self.act(value))


class SAFMBlock(nn.Module):
    """Multi-level spatial feature modulation from the supplied SAFM module."""

    def __init__(self, dim: int, levels: int = 4):
        super().__init__()
        if dim % levels != 0:
            raise ValueError(f"SAFM requires channels divisible by levels: {dim}")
        chunk = dim // levels
        self.levels = levels
        self.mfr = nn.ModuleList(
            [nn.Conv2d(chunk, chunk, 3, padding=1, groups=chunk) for _ in range(levels)]
        )
        self.aggr = nn.Conv2d(dim, dim, 1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        chunks = x.chunk(self.levels, dim=1)
        out = []
        for i, (part, conv) in enumerate(zip(chunks, self.mfr)):
            if i:
                ph, pw = max(h // (2 ** i), 1), max(w // (2 ** i), 1)
                part = F.adaptive_max_pool2d(part, (ph, pw))
                part = F.interpolate(conv(part), size=(h, w), mode="nearest")
            else:
                part = conv(part)
            out.append(part)
        return self.act(self.aggr(torch.cat(out, dim=1))) * x


class NAMBlock(nn.Module):
    """Normalization-based channel and spatial attention."""

    def __init__(self, dim: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(dim, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        z = self.bn(x)
        weight = self.bn.weight.abs() / self.bn.weight.abs().sum().clamp_min(1e-6)
        z = z * weight.view(1, -1, 1, 1)
        return torch.sigmoid(z) * residual


class SGEBlock(nn.Module):
    """Spatial group-wise enhancement from the supplied SGE module."""

    def __init__(self, dim: int, groups: int = 8):
        super().__init__()
        if dim % groups != 0:
            raise ValueError(f"SGE requires channels divisible by groups: {dim}")
        self.groups = groups
        self.weight = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        z = x.reshape(b * self.groups, c // self.groups, h, w)
        z = z * self.pool(z)
        t = z.sum(dim=1, keepdim=True)
        t = t.reshape(b * self.groups, -1)
        t = (t - t.mean(dim=1, keepdim=True)) / (t.std(dim=1, keepdim=True) + 1e-5)
        t = t.reshape(b, self.groups, h, w)
        t = (t * self.weight + self.bias).sigmoid()
        return (z * t.reshape(b * self.groups, 1, h, w)).reshape(b, c, h, w)


class _PKICAA(nn.Module):
    """Dependency-free Context Anchor Attention used by PKIBlock."""

    def __init__(self, dim: int, kernel_size: int = 11):
        super().__init__()
        self.pool = nn.AvgPool2d(7, 1, 3)
        self.in_proj = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False), nn.BatchNorm2d(dim), nn.SiLU()
        )
        self.h = nn.Conv2d(dim, dim, (1, kernel_size), padding=(0, kernel_size // 2), groups=dim)
        self.v = nn.Conv2d(dim, dim, (kernel_size, 1), padding=(kernel_size // 2, 0), groups=dim)
        self.out_proj = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False), nn.BatchNorm2d(dim), nn.SiLU()
        )
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.pool(x)
        return self.act(self.out_proj(self.v(self.h(self.in_proj(z)))))


class PKIBlock(nn.Module):
    """Poly-kernel inception block adapted from the supplied PKINet module."""

    def __init__(self, dim: int):
        super().__init__()
        self.pre = nn.Sequential(nn.Conv2d(dim, dim, 1, bias=False), nn.BatchNorm2d(dim), nn.SiLU())
        self.dw = nn.ModuleList([
            nn.Conv2d(dim, dim, k, padding=k // 2, groups=dim) for k in (3, 5, 7, 9, 11)
        ])
        self.mix = nn.Sequential(nn.Conv2d(dim, dim, 1, bias=False), nn.BatchNorm2d(dim), nn.SiLU())
        self.caa = _PKICAA(dim, 11)
        self.post = nn.Sequential(nn.Conv2d(dim, dim, 1, bias=False), nn.BatchNorm2d(dim), nn.SiLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.pre(x)
        y = self.dw[0](z)
        for branch in self.dw[1:]:
            y = y + branch(z)
        y = self.mix(y)
        return x + self.post(y + y * self.caa(z))


class CFBlock(nn.Module):
    """Lightweight dependency-free adaptation of the supplied CFBlock."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.BatchNorm2d(dim, eps=1e-6)
        self.h = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.v = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.ffn = nn.Sequential(
            nn.BatchNorm2d(dim, eps=1e-6), nn.Conv2d(dim, dim * 2, 3, padding=1),
            nn.GELU(), nn.Conv2d(dim * 2, dim, 3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm(x)
        z = self.proj(self.h(z) + self.v(z))
        x = x + z
        return x + self.ffn(x)


class PSABlock(nn.Module):
    """Parallel polarized self-attention, a pixel-regression-oriented plug-in."""

    def __init__(self, dim: int):
        super().__init__()
        half = max(dim // 2, 1)
        self.ch_v = nn.Conv2d(dim, half, 1)
        self.ch_q = nn.Conv2d(dim, 1, 1)
        self.ch_z = nn.Conv2d(half, dim, 1)
        self.ch_norm = nn.LayerNorm(dim)
        self.sp_v = nn.Conv2d(dim, half, 1)
        self.sp_q = nn.Conv2d(dim, half, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        cv = self.ch_v(x).reshape(b, c // 2, -1)
        cq = self.ch_q(x).reshape(b, -1, 1).softmax(dim=1)
        cz = torch.matmul(cv, cq).unsqueeze(-1)
        cw = self.ch_z(cz).reshape(b, c, 1).transpose(1, 2)
        cw = self.ch_norm(cw).transpose(1, 2).reshape(b, c, 1, 1).sigmoid()
        channel_out = cw * x
        sv = self.sp_v(x).reshape(b, c // 2, -1)
        sq = self.pool(self.sp_q(x)).reshape(b, 1, c // 2).softmax(dim=-1)
        sw = torch.matmul(sq, sv).reshape(b, 1, h, w).sigmoid()
        return channel_out + sw * x


class PlugInBlock(nn.Module):
    """Factory used by DPTHead; ``none`` is an exact identity."""

    def __init__(self, kind: str, dim: int):
        super().__init__()
        kind = (kind or "none").lower()
        self.kind = kind
        if kind == "none":
            self.block = nn.Identity()
        elif kind == "lsk":
            self.block = LSKBlock(dim)
        elif kind == "scsa":
            self.block = SCSABlock(dim)
        elif kind == "moga":
            self.block = MogaBlock(dim)
        elif kind == "safm":
            self.block = SAFMBlock(dim)
        elif kind == "nam":
            self.block = NAMBlock(dim)
        elif kind == "sge":
            self.block = SGEBlock(dim)
        elif kind == "pki":
            self.block = PKIBlock(dim)
        elif kind == "cf":
            self.block = CFBlock(dim)
        elif kind == "psa":
            self.block = PSABlock(dim)
        else:
            raise ValueError(f"Unknown DPT plug-in: {kind}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
