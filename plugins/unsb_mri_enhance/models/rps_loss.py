import json
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RPSConfig:
    r_low: int = 18
    r_high: int = 145
    eps: float = 1e-8
    clamp_min: float = 1e-8
    smooth: bool = True
    smooth_kernel: Tuple[float, ...] = (1, 2, 3, 2, 1)  # normalized internally


def _make_radius_index(H: int, W: int, device: torch.device) -> Tuple[torch.Tensor, int]:
    """Return ridx_flat (HW,), and R = max radius bins."""
    yy = torch.arange(H, device=device).view(H, 1).float()
    xx = torch.arange(W, device=device).view(1, W).float()
    cy = H // 2
    cx = W // 2
    r = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    ridx = torch.floor(r).long()
    R = int(ridx.max().item()) + 1
    return ridx.view(-1), R


class RPSBandLoss(nn.Module):
    """
    Radial Power Spectrum band-only loss.

    Input: img (B,C,H,W) in float (any range ok but consistent with stats)
    - Applies mask if provided (B,1,H,W) or (B,H,W)
    - Computes power spectrum |FFTshift(FFT2(img))|^2
    - Radial-bins into rho(r)
    - Band-only renormalize in [r_low, r_high)
    - Optional 1D smoothing on rho
    - L1 distance in log space to target band profile
    """

    def __init__(
        self,
        stats_npz_path: str,
        H: int,
        W: int,
        cfg: Optional[RPSConfig] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.cfg = cfg or RPSConfig()
        self.H, self.W = H, W
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Radius map
        ridx_flat, R = _make_radius_index(H, W, self.device)
        self.register_buffer("ridx_flat", ridx_flat, persistent=False)
        self.R = R
        self.HW = H * W

        # Load target profiles
        d = np.load(stats_npz_path, allow_pickle=True)
        meta = json.loads(d["meta"].item())
        # If stats file already band-only, great; otherwise we still apply band-only at runtime.
        t1 = torch.from_numpy(d["rho_mean_t1"]).float().to(self.device)
        t2 = torch.from_numpy(d["rho_mean_t2"]).float().to(self.device)

        # Ensure length matches our R
        if t1.numel() != R or t2.numel() != R:
            raise ValueError(f"Stats R={t1.numel()} but current R={R}. "
                             f"Check H/W or regenerate stats.")

        self.register_buffer("tgt_t1", t1, persistent=True)
        self.register_buffer("tgt_t2", t2, persistent=True)
        self.meta = meta  # for debugging

        # Smoothing kernel for 1D profile (depthwise conv)
        if self.cfg.smooth:
            k = torch.tensor(self.cfg.smooth_kernel, dtype=torch.float32, device=self.device)
            k = k / k.sum()
            self.register_buffer("smooth_k", k.view(1, 1, -1), persistent=False)

    @torch.no_grad()
    def _band_mask(self) -> torch.Tensor:
        w = torch.zeros(self.R, device=self.device, dtype=torch.float32)
        lo, hi = self.cfg.r_low, self.cfg.r_high
        w[lo:hi] = 1.0
        return w

    def _maybe_smooth(self, rho: torch.Tensor) -> torch.Tensor:
        # rho: (B,C,R)
        if not self.cfg.smooth:
            return rho
        # depthwise conv1d per (B*C)
        B, C, R = rho.shape
        x = rho.reshape(B * C, 1, R)
        pad = self.smooth_k.shape[-1] // 2
        x = F.pad(x, (pad, pad), mode="reflect")
        x = F.conv1d(x, self.smooth_k)
        return x.reshape(B, C, R)

    def _radial_profile(self, power: torch.Tensor) -> torch.Tensor:
        """
        power: (B,C,H,W) float
        returns rho: (B,C,R) with mean power per radius bin
        """
        B, C, H, W = power.shape
        assert H == self.H and W == self.W
        x = power.reshape(B, C, self.HW)  # (B,C,HW)

        # scatter add sums and counts
        ridx = self.ridx_flat  # (HW,)
        # sums: (B,C,R)
        sums = torch.zeros((B, C, self.R), device=power.device, dtype=torch.float32)
        sums.scatter_add_(dim=2, index=ridx.view(1, 1, -1).expand(B, C, -1), src=x)

        # counts: (R,)
        # compute once on device
        if not hasattr(self, "_counts"):
            counts = torch.bincount(ridx, minlength=self.R).float().to(power.device)
            self.register_buffer("_counts", counts, persistent=False)

        rho = sums / (self._counts.view(1, 1, -1) + self.cfg.eps)
        return rho

    def _fft_power(self, img: torch.Tensor) -> torch.Tensor:
        """
        img: (B,C,H,W) float
        return power spectrum: (B,C,H,W)
        """
        # torch.fft.fft2 returns complex
        Fimg = torch.fft.fft2(img, dim=(-2, -1))
        Fimg = torch.fft.fftshift(Fimg, dim=(-2, -1))
        power = (Fimg.real ** 2 + Fimg.imag ** 2).float()
        return power

    def _band_renorm(self, rho: torch.Tensor) -> torch.Tensor:
        """
        rho: (B,C,R)
        returns band-only normalized rho with zeros outside band.
        """
        lo, hi = self.cfg.r_low, self.cfg.r_high
        band = rho[:, :, lo:hi]
        denom = band.sum(dim=-1, keepdim=True) + self.cfg.eps
        band = band / denom
        out = torch.zeros_like(rho)
        out[:, :, lo:hi] = band
        return out

    def forward(self, img: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        img: (B,3,H,W) with RGB [T1,T2,T1] or (B,2,H,W) [T1,T2]
        mask: optional (B,1,H,W) or (B,H,W)
        returns scalar loss
        """
        assert img.dim() == 4, "img should be (B,C,H,W)"
        B, C, H, W = img.shape
        if (H, W) != (self.H, self.W):
            raise ValueError(f"Expected {(self.H,self.W)}, got {(H,W)}")

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            img = img * mask

        # Use only unique channels: ch0=T1, ch1=T2 (ignore ch2 duplicate if present)
        if C >= 2:
            img2 = img[:, :2, :, :]
        else:
            raise ValueError("Need at least 2 channels (T1,T2).")

        power = self._fft_power(img2)          # (B,2,H,W)
        rho = self._radial_profile(power)      # (B,2,R)

        rho = self._maybe_smooth(rho)
        rho = self._band_renorm(rho)
        rho = torch.clamp(rho, min=self.cfg.clamp_min)

        # Target: also ensure band-only behavior (safe even if already band-only)
        tgt = torch.stack([self.tgt_t1, self.tgt_t2], dim=0).unsqueeze(0)  # (1,2,R)
        tgt = tgt.expand(B, -1, -1)
        tgt = self._band_renorm(tgt)
        tgt = torch.clamp(tgt, min=self.cfg.clamp_min)

        # Log-L1
        loss = (torch.abs(torch.log(rho) - torch.log(tgt))).mean()
        return loss