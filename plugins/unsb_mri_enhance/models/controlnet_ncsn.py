import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

from .ncsn_networks import get_timestep_embedding


class ZeroConv2d(nn.Module):
    """1x1 conv initialized to zero (so ControlNet starts as a no-op)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _ResBlock(nn.Module):
    def __init__(self, ch: int, norm_layer, use_bias: bool):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, 1, 1, bias=use_bias)
        self.norm1 = norm_layer(ch)
        self.act = nn.LeakyReLU(0.2, True)
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1, bias=use_bias)
        self.norm2 = norm_layer(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(x + h)


def _schedule(time_idx: torch.Tensor,
              num_timesteps: int,
              mode: str,
              power: float = 1.0,
              step_start: int = 0) -> torch.Tensor:
    """Returns a gating scalar g(t) in shape (1 or B, 1, 1, 1) suitable for broadcasting."""
    if num_timesteps <= 1:
        t = torch.zeros_like(time_idx).float()
    else:
        t = time_idx.float() / float(num_timesteps - 1)

    if mode == 'off':
        g = torch.zeros_like(t)
    elif mode == 'const':
        g = torch.ones_like(t)
    elif mode == 'linear':
        g = t
    elif mode == 'pow':
        g = torch.pow(t, power)
    elif mode == 'step':
        g = (time_idx >= step_start).float()
    else:
        raise ValueError(f'Unknown gate mode: {mode}')

    return g.view(-1, 1, 1, 1)


def _ports_from_preset(preset: str, n_res: int = 9):
    """Return a list of port names used for injection."""
    preset = (preset or 'low').lower()
    if preset == 'high':
        ports = ['down1', 'down2'] + [f'res{i}' for i in range(n_res)] + ['up1', 'up2']
    elif preset == 'mid':
        ports = ['down1', 'down2'] + [f'res{i}' for i in range(n_res)]
    elif preset == 'low':
        # 2 downs + 3 mid res (centered) + 2 ups = 7 ports
        mid = [f'res{i}' for i in range(3, 6)] if n_res >= 6 else [f'res{i}' for i in range(n_res)]
        ports = ['down1', 'down2'] + mid + ['up1', 'up2']
    else:
        raise ValueError(f'Unknown ctrl_ports preset: {preset}')
    return ports


class ControlNetNCSN(nn.Module):
    """A lightweight ControlNet that produces residual feature injections for ResnetGenerator_ncsn.

    It takes a condition tensor (e.g., lowpass + grad + boundary band) and outputs
    a dict of residual feature maps for selected injection "ports".

    Port shapes are aligned with ResnetGenerator_ncsn (ngf=64 default):
      - down1: (B, 2*ngf, H/2, W/2)
      - down2: (B, 4*ngf, H/4, W/4)
      - res{i}: (B, 4*ngf, H/4, W/4) for i=0..8
      - up1:   (B, 2*ngf, H/2, W/2)
      - up2:   (B, 1*ngf, H,   W)
    """

    def __init__(self, cond_nc: int, ngf: int = 64, norm_layer=nn.InstanceNorm2d, opt=None):
        super().__init__()
        self.cond_nc = cond_nc
        self.ngf = ngf
        self.opt = opt
        self.num_timesteps = getattr(opt, 'num_timesteps', 5) if opt is not None else 5

        # gating schedule (defaults are chosen to be stable for T=5)
        self.gate_down = getattr(opt, 'gate_down', 'linear') if opt is not None else 'linear'
        self.gate_mid = getattr(opt, 'gate_mid', 'pow') if opt is not None else 'pow'
        self.gate_up = getattr(opt, 'gate_up', 'pow') if opt is not None else 'pow'
        self.gate_down_pow = float(getattr(opt, 'gate_down_pow', 1.0) if opt is not None else 1.0)
        self.gate_mid_pow = float(getattr(opt, 'gate_mid_pow', 2.0) if opt is not None else 2.0)
        self.gate_up_pow = float(getattr(opt, 'gate_up_pow', 3.0) if opt is not None else 3.0)
        self.gate_step_start = int(getattr(opt, 'gate_step_start', 2) if opt is not None else 2)
        self.ctrl_scale = float(getattr(opt, 'ctrl_scale', 1.0) if opt is not None else 1.0)
        self.ctrl_ports = getattr(opt, 'ctrl_ports', 'low') if opt is not None else 'low'
        self.ports = _ports_from_preset(self.ctrl_ports, n_res=9)

        # bias usage follows generator convention
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # --- Time embedding (optional but cheap) ---
        # We keep it very small; it mostly helps distinguish early vs late timesteps beyond scalar gating.
        self.time_mlp = nn.Sequential(
            nn.Linear(ngf, 4 * ngf),
            nn.LeakyReLU(0.2, True),
            nn.Linear(4 * ngf, 4 * ngf),
            nn.LeakyReLU(0.2, True),
        )
        self.to_ch0 = nn.Linear(4 * ngf, ngf)
        self.to_ch1 = nn.Linear(4 * ngf, 2 * ngf)
        self.to_ch2 = nn.Linear(4 * ngf, 4 * ngf)
        self.to_up1 = nn.Linear(4 * ngf, 2 * ngf)
        self.to_up2 = nn.Linear(4 * ngf, ngf)
        for lin in [self.to_ch0, self.to_ch1, self.to_ch2, self.to_up1, self.to_up2]:
            nn.init.zeros_(lin.bias)

        # --- Encoder ---
        self.conv0 = nn.Conv2d(cond_nc, ngf, 3, 1, 1, bias=use_bias)
        self.norm0 = norm_layer(ngf)

        self.down1 = nn.Conv2d(ngf, 2 * ngf, 3, 2, 1, bias=use_bias)
        self.norm1 = norm_layer(2 * ngf)

        self.down2 = nn.Conv2d(2 * ngf, 4 * ngf, 3, 2, 1, bias=use_bias)
        self.norm2 = norm_layer(4 * ngf)

        # --- Mid ---
        self.mid = nn.Sequential(
            _ResBlock(4 * ngf, norm_layer, use_bias),
            _ResBlock(4 * ngf, norm_layer, use_bias),
        )

        # --- Decoder ---
        self.up1_conv = nn.Conv2d(4 * ngf, 2 * ngf, 3, 1, 1, bias=use_bias)
        self.up1_norm = norm_layer(2 * ngf)

        self.up2_conv = nn.Conv2d(2 * ngf, ngf, 3, 1, 1, bias=use_bias)
        self.up2_norm = norm_layer(ngf)

        self.act = nn.LeakyReLU(0.2, True)

        # --- Zero-init heads per port ---
        heads = nn.ModuleDict()
        if 'down1' in self.ports:
            heads['down1'] = ZeroConv2d(2 * ngf, 2 * ngf)
        if 'down2' in self.ports:
            heads['down2'] = ZeroConv2d(4 * ngf, 4 * ngf)
        for i in range(9):
            name = f'res{i}'
            if name in self.ports:
                heads[name] = ZeroConv2d(4 * ngf, 4 * ngf)
        if 'up1' in self.ports:
            heads['up1'] = ZeroConv2d(2 * ngf, 2 * ngf)
        if 'up2' in self.ports:
            heads['up2'] = ZeroConv2d(ngf, ngf)

        self.heads = heads

    def _temb(self, time_idx: torch.Tensor) -> torch.Tensor:
        temb = get_timestep_embedding(time_idx, self.ngf)
        return self.time_mlp(temb)

    def forward(self, cond: torch.Tensor, time_idx: torch.Tensor) -> dict:
        """
        Args:
            cond: (B, cond_nc, H, W)
            time_idx: (B,) or (1,) integer timestep indices

        Returns:
            dict: {port_name: residual_tensor}
        """
        # time embedding (broadcasts across batch if time_idx has shape (1,))
        temb = self._temb(time_idx)

        # gates per group
        g_down = _schedule(time_idx, self.num_timesteps, self.gate_down, self.gate_down_pow, self.gate_step_start)
        g_mid = _schedule(time_idx, self.num_timesteps, self.gate_mid, self.gate_mid_pow, self.gate_step_start)
        g_up = _schedule(time_idx, self.num_timesteps, self.gate_up, self.gate_up_pow, self.gate_step_start)

        # --- encoder ---
        h0 = self.conv0(cond)
        h0 = h0 + self.to_ch0(temb).view(-1, self.ngf, 1, 1)
        h0 = self.act(self.norm0(h0))

        h1 = self.down1(h0)
        h1 = h1 + self.to_ch1(temb).view(-1, 2 * self.ngf, 1, 1)
        h1 = self.act(self.norm1(h1))

        h2 = self.down2(h1)
        h2 = h2 + self.to_ch2(temb).view(-1, 4 * self.ngf, 1, 1)
        h2 = self.act(self.norm2(h2))

        hmid = self.mid(h2)

        # --- decoder ---
        u1 = F.interpolate(hmid, scale_factor=2.0, mode='nearest')
        u1 = self.up1_conv(u1)
        u1 = u1 + self.to_up1(temb).view(-1, 2 * self.ngf, 1, 1)
        u1 = self.act(self.up1_norm(u1))

        u2 = F.interpolate(u1, scale_factor=2.0, mode='nearest')
        u2 = self.up2_conv(u2)
        u2 = u2 + self.to_up2(temb).view(-1, self.ngf, 1, 1)
        u2 = self.act(self.up2_norm(u2))

        # --- heads ---
        outs = {}
        for name, head in self.heads.items():
            if name == 'down1':
                outs[name] = self.ctrl_scale * g_down * head(h1)
            elif name == 'down2':
                outs[name] = self.ctrl_scale * g_down * head(hmid)
            elif name.startswith('res'):
                outs[name] = self.ctrl_scale * g_mid * head(hmid)
            elif name == 'up1':
                outs[name] = self.ctrl_scale * g_up * head(u1)
            elif name == 'up2':
                outs[name] = self.ctrl_scale * g_up * head(u2)
            else:
                raise ValueError(f'Unknown port name: {name}')

        return outs
