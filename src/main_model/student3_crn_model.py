#!/usr/bin/env python3
"""Student 3: causal CRN with a two-layer recurrent bottleneck."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .profile_mvdr_stream_model import CausalConv2d, CausalResBlock
from .mvdr_stream_model import _causal_conv_step, _resblock_step


class Student3StreamingState:
    def __init__(self, batch: int, device):
        self.conv = {}
        self.gru = {}
        self.batch = int(batch)
        self.device = device


class CRULayer(nn.Module):
    """Unidirectional recurrent bottleneck over time, per frequency bin."""

    def __init__(self, channels: int, hidden: int):
        super().__init__()
        self.rnn = nn.GRU(channels, hidden, num_layers=1, batch_first=True)
        self.out = nn.Linear(hidden, channels)

    def forward(self, h, state, key):
        b, c, f, t = h.shape
        seq = h.permute(0, 2, 3, 1).reshape(b * f, t, c)
        h0 = state.gru.get(key) if state is not None else None
        with torch.autocast('cuda', enabled=False):
            y, hn = self.rnn(seq.float(), h0)
            y = self.out(y).to(dtype=h.dtype)
        if state is not None:
            state.gru[key] = hn.detach()
        y = y.reshape(b, f, t, c).permute(0, 3, 1, 2)
        return h + y


class Student3CRN(nn.Module):
    def __init__(self, cond_dim=512, widths=(64, 128, 256, 256),
                 gru_hidden=128, residual_bound=0.5):
        super().__init__()
        self.residual_bound = float(residual_bound)
        self.input_norm = nn.BatchNorm2d(8)
        self.residual_norm = nn.BatchNorm2d(1)

        self.stem = nn.Sequential(
            CausalConv2d(25, widths[0], (7, 7)),
            nn.BatchNorm2d(widths[0]), nn.GELU(),
        )
        self.enc0 = CausalResBlock(widths[0], widths[0], (1, 1))
        self.enc1 = CausalResBlock(widths[0], widths[1], (2, 1))
        self.enc2 = CausalResBlock(widths[1], widths[2], (2, 1))
        self.enc3 = CausalResBlock(widths[2], widths[3], (1, 1))

        self.film = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, widths[3] * 2),
        )
        self.cru1 = CRULayer(widths[3], gru_hidden)
        self.cru2 = CRULayer(widths[3], gru_hidden)

        self.dec2 = CausalResBlock(widths[3] + widths[1], widths[1], (1, 1))
        self.dec1 = CausalResBlock(widths[1] + widths[0], widths[0], (1, 1))
        self.head = CausalConv2d(widths[0], 2, (1, 1))
        nn.init.zeros_(self.head.conv.weight)
        if self.head.conv.bias is not None:
            nn.init.zeros_(self.head.conv.bias)

    def load_teacher_backbone(self, state_dict, verbose=True):
        dst = self.state_dict()
        copied = 0
        for key, value in state_dict.items():
            if key == 'stem.0.conv.weight':
                expanded = torch.zeros_like(dst[key])
                expanded[:, :22] = value[:, :22]
                dst[key] = expanded
                copied += 1
                continue
            if key.startswith('head.') or key.startswith('films.'):
                continue
            mapped = key
            for src, dst_prefix in (('stages.0.', 'enc0.'), ('stages.1.', 'enc1.'),
                                    ('stages.2.', 'enc2.'), ('stages.3.', 'enc3.')):
                if key.startswith(src):
                    mapped = dst_prefix + key[len(src):]
                    break
            if mapped in dst and dst[mapped].shape == value.shape:
                dst[mapped] = value
                copied += 1
        self.load_state_dict(dst)
        if verbose:
            print(f'STUDENT3_INIT_FROM_MASK copied={copied}', flush=True)

    def train(self, mode=True):
        super().train(mode)
        if mode:
            for module in self.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
            self.residual_norm.train(True)
        return self

    def _features(self, X, Y_mvdr):
        ref = X[:, :1]
        mag = torch.log1p(X.abs())
        cross = X[:, 1:] * ref.conj()
        den = cross.abs() + 1e-6
        ipd = torch.cat([cross.real / den, cross.imag / den], dim=1)

        residual = Y_mvdr - X[:, 0]
        rmag = torch.log1p(residual.abs())[:, None]
        rr = residual * X[:, 0].conj()
        rden = residual.abs() * X[:, 0].abs() + 1e-6
        rphase = torch.cat([
            (rr.real / rden)[:, None],
            (rr.imag / rden)[:, None],
        ], dim=1)
        return torch.cat([
            self.input_norm(mag), ipd,
            self.residual_norm(rmag), rphase,
        ], dim=1)

    def _conv_step(self, module, x, state, key):
        return _causal_conv_step(module, x, state, key) if state is not None else module(x)

    def _res_step(self, module, x, state, key):
        return _resblock_step(module, x, state, key) if state is not None else module(x)

    def forward(self, X, Y_mvdr, cond, state=None):
        h = self._features(X, Y_mvdr)
        h = self._conv_step(self.stem[0], h, state, 's3.stem')
        h = self.stem[2](self.stem[1](h))
        s0 = self._res_step(self.enc0, h, state, 's3.enc0')
        s1 = self._res_step(self.enc1, s0, state, 's3.enc1')
        s2 = self._res_step(self.enc2, s1, state, 's3.enc2')
        s3 = self._res_step(self.enc3, s2, state, 's3.enc3')

        gb = self.film(cond.float())[:, :, None, None]
        g, b = gb.chunk(2, 1)
        s3 = s3 * (1 + g) + b
        h = self.cru1(s3, state, 's3.cru1')
        h = self.cru2(h, state, 's3.cru2')

        d2 = F.interpolate(h, size=s1.shape[-2:], mode='nearest')
        d2 = torch.cat([d2, s1], dim=1)
        d2 = self._res_step(self.dec2, d2, state, 's3.dec2')
        d1 = F.interpolate(d2, size=s0.shape[-2:], mode='nearest')
        d1 = torch.cat([d1, s0], dim=1)
        d1 = self._res_step(self.dec1, d1, state, 's3.dec1')

        out = self._conv_step(self.head, d1, state, 's3.head')
        if out.shape[-2:] != X.shape[-2:]:
            out = F.interpolate(out, size=X.shape[-2:], mode='nearest')
        real = self.residual_bound * torch.tanh(out[:, 0])
        imag = self.residual_bound * torch.tanh(out[:, 1])
        return torch.complex(real.float(), imag.float())[:, None]

    def step(self, X, Y_mvdr, cond, state):
        return self.forward(X, Y_mvdr, cond, state=state)
