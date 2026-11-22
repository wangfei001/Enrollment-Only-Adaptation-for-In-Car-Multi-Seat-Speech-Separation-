#!/usr/bin/env python3
"""Seat-profile conditioned causal-compatible MVDR model.

Training uses full 3-second clips. The expensive registration encoder is
non-streaming; separation operators are causal-compatible so the same model can
be converted to inference with rolling state.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .profile_fast_resnet_model import (
    N_FFT, HOP, PROFILE_DIM, PROFILE_RAW_DIM, ProfileEncoder, ProfileRelation, stft_mag, estimate_rtf,
)
from .streaming_hamming_frontend import causal_stft, causal_istft


MASK_INPUT_CHANNELS = 23


def mask_input_features(X, rtf=None):
    """Shared causal mask-base features with optional RTF spatial evidence.

    IPD is referenced to microphone 0 and encoded as unit phasors.  Keeping
    this function in one place guarantees offline and streaming paths use
    exactly the same feature ordering and normalization.
    """
    mag = torch.log1p(X.abs())
    ref = X[:, :1]
    cross = X[:, 1:] * ref.conj()
    ipd = cross / (cross.abs() + 1e-6)
    if rtf is None:
        spatial = torch.zeros_like(mag[:, :1])
    else:
        # RTF is a one-way teacher for the mask.  Do not let mask losses update
        # the RTF branch through this spatial evidence path.
        rtf = rtf.detach()
        if rtf.ndim == 4:
            ones = torch.ones(
                X.shape[0], 1, X.shape[2], X.shape[3], device=X.device, dtype=rtf.dtype)
        else:
            ones = torch.ones(X.shape[0], 1, X.shape[2], device=X.device, dtype=rtf.dtype)
        h = torch.cat([
            ones,
            rtf.to(torch.complex64),
        ], 1)
        if h.ndim == 3:
            h = h[..., None]
        den = h.abs().square().sum(1).clamp_min(1e-6)
        y_bf = torch.einsum('bcft,bcft->bft', h.conj(), X.to(torch.complex64)) / den
        ratio = y_bf.abs().square() / (ref[:, 0].abs().square() + 1e-6)
        spatial = torch.log1p(ratio.clamp_min(0)).clamp(max=20.0)[:, None]
    return mag, torch.cat([ipd.real, ipd.imag], dim=1), spatial


def full_cov_rtf(X, mask):
    """[B,8,F,T], [B,2,F,T] -> covariance RTF [B,7,F]."""
    p = mask[:, 0].float().clamp(0, 1)
    xw = X.to(torch.complex64) * torch.sqrt(p)[:, None]
    obs = torch.einsum('bcft,bdft->bfcdt', xw, xw.conj())
    R = obs.sum(-1) / p.sum(-1)[:, :, None, None].clamp_min(1e-6)
    den = R[:, :, 0, 0].abs().clamp_min(1e-8)
    return (R[:, :, 1:, 0] / den[:, :, None]).permute(0, 2, 1)


class CausalConv2d(nn.Module):
    """2-D convolution with causal padding only on the time axis."""
    def __init__(self, cin, cout, kernel_size, stride=(1, 1), bias=False):
        super().__init__()
        kf, kt = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.pad_f = kf // 2
        self.pad_t = kt - 1
        self.conv = nn.Conv2d(cin, cout, (kf, kt), stride=stride, padding=0, bias=bias)

    def forward(self, x):
        x = F.pad(x, (self.pad_t, 0, self.pad_f, self.pad_f))
        return self.conv(x)


class CausalResBlock(nn.Module):
    def __init__(self, cin, cout, stride=(1, 1)):
        super().__init__()
        self.c1 = CausalConv2d(cin, cout, (3, 3), stride=stride)
        self.bn1 = nn.BatchNorm2d(cout)
        self.c2 = CausalConv2d(cout, cout, (3, 3), stride=(1, 1))
        self.bn2 = nn.BatchNorm2d(cout)
        self.act = nn.GELU()
        self.short = None
        if cin != cout or stride != (1, 1):
            self.short = nn.Sequential(
                CausalConv2d(cin, cout, (1, 1), stride=stride),
                nn.BatchNorm2d(cout),
            )

    def forward(self, x):
        s = x if self.short is None else self.short(x)
        h = self.act(self.bn1(self.c1(x)))
        h = self.bn2(self.c2(h))
        return self.act(h + s)


class CausalMixQuery(nn.Module):
    """Causal mixture query used by the RTF-candidate attention."""
    def __init__(self, dim=PROFILE_DIM):
        super().__init__()
        self.stem = nn.Sequential(
            CausalConv2d(24, 64, (5, 5)),
            nn.BatchNorm2d(64), nn.GELU(),
            CausalConv2d(64, 96, (3, 3)),
            nn.BatchNorm2d(96), nn.GELU(),
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(96), nn.Linear(96, dim), nn.GELU(), nn.LayerNorm(dim)
        )

    def forward(self, X):
        mag = torch.log1p(X.abs())
        feat = torch.cat([mag, X.real, X.imag], 1)
        h = self.stem(feat)
        h = h.mean(dim=2).transpose(1, 2)  # B,T,C
        return self.proj(h).transpose(1, 2)  # B,D,T


class RTFCandidateTokenizer(nn.Module):
    def __init__(self, dim=PROFILE_DIM):
        super().__init__()
        self.rtf = nn.Sequential(
            nn.Conv1d(14, 64, 5, padding=2), nn.GELU(),
            nn.Conv1d(64, 128, 3, padding=1), nn.GELU(),
        )
        self.rtf_out = nn.Linear(128, dim)
        self.geo = nn.Sequential(
            nn.Linear(4, 32), nn.GELU(), nn.Linear(32, dim),
        )
        self.spatial = nn.Sequential(
            nn.Linear(2, 32), nn.GELU(), nn.Linear(32, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, h, geometry, spatial):
        # h [B,K,7,F], geometry [B,K,4], spatial [B,K,2]
        b, k, _, _ = h.shape
        feat = torch.cat([h.real, h.imag], 2).reshape(b * k, 14, -1)
        feat = self.rtf(feat).mean(-1)
        feat = self.rtf_out(feat).reshape(b, k, -1)
        return self.norm(feat + self.geo(geometry.float()) + self.spatial(spatial.float()))


class RTFBankAttention(nn.Module):
    """Causal-compatible query pooling + RTF candidate attention."""
    def __init__(self, dim=PROFILE_DIM):
        super().__init__()
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.pool = nn.Linear(dim, 1)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))

    def forward(self, query, tokens, rtf_bank):
        # query [B,D,T], tokens [B,K,D], rtf_bank [B,K,7,F]
        q = self.q(query.transpose(1, 2))       # B,T,D
        q = F.normalize(q.float(), dim=-1)
        pool_alpha = torch.softmax(self.pool(q).squeeze(-1), dim=1)
        q_global = (pool_alpha[..., None] * q).sum(1)
        q_global = F.normalize(q_global, dim=-1)
        k = F.normalize(self.k(tokens).float(), dim=-1)
        score = torch.einsum('bd,bkd->bk', q_global, k) * self.logit_scale.exp().clamp(max=100.0)
        alpha = torch.softmax(score, dim=1)
        alpha_h = alpha.to(rtf_bank.dtype)
        h_sel = torch.einsum('bk,bkcf->bcf', alpha_h, rtf_bank)
        return h_sel, alpha, score


class CausalMaskNet(nn.Module):
    def __init__(self, dim=PROFILE_DIM, widths=(64, 128, 256, 256)):
        super().__init__()
        self.cond = nn.Sequential(
            nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim),
            nn.GELU(), nn.LayerNorm(dim),
        )
        self.input_norm = nn.BatchNorm2d(8)
        self.stem = nn.Sequential(
            CausalConv2d(MASK_INPUT_CHANNELS, widths[0], (7, 7)),
            nn.BatchNorm2d(widths[0]), nn.GELU(),
        )
        self.stages = nn.ModuleList([
            CausalResBlock(widths[0], widths[0], (1, 1)),
            CausalResBlock(widths[0], widths[1], (2, 1)),
            CausalResBlock(widths[1], widths[2], (2, 1)),
            CausalResBlock(widths[2], widths[3], (1, 1)),
        ])
        self.films = nn.ModuleList([nn.Linear(dim, 2 * c) for c in (widths[0], widths[0], widths[1], widths[2])])
        self.head = CausalConv2d(widths[-1], 2, (1, 1))
        nn.init.normal_(self.head.conv.weight, std=0.01)

    def forward(self, X, cond, rtf=None):
        mag, ipd, spatial = mask_input_features(X, rtf)
        feat = torch.cat([self.input_norm(mag), ipd, spatial], dim=1)
        h = self.stem(feat)
        for block, film in zip(self.stages, self.films):
            gb = film(cond.float())[:, :, None, None]
            g, b = gb.chunk(2, 1)
            h = block(h * (1 + g) + b)
        m = torch.softmax(self.head(h), dim=1)
        if m.shape[-2:] != X.shape[-2:]:
            m = F.interpolate(m, size=X.shape[-2:], mode='nearest')
        return m


class MaskRefiner(nn.Module):
    """Small bounded residual on top of the target/noise mask probabilities."""
    def __init__(self, dim=PROFILE_DIM):
        super().__init__()
        self.cond = nn.Linear(dim, 32)
        self.stem = nn.Sequential(
            CausalConv2d(51, 64, (3, 3)), nn.BatchNorm2d(64), nn.GELU(),
            CausalConv2d(64, 64, (3, 3)), nn.BatchNorm2d(64), nn.GELU(),
        )
        self.head = CausalConv2d(64, 1, (1, 1))
        nn.init.zeros_(self.head.conv.weight)
        self.gate = nn.Parameter(torch.tensor(-4.0))

    def forward(self, X, mask_base, h_mvdr, cond):
        # mask_base [B,2,F,T], X [B,8,F,T], h_mvdr [B,7,F]
        b, c, f, t = X.shape
        p = mask_base[:, 0].clamp(1e-4, 1 - 1e-4)
        hfull = torch.cat([torch.ones(b, 1, f, device=X.device, dtype=h_mvdr.dtype), h_mvdr], 1)
        den = hfull.abs().square().sum(1).clamp_min(1e-6)
        ybf = torch.einsum('bcf,bcft->bft', hfull.conj(), X) / den[:, :, None]
        cfeat = self.cond(cond.float())[:, :, None, None].expand(-1, -1, f, t)
        feat = torch.cat([
            p[:, None], X.real, X.imag,
            ybf.real[:, None], ybf.imag[:, None], cfeat,
        ], 1)
        delta = torch.tanh(self.head(self.stem(feat)))[:, 0]
        alpha = 0.10 * torch.sigmoid(self.gate)
        p_ref = torch.sigmoid(torch.logit(p) + alpha * delta)
        return torch.stack([p_ref, 1 - p_ref], 1)


class SimpleMaskRefiner(nn.Module):
    """Bounded causal time/frequency residual using mask_base only."""
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv2d(1, hidden, (3, 3)),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            CausalConv2d(hidden, hidden, (3, 3)),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            CausalConv2d(hidden, 1, (1, 1)),
        )
        nn.init.zeros_(self.net[-1].conv.weight)
        self.scale = nn.Parameter(torch.tensor(-2.0))

    def forward(self, mask_base):
        p = mask_base[:, 0].float().clamp(1e-4, 1 - 1e-4)
        delta = torch.tanh(self.net(p[:, None])[:, 0])
        alpha = torch.sigmoid(self.scale)
        p_ref = torch.sigmoid(torch.logit(p) + alpha * delta)
        return torch.stack([p_ref, 1.0 - p_ref], 1)


class CausalRTFRefiner(nn.Module):
    """Refine the selected registration RTF using the predicted target mask.

    Training uses full-utterance statistics. The same parameters can be used
    with recursive target covariance in streaming inference.
    """
    def __init__(self, dim=PROFILE_DIM):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim + 2, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1)
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -4.0)

    def forward(self, X, mask, h_sel, cond):
        target_power = mask.float()[:, 0].clamp(0, 1)
        Xw = X * torch.sqrt(target_power)[:, None]
        R_s = torch.einsum('bcft,bdft->bfcd', Xw, Xw.conj()).permute(0, 2, 3, 1)
        denom = R_s[:, 0, 0].abs().clamp_min(1e-6)
        h_hat = R_s[:, 1:, 0] / denom[:, None, :]
        delta = (h_hat - h_sel).abs().mean((-1, -2))
        scale = h_sel.abs().mean((-1, -2)).clamp_min(1e-6)
        disagreement = delta / scale
        confidence = target_power.mean((-1, -2)).clamp(0, 1)
        gate_in = torch.cat([cond.float(), confidence.float()[:, None], disagreement.float()[:, None]], -1)
        beta = 0.5 * torch.sigmoid(self.gate(gate_in))
        h_ref = h_sel + beta.squeeze(-1)[:, None, None] * (h_hat - h_sel)
        return h_ref, beta, h_hat


class FullUtteranceMVDR(nn.Module):
    """Offline training MVDR. Streaming inference replaces mean with recursion."""
    def __init__(self, diag=1e-3):
        super().__init__()
        self.diag = diag

    def forward(self, X, mask, rtf):
        b, c, f, t = X.shape
        target_power = mask.float()[:, 0].clamp(0, 1)
        h = torch.cat([
            torch.ones(b, 1, f, device=X.device, dtype=X.dtype), rtf.to(X.dtype)
        ], 1)
        # Reconstruct the target spatial image, then estimate noise covariance
        # from the residual X - target_est. This is stable for soft power masks.
        target_est = h[..., None] * (target_power * X[:, 0])[:, None]
        noise = X - target_est
        R = torch.einsum('bcft,bdft->bfcd', noise, noise.conj()) / max(t, 1)
        eye = torch.eye(c, device=X.device, dtype=X.dtype)[None, None]
        R = R + self.diag * eye
        rhs = h.permute(0, 2, 1)
        w = torch.linalg.solve(R, rhs)
        den = torch.einsum('bfc,bfc->bf', w.conj(), rhs).abs().clamp_min(1e-6)
        w = w / den[:, :, None]
        return torch.einsum('bfc,bcft->bft', w.conj(), X)


class RecursiveMVDRState:
    """Inference helper: recursive noise covariance and single-RTF MVDR."""
    def __init__(self, channels=8, diag=1e-3, decay=0.98):
        self.channels = channels
        self.diag = diag
        self.decay = decay
        self.Rn = None
        self.Wn = None

    def reset(self, batch, freq, device, dtype=torch.complex64):
        self.Rn = torch.zeros(batch, freq, self.channels, self.channels, device=device, dtype=dtype)
        self.Wn = torch.zeros(batch, freq, 1, 1, device=device, dtype=torch.float32)

    def step(self, x_t, mask_t, rtf_t):
        # x_t [B,C,F], mask_t [B,F] is a target-power mask, rtf_t [B,7,F].
        if self.Rn is None:
            self.reset(x_t.shape[0], x_t.shape[2], x_t.device, x_t.dtype)
        p = mask_t.float().clamp(0, 1)
        b, c, f = x_t.shape
        h = torch.cat([
            torch.ones(b, 1, f, device=x_t.device, dtype=x_t.dtype), rtf_t
        ], 1)
        target_est = h * (p * x_t[:, 0]).unsqueeze(1)
        noise = x_t - target_est
        obs = noise[..., :, None] * noise[..., None, :].conj()
        self.Rn = self.decay * self.Rn + (1 - self.decay) * obs
        rhs = h.permute(0, 2, 1)
        w = torch.linalg.solve(self.Rn + self.diag * torch.eye(c, device=x_t.device, dtype=x_t.dtype)[None, None], rhs)
        den = torch.einsum('bfc,bfc->bf', w.conj(), rhs).abs().clamp_min(1e-6)
        w = w / den[:, :, None]
        return torch.einsum('bfc,bcf->bf', w.conj(), x_t)


class ContinuousRTFPredictor(nn.Module):
    """Refine the attention-fused RTF with a continuous frequency residual."""
    def __init__(self, dim=PROFILE_DIM):
        super().__init__()
        self.cond = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 32), nn.GELU())
        self.net = nn.Sequential(
            nn.Conv1d(46, 64, 5, padding=2), nn.GELU(),
            nn.Conv1d(64, 64, 3, padding=1), nn.GELU(),
            nn.Conv1d(64, 14, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, query, h_fusion):
        b, _, nf = h_fusion.shape
        cond = self.cond(query.mean(-1)).unsqueeze(-1).expand(-1, -1, nf)
        feat = torch.cat([h_fusion.real, h_fusion.imag], 1)
        residual = self.net(torch.cat([feat, cond], 1))
        delta = residual[:, :7] + 1j * residual[:, 7:]
        return h_fusion + delta


class RTFNet(nn.Module):
    """Single-frame causal RTF estimator: features -> 7 complex RTFs."""
    def __init__(self, dim=PROFILE_DIM, hidden=256):
        super().__init__()
        # h_cov 14 + h_prior 14
        self.inp = nn.Conv1d(28, hidden, 1)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden, hidden, 5, padding=2),
                nn.GroupNorm(8, hidden),
                nn.GELU(),
            )
            for _ in range(2)
        ])
        self.film = nn.Linear(dim, 2 * hidden)
        self.head = nn.Conv1d(hidden, 14, 1)
        self.residual_scale = nn.Parameter(torch.tensor(0.0))
        # Small random residual head; final RTF is anchored at h_cov.
        nn.init.normal_(self.head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, h_cov, h_prior, cond):
        b, _, f, t = h_cov.shape
        if h_prior.ndim == 3:
            h_prior = h_prior[..., None]
        h_prior = h_prior.expand(b, 7, f, t)
        feat = torch.cat([
            h_cov.real, h_cov.imag,
            h_prior.real, h_prior.imag,
        ], 1)
        feat = feat.permute(0, 3, 1, 2).reshape(b * t, feat.shape[1], f)
        h = self.inp(feat)
        film = self.film(cond.float()).reshape(b, 2, 1, -1)
        film = film.expand(b, 2, t, -1).reshape(b * t, 2, -1)
        scale, shift = film[:, 0, :, None], film[:, 1, :, None]
        for block in self.blocks:
            h = block(h)
            h = h * (1.0 + scale) + shift
        out = self.head(h).reshape(b, t, 14, f).permute(0, 2, 3, 1).float()
        return torch.complex(out[:, :7], out[:, 7:])


def complex_rtf_gate(h_res, scale):
    """Bounded complex multiplicative correction for an RTF estimate."""
    delta = torch.complex(torch.tanh(h_res.real), torch.tanh(h_res.imag))
    return torch.exp(scale * delta)


class AdaptiveRTFEstimator(nn.Module):
    """Chunk-causal prefix RTF state with increasing evidence confidence."""
    def __init__(self, decay=0.98, tau=5.0, beta_max=0.50, stride=10):
        super().__init__()
        self.decay = decay
        self.tau = tau
        self.beta_max = beta_max
        self.stride = stride

    def forward(self, X, mask, h_prior):
        # X [B,C,F,T], mask [B,2,F,T], h_prior [B,7,F]
        b, c, f, t = X.shape
        R = torch.zeros(b, f, c, c, device=X.device, dtype=torch.complex64)
        evidence = torch.zeros(b, device=X.device, dtype=torch.float32)
        h_state = h_prior.to(torch.complex64)
        m = torch.complex(mask[:, 0].float(), mask[:, 1].float()).abs().square().clamp(0, 1)
        outs = []
        for start in range(0, t, self.stride):
            end = min(t, start + self.stride)
            mt = m[:, :, start:end]
            xt = X[:, :, :, start:end].to(torch.complex64)  # B,C,F,K
            xw = xt * torch.sqrt(mt)[:, None]
            obs = torch.einsum('bcft,bdft->bfcd', xw, xw.conj())
            R = self.decay * R + obs
            den = R[:, :, 0, 0].abs() + 1e-6
            h_obs = R[:, :, 1:, 0].permute(0, 2, 1) / den[:, None, :]
            evidence = self.decay * evidence + mt.sum(-1).mean(-1)
            beta = (1.0 - torch.exp(-evidence / self.tau)).clamp(0, self.beta_max)
            h_state = (1 - beta[:, None, None]) * h_state + beta[:, None, None] * h_obs
            outs.append(h_state)
        h_seq = torch.stack(outs, 1)  # B,num_chunks,7,F
        return h_seq, h_state


class GatedConditionFusion(nn.Module):
    """Gated blend of a seat/task embedding and the seat-relation embedding."""
    def __init__(self, dim=PROFILE_DIM):
        super().__init__()
        self.gate = nn.Sequential(
            nn.LayerNorm(2 * dim),
            nn.Linear(2 * dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, task_emb, relation):
        g = torch.sigmoid(self.gate(torch.cat([task_emb, relation], -1)))
        return F.normalize(g * task_emb + (1.0 - g) * relation, dim=-1)


class RegistrationRTFEncoder(nn.Module):
    """Encode each registration RTF into an independent direction token."""
    def __init__(self, dim=PROFILE_DIM, hidden=192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(14, hidden, 5, padding=2),
            nn.GroupNorm(8, hidden), nn.GELU(),
            nn.Conv1d(hidden, dim, 5, padding=2),
            nn.GroupNorm(8, dim), nn.GELU(),
        )
        self.out = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim),
        )

    def forward(self, rtf):
        # rtf [B,4,K,7,F]; preserve every registration position.
        b, nseat, k, _, nf = rtf.shape
        feat = torch.cat([rtf.real, rtf.imag], 1)
        feat = feat.reshape(b * nseat * k, 14, nf)
        h = self.net(feat).mean(-1)
        h = self.out(h.reshape(b * nseat * k, -1)).reshape(b, nseat, k, -1)
        return F.normalize(h, dim=-1)


class RegistrationTokenEncoder(nn.Module):
    """Set Transformer encoder for five ordered registration tokens.

    A CLS query attends over the five position tokens. Learned slot embeddings
    preserve which registration position each token came from, while attention
    lets the encoder learn cross-position interactions without averaging.
    """
    def __init__(self, dim=PROFILE_DIM, slots=5, layers=3, heads=8,
                 ffn_dim=4 * PROFILE_DIM, dropout=0.1):
        super().__init__()
        self.slots = slots
        self.slot_embed = nn.Parameter(torch.empty(1, slots, dim))
        self.cls = nn.Parameter(torch.empty(1, 1, dim))
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=ffn_dim,
            dropout=dropout, activation='gelu', batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.out = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)
        nn.init.normal_(self.slot_embed, std=0.02)
        nn.init.normal_(self.cls, std=0.02)

    def forward(self, tokens):
        # tokens [B,4,K,dim]
        b, nseat, k, dim = tokens.shape
        if k != self.slots:
            raise ValueError(f'expected {self.slots} registration tokens, got {k}')
        x = tokens.reshape(b * nseat, k, dim) + self.slot_embed
        cls = self.cls.expand(b * nseat, -1, -1)
        h = self.encoder(torch.cat([cls, x], dim=1))[:, 0]
        h = self.norm(h + self.out(h)).reshape(b, nseat, dim)
        return F.normalize(h, dim=-1)


class SeatProfileMVDR(nn.Module):
    def __init__(self, dim=PROFILE_DIM, rtf_decay=0.99, rtf_tau=5.0, rtf_stride=10):
        super().__init__()
        self.register_buffer('win', torch.hann_window(N_FFT), persistent=False)
        self.profile = ProfileEncoder(dim=dim, raw_dim=PROFILE_RAW_DIM)
        self.pos_proj = nn.Sequential(
            nn.Linear(PROFILE_RAW_DIM, PROFILE_RAW_DIM), nn.GELU(),
            nn.Linear(PROFILE_RAW_DIM, PROFILE_RAW_DIM), nn.LayerNorm(PROFILE_RAW_DIM)
        )
        self.prof_proj = nn.Sequential(
            nn.Linear(PROFILE_RAW_DIM, dim), nn.GELU(), nn.LayerNorm(dim)
        )
        self.prof_encoder = RegistrationTokenEncoder(dim)
        self.rtf_encoder = RegistrationRTFEncoder(dim)
        self.dir_encoder = RegistrationTokenEncoder(dim)
        self.relation_proj = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.GELU(), nn.LayerNorm(dim)
        )
        self.relation = ProfileRelation(dim=dim)
        self.mask_net = CausalMaskNet(dim)
        self.rtf_net = RTFNet(dim)
        self.mask_cond_proj = nn.Sequential(
            nn.Linear(dim * 3, dim), nn.GELU(), nn.LayerNorm(dim)
        )
        self.rtf_cond_proj = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.GELU(), nn.LayerNorm(dim)
        )
        self.mask_refiner = SimpleMaskRefiner()
        self.mvdr = FullUtteranceMVDR()

    def encode_profiles(self, enroll):
        # enroll [B,4,K,8,T]
        if enroll.ndim == 4:
            enroll = enroll[:, :, None]
        b, nseat, k, c, t = enroll.shape
        raw = self.profile(enroll.reshape(b * nseat * k, c, t)).reshape(b, nseat, k, -1)
        projected = self.pos_proj(raw)
        z_pos = F.normalize(projected, dim=-1)
        u = F.normalize(self.prof_proj(projected), dim=-1)
        z_prof = self.prof_encoder(u)
        return z_pos, z_prof

    def encode_condition(self, z_pos, z_prof, rtf_all):
        """Build independent profile, direction, and seat-relation conditions."""
        v = self.rtf_encoder(rtf_all)
        z_dir = self.dir_encoder(v)
        relation_input = self.relation_proj(torch.cat([z_prof, z_dir], dim=-1))
        r, seat_logits, pair_logits = self.relation(relation_input)
        return {
            'z_pos': z_pos,
            'z_prof': z_prof,
            'z_dir': z_dir,
            'r': r,
            'seat_logits': seat_logits,
            'pair_logits': pair_logits,
            'rtf_all': rtf_all,
        }

    def encode_registration(self, enroll):
        """Encode one vehicle registration once for all mixture scenes.

        The result is scene-independent and can be reused by every mixture in
        the same vehicle group.  Gradients remain attached so the profile
        branch is still jointly optimized with separation.
        """
        # enroll [B,4,K,8,T]
        if enroll.ndim == 4:
            enroll = enroll[:, :, None]
        b, nseat, k, c, t = enroll.shape
        z_pos, z_prof = self.encode_profiles(enroll)
        rtf_all = estimate_rtf(enroll.reshape(b * nseat * k, c, t), self.win)
        rtf_all = rtf_all.reshape(b, nseat, k, 7, -1)
        return self.encode_condition(z_pos, z_prof, rtf_all)

    def forward_from_registration(self, mix, target_seat, geometry,
                                  registration, rtf_mode='attention'):
        """Separate scenes using a cached vehicle registration encoding."""
        b = mix.shape[0]
        reg_b = registration['rtf_all'].shape[0]
        k = registration['rtf_all'].shape[2]
        if reg_b == 1:
            # One cached vehicle registration can serve every scene in the group.
            z_prof = registration['z_prof'][0, target_seat]
            z_dir = registration['z_dir'][0, target_seat]
            r = registration['r'][0, target_seat]
            h_bank = registration['rtf_all'][0, target_seat]
            geo_bank = geometry[target_seat] if geometry.ndim == 3 else geometry[0, target_seat]
        else:
            if reg_b != b:
                raise ValueError(f'registration batch {reg_b} does not match mixture batch {b}')
            idx = torch.arange(b, device=mix.device)
            z_prof = registration['z_prof'][idx, target_seat]
            z_dir = registration['z_dir'][idx, target_seat]
            r = registration['r'][idx, target_seat]
            h_bank = registration['rtf_all'][idx, target_seat]
            geo_bank = geometry[idx, target_seat]
        mask_cond = self.mask_cond_proj(torch.cat([z_prof, z_dir, r], dim=-1))
        rtf_cond = self.rtf_cond_proj(torch.cat([z_dir, r], dim=-1))

        X = causal_stft(mix, self.win)
        # RTF is estimated from the unmasked mixture covariance first. The
        # predicted mask never participates in RTF estimation.
        h_sel = h_bank[:, 0]
        uniform_mask = torch.ones(
            X.shape[0], 2, X.shape[2], X.shape[3], device=X.device, dtype=X.real.dtype)
        h_cov = full_cov_rtf(X, uniform_mask)
        geo_center = (geo_bank[0].unsqueeze(0).expand(b, -1)
                      if geo_bank.ndim == 2 else geo_bank[:, 0])
        h_cov_t = h_cov[..., None]
        h_res_t = self.rtf_net(h_cov_t, h_sel, rtf_cond)
        scale = torch.sigmoid(self.rtf_net.residual_scale)
        h_ref_t = h_cov_t * complex_rtf_gate(h_res_t, scale)
        h_ref = h_ref_t.mean(-1)
        h_pred = h_ref
        h_hat = h_cov
        h_mvdr = h_ref
        # RTF guides the mask through the beamformer target-evidence feature.
        mask_base = self.mask_net(X, mask_cond, h_ref)
        mask_ref = self.mask_refiner(mask_base)
        Ym = self.mvdr(X, mask_ref, h_mvdr)
        alpha = torch.ones(b, 1, device=mix.device, dtype=X.real.dtype)
        beta = torch.zeros(b, 1, device=mix.device, dtype=X.real.dtype)
        rtf_logits = torch.zeros(b, 1, device=mix.device, dtype=X.real.dtype)

        Ym_mask = mask_ref[:, 0] * X[:, 0]
        y = causal_istft(Ym, mix.shape[-1], self.win)
        y_mask = causal_istft(Ym_mask, mix.shape[-1], self.win)
        return {
            'y': y, 'y_mvdr': y, 'y_mask': y_mask,
            'mask': mask_ref, 'mask_base': mask_base, 'mask_ref': mask_ref, 'alpha': alpha, 'rtf_logits': rtf_logits,
            'h_sel': h_sel, 'h_pred': h_pred, 'h_ref': h_ref, 'beta': beta, 'h_hat': h_hat, 'h_mvdr': h_mvdr,
            'z_pos': registration.get('z_pos'), 'z_prof': registration.get('z_prof'),
            'z_dir': registration.get('z_dir'), 'r': registration.get('r'),
            'seat_logits': registration.get('seat_logits'),
            'pair_logits': registration.get('pair_logits'),
        }

    def forward(self, enroll, mix, target_seat, geometry, rtf_mode='attention'):
        """Compatibility wrapper for scene-level callers."""
        registration = self.encode_registration(enroll)
        return self.forward_from_registration(
            mix, target_seat, geometry, registration, rtf_mode=rtf_mode
        )
