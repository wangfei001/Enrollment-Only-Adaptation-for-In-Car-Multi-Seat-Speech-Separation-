#!/usr/bin/env python3
"""MVDR stage-1 streaming fine-tune.

Train on full utterances while every separation block after STFT uses only
current/past frames. Registration is precomputed per vehicle and may be
fine-tuned. RTF attention, target covariance, and noise covariance use independent causal EMA windows
(500 ms, 500 ms, and 750 ms by default) to track non-stationary speech/noise.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from .profile_mvdr_stream_model import (
    SeatProfileMVDR, N_FFT, HOP, complex_rtf_gate, mask_input_features,
)
from .profile_fast_resnet_model import (
    supcon_loss, seat_classification_loss, pair_relation_loss,
    make_pair_targets, wrong_seat_hinge,
)
from .streaming_hamming_frontend import (
    causal_stft, causal_istft, StreamingCausalSTFT, StreamingOLAISTFT,
)

FS = 16000

def chunks(n, size):
    for start in range(0, n, size):
        yield start, min(n, start + size)


def aligned_profile_features(x, win):
    """Profile input features using the same zero-padded causal STFT."""
    X = causal_stft(x, win)
    mag = torch.log1p(X.abs())
    ipd = X[:, :-1] * X[:, 1:].conj()
    ipd = ipd / (ipd.abs() + 1e-6)
    ild = mag[:, 1:] - mag[:, :1]
    ref = X[:, :1]
    cross = (X[:, 1:] * ref.conj()).mean(-1)
    power = ref.abs().square().mean(-1).clamp_min(1e-6)
    rtf = cross / power
    rtf = torch.cat([rtf.real, rtf.imag], 1).flatten(1)
    return torch.cat([mag, ipd.real, ipd.imag, ild], dim=1), rtf


def aligned_estimate_rtf(x, win):
    X = causal_stft(x, win)
    ref = X[:, :1]
    cross = (X[:, 1:] * ref.conj()).mean(-1)
    power = ref.abs().square().mean(-1).clamp_min(1e-6)
    return cross / power


def weighted_rtf_loss(pred, target, weight):
    pred = pred.to(torch.complex64)
    target = target.to(torch.complex64)
    weight = weight.float().clamp_min(0)
    if pred.ndim == 4:
        target = target[..., None]
        weight = weight[..., None]
        reduce_dims = (-3, -2)
    else:
        reduce_dims = (-2, -1)
    num = (pred.conj() * target).real * weight[:, None, :]
    num = num.sum(reduce_dims)
    pe = (pred.abs().square() * weight[:, None, :]).sum(reduce_dims)
    te = (target.abs().square() * weight[:, None, :]).sum(reduce_dims)
    den = torch.sqrt((pe * te).clamp_min(1e-12))
    return (1.0 - num / den).clamp(0.0, 2.0).mean()


def weighted_rtf_logmag_loss(pred, target, weight):
    """Stable magnitude auxiliary loss for the final RTF."""
    pred = pred.to(torch.complex64)
    target = target.to(torch.complex64)
    weight = weight.float().clamp_min(0)
    if pred.ndim == 4:
        target = target[..., None]
        weight = weight[..., None]
        sum_dims = (-1, -2, -3)
    else:
        sum_dims = (-1, -2)
    diff = (torch.log1p(pred.abs()) - torch.log1p(target.abs())).square()
    w = weight[:, None, :]
    den = w.expand_as(diff).sum(sum_dims).clamp_min(1e-8)
    return ((diff * w).sum(sum_dims) / den).mean()


def causal_oracle_power_mask(mixture, target, window):
    """Ideal target-power mask in the causal Hamming domain."""
    X0 = causal_stft(mixture[:, 0], window)[:, 0]
    S0 = causal_stft(target[:, None], window)[:, 0]
    p = (S0.abs() / X0.abs().clamp_min(1e-6)).square().clamp(0, 1)
    return torch.stack([p, 1.0 - p], 1)


def causal_mask_aux_loss(pred_mask, mixture, target, window):
    """Weighted MSE to the causal target-power-ratio mask."""
    X0 = causal_stft(mixture[:, 0], window)[:, 0]
    S0 = causal_stft(target[:, None], window)[:, 0]
    target_power = S0.abs().square()
    mixture_power = X0.abs().square()
    ideal = (target_power / mixture_power.clamp_min(1e-6)).clamp(0, 1)
    target_weight = target_power / target_power.mean((-1, -2), keepdim=True).clamp_min(1e-8)
    mixture_weight = mixture_power / mixture_power.mean((-1, -2), keepdim=True).clamp_min(1e-8)
    weight = 0.5 * target_weight + 0.5 * mixture_weight
    with torch.autocast(device_type='cuda', enabled=False):
        pred = pred_mask[:, 0].float().clamp(0, 1)
        mse = (pred - ideal.float()).square()
        return (mse * weight.float()).sum() / weight.float().sum().clamp_min(1e-8)


def causal_oracle_cov_rtf(mixture, target, window):
    """Oracle-mask covariance RTF in the current causal STFT domain."""
    X = causal_stft(mixture, window)
    p = causal_oracle_power_mask(mixture, target, window)[:, 0].float()
    xw = X.to(torch.complex64) * torch.sqrt(p)[:, None]
    obs = torch.einsum('bcft,bdft->bfcdt', xw, xw.conj())
    R = obs.sum(-1) / p.sum(-1)[:, :, None, None].clamp_min(1e-6)
    den = R[:, :, 0, 0].abs().clamp_min(1e-8)
    rtf = (R[:, :, 1:, 0] / den[:, :, None]).permute(0, 2, 1)
    S0 = causal_stft(target[:, None], window)[:, 0]
    weight = S0.abs().square().mean(-1)
    return rtf, weight


class AlignedSeatProfileMVDR(SeatProfileMVDR):
    """SeatProfileMVDR with zero-padded causal STFT in profile and RTF paths."""

    def encode_profiles(self, enroll):
        if enroll.ndim == 4:
            enroll = enroll[:, :, None]
        b, nseat, k, c, t = enroll.shape
        flat = enroll.reshape(b * nseat * k, c, t)
        f, rtf = aligned_profile_features(flat, self.profile.win)
        h = self.profile.stages(self.profile.stem(f))
        h = torch.cat([h.mean(dim=(2, 3)), h.std(dim=(2, 3)).clamp_min(1e-5)], dim=1)
        raw = self.profile.out(h) + self.profile.rtf_proj(rtf)
        raw = raw.reshape(b, nseat, k, -1)
        projected = self.pos_proj(raw)
        z_pos = F.normalize(projected, dim=-1)
        u = F.normalize(self.prof_proj(projected), dim=-1)
        z_prof = self.prof_encoder(u)
        return z_pos, z_prof

    def encode_registration(self, enroll):
        if enroll.ndim == 4:
            enroll = enroll[:, :, None]
        b, nseat, k, c, t = enroll.shape
        z_pos, z_prof = self.encode_profiles(enroll)
        rtf_all = aligned_estimate_rtf(enroll.reshape(b * nseat * k, c, t), self.win)
        rtf_all = rtf_all.reshape(b, nseat, k, 7, -1)
        return self.encode_condition(z_pos, z_prof, rtf_all)


def load_compatible_state_dict(model, state_dict):
    current = model.state_dict()
    compatible = {}
    skipped = []
    for key, value in state_dict.items():
        target_key = key.replace('seat_proj.', 'prof_proj.', 1) if key.startswith('seat_proj.') else key
        if target_key in current and current[target_key].shape == value.shape:
            compatible[target_key] = value
        elif (target_key.endswith('mask_net.stem.0.conv.weight')
              and value.ndim == 4
              and target_key in current
              and value.shape[0] == current[target_key].shape[0]
              and value.shape[1] == 22
              and current[target_key].shape[1] == 23
              and value.shape[2:] == current[target_key].shape[2:]):
            expanded = torch.zeros_like(current[target_key])
            expanded[:, :22] = value.to(device=expanded.device, dtype=expanded.dtype)
            compatible[target_key] = expanded
            print('CHECKPOINT_EXPAND_MASK_INPUT', tuple(value.shape), '->', tuple(expanded.shape), flush=True)
        else:
            skipped.append(key)
    model.load_state_dict(compatible, strict=False)
    if skipped:
        print('CHECKPOINT_SKIP', len(skipped), skipped[:8], flush=True)
    return skipped


def _is_bn(module):
    return isinstance(module, nn.modules.batchnorm._BatchNorm)


def _set_stream_train_mode(model):
    """Train neural weights while keeping BatchNorm in inference mode."""
    model.train()
    for module in model.modules():
        if _is_bn(module):
            module.eval()


def _reset_bn_stats(model):
    for module in model.modules():
        if not _is_bn(module):
            continue
        if module.running_mean is not None:
            module.running_mean.zero_()
        if module.running_var is not None:
            module.running_var.fill_(1.0)
        if module.num_batches_tracked is not None:
            module.num_batches_tracked.zero_()


def _set_bn_calibration_mode(model):
    """Disable dropout while allowing BatchNorm to update running stats."""
    model.eval()
    for module in model.modules():
        if _is_bn(module):
            module.train()


def _causal_conv_step(conv, x, state, key):
    """Run a CausalConv2d on a time chunk while preserving convolution history.

    The cache stores raw inputs, so the concatenation plus an un-padded kernel
    produces only outputs for the current chunk. Frequency padding is identical
    to training. Temporal stride must remain one.
    """
    assert conv.conv.stride[1] == 1, 'streaming requires temporal stride=1'
    kt = conv.conv.kernel_size[1]
    hist = state.conv.get(key)
    if hist is None:
        hist = x.new_zeros(x.shape[0], x.shape[1], x.shape[2], kt - 1)
    elif hist.shape[0] != x.shape[0] or hist.shape[1] != x.shape[1] or hist.shape[2] != x.shape[2]:
        raise RuntimeError(f'conv cache mismatch {key}: {tuple(hist.shape)} vs {tuple(x.shape)}')
    inp = torch.cat([hist, x], dim=-1)
    padded = F.pad(inp, (0, 0, conv.pad_f, conv.pad_f))
    y = conv.conv(padded)
    state.conv[key] = inp[..., -(kt - 1):].detach() if kt > 1 else inp[..., :0]
    return y


def _resblock_step(block, x, state, key):
    if block.short is None:
        short = x
    else:
        short = _causal_conv_step(block.short[0], x, state, key + '.short.conv')
        short = block.short[1](short)
    h = _causal_conv_step(block.c1, x, state, key + '.c1')
    h = block.act(block.bn1(h))
    h = _causal_conv_step(block.c2, h, state, key + '.c2')
    h = block.bn2(h)
    return block.act(h + short)


def ema_decay_from_window(window_ms):
    """Convert an equivalent causal EMA window to a per-frame decay."""
    if window_ms <= 0:
        raise ValueError('EMA window must be positive')
    hop_ms = HOP * 1000.0 / FS
    return math.exp(-hop_ms / float(window_ms))


def make_window(name, device, dtype=torch.float32):
    if name == 'hamming':
        return torch.hamming_window(N_FFT, periodic=True, device=device, dtype=dtype)
    if name == 'hann':
        return torch.hann_window(N_FFT, periodic=True, device=device, dtype=dtype)
    raise ValueError(f'unsupported window: {name}')


def set_model_window(model, name, device):
    """Set all registered analysis windows, including the profile encoder."""
    win = make_window(name, device)
    changed = 0
    for module in model.modules():
        if 'win' in module._buffers and module._buffers['win'] is not None:
            module._buffers['win'] = win.to(device=module._buffers['win'].device,
                                              dtype=module._buffers['win'].dtype)
            changed += 1
    if changed == 0:
        raise RuntimeError('no registered STFT window found in model')
    return win


class StreamingState:
    """State for one batch of independent mixture streams."""

    def __init__(self, batch, k, device):
        self.conv = {}
        self.q_m = torch.full((batch,), -float('inf'), device=device)
        self.q_d = torch.zeros(batch, device=device)
        self.q_n = None
        self.q_dim = None

        # Recency-weighted prefix statistics. Each statistic has its own window.
        self.bank_ema = None
        self.ref_ema = None
        self.energy_initialized = False

        self.Rs = None
        self.target_initialized = False
        self.confidence_ema = torch.zeros(batch, device=device)

        self.Rn = None
        self.Rn_num = None
        self.Rn_den = None
        self.noise_initialized = False

        # RTF is static within an utterance; selection is updated at fixed
        # absolute frame boundaries and latched for the following block.
        self.frame_count = 0
        self.h_sel_hold = None
        self.alpha_hold = None


def _ema_prefix(x, old, initialized, decay):
    """Vectorized exact EMA prefix for x[..., T]."""
    t = x.shape[-1]
    idx = torch.arange(t, device=x.device, dtype=x.real.dtype)
    lam_pow = decay ** idx
    if initialized:
        base = decay * old
        obs = x
    else:
        base = x[..., 0]
        obs = x.clone()
        obs[..., 0] = 0
    weighted = obs * ((1.0 - decay) / lam_pow).view(*([1] * (x.ndim - 1)), t)
    return base[..., None] * lam_pow + torch.cumsum(weighted, dim=-1) * lam_pow


def _stream_mix_query(model, X, state):
    mq = model.mix_query
    feat = torch.cat([torch.log1p(X.abs()), X.real, X.imag], 1)
    h = _causal_conv_step(mq.stem[0], feat, state, 'mixquery.stem0')
    h = mq.stem[2](mq.stem[1](h))
    h = _causal_conv_step(mq.stem[3], h, state, 'mixquery.stem3')
    h = mq.stem[5](mq.stem[4](h))
    h = h.mean(dim=2).transpose(1, 2)
    return mq.proj(h).transpose(1, 2)


def _stream_mask_net(model, X, cond, state, rtf=None):
    net = model.mask_net
    mag, ipd, spatial = mask_input_features(X, rtf)
    feat = torch.cat([net.input_norm(mag), ipd, spatial], dim=1)
    h = _causal_conv_step(net.stem[0], feat, state, 'mask.stem0')
    h = net.stem[2](net.stem[1](h))
    for i, (block, film) in enumerate(zip(net.stages, net.films)):
        gb = film(cond.float())[:, :, None, None]
        g, b = gb.chunk(2, 1)
        h = _resblock_step(block, h * (1 + g) + b, state, f'mask.block{i}')
    m = torch.softmax(_causal_conv_step(net.head, h, state, 'mask.head'), dim=1)
    if m.shape[-2:] != X.shape[-2:]:
        m = F.interpolate(m, size=X.shape[-2:], mode='nearest')
    return m


def _stream_mask_refiner(model, X, mask_base, h_mvdr, cond, state):
    net = model.mask_refiner
    p = mask_base[:, 0].clamp(1e-4, 1 - 1e-4)
    ones = torch.ones_like(h_mvdr[:, :1])
    hfull = torch.cat([ones, h_mvdr], 1)
    den = hfull.abs().square().sum(1).clamp_min(1e-6)
    ybf = torch.einsum('bcft,bcft->bft', hfull.conj(), X) / den
    cfeat = net.cond(cond.float())[:, :, None, None].expand(-1, -1, X.shape[2], X.shape[3])
    feat = torch.cat([
        p[:, None], X.real, X.imag,
        ybf.real[:, None], ybf.imag[:, None], cfeat,
    ], 1)
    h = _causal_conv_step(net.stem[0], feat, state, 'maskref.stem0')
    h = net.stem[2](net.stem[1](h))
    h = _causal_conv_step(net.stem[3], h, state, 'maskref.stem3')
    h = net.stem[5](net.stem[4](h))
    delta = torch.tanh(_causal_conv_step(net.head, h, state, 'maskref.head'))[:, 0]
    alpha = 0.10 * torch.sigmoid(net.gate)
    p_ref = torch.sigmoid(torch.logit(p) + alpha * delta)
    return torch.stack([p_ref, 1 - p_ref], 1)


def _online_q_pooling(model, query, state, decay):
    """Parallel causal EMA-softmax pooling with sequential-equivalent state."""
    att = model.rtf_attention
    q = F.normalize(F.linear(query.transpose(1, 2).float(), att.q.weight.float()).double(), dim=-1)
    score = F.linear(q, att.pool.weight.double(), att.pool.bias.double()).squeeze(-1)
    b, t, _ = q.shape
    time = torch.arange(t, device=q.device)
    lag = time[:, None] - time[None, :]
    log_w = score[:, None, :] + lag[None].float() * math.log(decay)
    causal = lag[None] >= 0
    log_w = log_w.masked_fill(~causal, -float('inf'))
    if state.q_n is None:
        prev_log = torch.full((b, t), -float('inf'), device=q.device)
    else:
        prev_log = state.q_m[:, None] + (time[None] + 1).float() * math.log(decay)
    all_log = torch.cat([log_w, prev_log[..., None]], dim=-1)
    m = all_log.max(dim=-1, keepdim=True).values
    w = torch.exp(all_log - m)
    curr_w = w[..., :t]
    prev_w = w[..., t]
    num = torch.einsum('bts,bsd->btd', curr_w, q)
    den = curr_w.sum(-1)
    if state.q_n is not None:
        # state.q_n/q_d are unnormalized relative to state.q_m.
        num = num + prev_w[..., None] * state.q_n[:, None]
        den = den + prev_w * state.q_d[:, None]
    q_global = F.normalize(num / den[..., None].clamp_min(1e-12), dim=-1).to(query.dtype)
    state.q_m = m[:, -1, 0].detach()
    state.q_d = den[:, -1].detach()
    state.q_n = num[:, -1].detach()
    return q_global


def _spatial_prefix(model, X, h_bank, state, decay):
    """Vectorized candidate-energy EMA features."""
    b, k = h_bank.shape[:2]
    hfull = torch.cat([
        torch.ones(b, k, 1, X.shape[2], device=X.device, dtype=h_bank.dtype),
        h_bank,
    ], 2)
    den = hfull.abs().square().sum(2).clamp_min(1e-6)
    ybank = torch.einsum('bkcf,bcft->bkft', hfull.conj(), X) / den[:, :, :, None]
    bank_energy = ybank.abs().square().mean(-2)
    ref_energy = X[:, 0].abs().square().mean(-2)
    old_bank = state.bank_ema if state.energy_initialized else torch.zeros_like(bank_energy[..., 0])
    old_ref = state.ref_ema if state.energy_initialized else torch.zeros_like(ref_energy[..., 0])
    bank = _ema_prefix(bank_energy, old_bank, state.energy_initialized, decay)
    ref = _ema_prefix(ref_energy, old_ref, state.energy_initialized, decay)
    spatial = torch.stack([
        torch.log1p(bank),
        torch.log((bank + 1e-8) / (ref[:, None] + 1e-8)),
    ], -1)
    state.bank_ema = bank[..., -1].detach()
    state.ref_ema = ref[..., -1].detach()
    state.energy_initialized = True
    return spatial


def _stream_rtf_attention(model, q_global, h_bank, geometry, spatial, state):
    """Vectorized per-frame RTF candidate attention."""
    att = model.rtf_attention
    b, t = q_global.shape[:2]
    k = h_bank.shape[1]
    h_rep = h_bank[:, None].expand(b, t, k, 7, h_bank.shape[-1]).reshape(b * t, k, 7, h_bank.shape[-1])
    geo_rep = geometry[:, None].expand(b, t, k, geometry.shape[-1]).reshape(b * t, k, geometry.shape[-1])
    sp_rep = spatial.permute(0, 3, 1, 2).reshape(b * t, k, 2)
    tokens = model.rtf_tokenizer(h_rep, geo_rep, sp_rep).reshape(b, t, k, -1)
    keys = F.normalize(att.k(tokens).float(), dim=-1)
    score = torch.einsum('btd,btkd->btk', q_global, keys) * att.logit_scale.exp().clamp(max=100.0)
    alpha = torch.softmax(score, dim=-1)
    hsel = torch.einsum('btk,btkcf->btcf', alpha.to(h_bank.dtype), h_bank[:, None].expand(-1, t, -1, -1, -1))
    return hsel.permute(0, 2, 3, 1), alpha


def _rtf_selection_at(model, q_global_t, h_bank, geometry, spatial_t, state):
    """Compute one RTF candidate choice at an absolute frame boundary."""
    att = model.rtf_attention
    tokens = model.rtf_tokenizer(h_bank, geometry, spatial_t)
    keys = F.normalize(att.k(tokens).float(), dim=-1)
    score = torch.einsum('bd,bkd->bk', q_global_t, keys) * att.logit_scale.exp().clamp(max=100.0)
    alpha = torch.softmax(score, dim=1)
    hsel = torch.einsum('bk,bkcf->bcf', alpha.to(h_bank.dtype), h_bank)
    return hsel, alpha


def _latched_rtf_selection(model, q_global, h_bank, geometry, spatial, state,
                           update_frames):
    """Vectorized block-wise RTF selection at absolute frame boundaries."""
    t = q_global.shape[1]
    pos = 0
    current_h = state.h_sel_hold
    current_alpha = state.alpha_hold
    hparts, aparts = [], []
    while pos < t:
        global_pos = state.frame_count + pos
        until_boundary = update_frames - (global_pos % update_frames)
        seg = min(t - pos, until_boundary)
        hparts.append(current_h[:, None].expand(-1, seg, -1, -1))
        aparts.append(current_alpha[:, None].expand(-1, seg, -1))
        pos += seg
        if (state.frame_count + pos) % update_frames == 0:
            current_h, current_alpha = _rtf_selection_at(
                model, q_global[:, pos - 1], h_bank, geometry,
                spatial[..., pos - 1, :], state)
            state.h_sel_hold = current_h.detach() if not torch.is_grad_enabled() else current_h
            state.alpha_hold = current_alpha.detach() if not torch.is_grad_enabled() else current_alpha
    state.frame_count += t
    hseq = torch.cat(hparts, dim=1).permute(0, 2, 3, 1)
    aseq = torch.cat(aparts, dim=1)
    return hseq, aseq


def _stream_rtf_predictor(model, q_global, h_sel):
    """Apply the frame-wise predictor using the causal pooled query."""
    pred = model.rtf_predictor
    b, _, f, t = h_sel.shape
    q = q_global  # B,T,D
    cond = pred.cond(q).reshape(b * t, -1)[:, :, None].expand(-1, -1, f)
    feat = torch.cat([h_sel.real, h_sel.imag], 1).permute(0, 3, 1, 2).reshape(b * t, 14, f)
    residual = pred.net(torch.cat([feat, cond], 1)).reshape(b, t, 14, f).permute(0, 2, 3, 1)
    delta = residual[:, :7] + 1j * residual[:, 7:]
    return h_sel + delta


def _stream_cov_rtf(model, X, mask, state, decay):
    """Causal mask-weighted target covariance -> RTF observation."""
    p = mask[:, 0].float().clamp(0, 1)
    xw = X.to(torch.complex64) * torch.sqrt(p)[:, None]
    obs = torch.einsum('bcft,bdft->bfcdt', xw, xw.conj())
    old = state.Rs if state.target_initialized else torch.zeros_like(obs[..., 0])
    prefix = _ema_prefix(obs, old, state.target_initialized, decay)
    den = prefix[:, :, 0, 0].abs().clamp_min(1e-8)
    h_cov = (prefix[:, :, 1:, 0] / den[:, :, None]).permute(0, 2, 1, 3)
    state.Rs = prefix[..., -1].detach()
    state.target_initialized = True
    return h_cov


def _stream_rtf_refiner(model, X, mask, h_sel, cond, state, decay):
    """Vectorized EMA target covariance with the existing gate head."""
    p = mask[:, 0].float().clamp(0, 1)
    xw = X.to(torch.complex64) * torch.sqrt(p)[:, None]
    obs = torch.einsum('bcft,bdft->bfcdt', xw, xw.conj())
    old = state.Rs if state.target_initialized else torch.zeros_like(obs[..., 0])
    prefix = _ema_prefix(obs, old, state.target_initialized, decay)
    denom = prefix[:, :, 0, 0].abs().clamp_min(1e-6)
    h_hat = (prefix[:, :, 1:, 0] / denom[:, :, None]).permute(0, 2, 1, 3)
    delta = (h_hat - h_sel).abs().mean((-2, -3))
    scale = h_sel.abs().mean((-2, -3)).clamp_min(1e-6)
    disagreement = delta / scale
    confidence = _ema_prefix(
        p.mean(1), state.confidence_ema, state.target_initialized, decay).clamp(0, 1)
    gate_in = torch.cat([
        cond.float()[:, None].expand(-1, X.shape[-1], -1),
        confidence[..., None], disagreement[..., None],
    ], -1)
    beta = 0.5 * torch.sigmoid(model.rtf_refiner.gate(gate_in)).squeeze(-1)
    href = h_sel + beta[:, None, None] * (h_hat - h_sel)
    state.Rs = prefix[..., -1].detach()
    state.confidence_ema = confidence[:, -1].detach()
    state.target_initialized = True
    return href, beta, h_hat


def _stream_mvdr(model, X, mask, h_ref, state, decay):
    """Causal MVDR with target-reconstruction noise covariance.

    The soft target mask and current RTF reconstruct the target spatial image;
    Rn is accumulated from the residual X - target_est.  This replaces the
    unstable X*(1-p) noise approximation.
    """
    p = mask[:, 0].float().clamp(0, 1)
    x = X.to(torch.complex64)
    b, c, f, t = x.shape
    h_ref = h_ref.to(torch.complex64)
    if h_ref.ndim == 3:
        h_ref = h_ref[..., None]
    h = torch.cat([
        torch.ones(b, 1, f, t, device=X.device, dtype=torch.complex64),
        h_ref.expand(b, 7, f, t),
    ], 1)
    target_est = h * (p * x[:, 0])[:, None]
    noise = x - target_est
    obs = torch.einsum('bcft,bdft->bfcdt', noise, noise.conj()).to(torch.complex64)
    old = state.Rn if state.noise_initialized else torch.zeros_like(obs[..., 0])
    prefix = _ema_prefix(obs, old, state.noise_initialized, decay)
    state.Rn = prefix[..., -1].detach()
    eye = torch.eye(c, device=X.device, dtype=torch.complex64)
    rbar = prefix + model.mvdr.diag * eye[None, None, :, :, None]
    h_seq = h.permute(0, 3, 1, 2)
    af = rbar.permute(0, 4, 1, 2, 3).reshape(b * t, f, c, c)
    rhs = h_seq.permute(0, 1, 3, 2).reshape(b * t, f, c, 1)
    w = torch.linalg.solve(af, rhs)
    den = torch.einsum('bfc,bfc->bf', w.conj()[..., 0], rhs[..., 0]).abs().clamp_min(1e-6)
    w = w[..., 0] / den[..., None]
    xf = x.permute(0, 3, 1, 2).reshape(b * t, c, f)
    y = torch.einsum('bfc,bcf->bf', w.conj(), xf).reshape(b, t, f).permute(0, 2, 1)
    state.noise_initialized = True
    return y


class StreamingMVDRSession:
    """Incremental separation session: each step sees only current X frames."""

    def __init__(self, model, registration, target_seat, geometry,
                 attention_window_ms=500, target_window_ms=500,
                 noise_window_ms=750, rtf_update_frames=10):
        self.model = model
        self.registration = registration
        self.target_seat = target_seat
        self.geometry = geometry
        self.attention_window_ms = attention_window_ms
        self.target_window_ms = target_window_ms
        self.noise_window_ms = noise_window_ms
        b = target_seat.shape[0]
        reg_b = registration['z_prof'].shape[0]
        if reg_b == 1:
            z_prof = registration['z_prof'][0, target_seat]
            z_dir = registration['z_dir'][0, target_seat]
            r = registration['r'][0, target_seat]
            self.h_center = registration['rtf_all'][0, target_seat, 0]
            idx = torch.arange(b, device=target_seat.device)
        elif reg_b == b:
            idx = torch.arange(b, device=target_seat.device)
            z_prof = registration['z_prof'][idx, target_seat]
            z_dir = registration['z_dir'][idx, target_seat]
            r = registration['r'][idx, target_seat]
            self.h_center = registration['rtf_all'][idx, target_seat, 0]
        else:
            raise ValueError(f'registration batch {reg_b} does not match scene batch {b}')
        self.mask_cond = self.model.mask_cond_proj(torch.cat([z_prof, z_dir, r], dim=-1))
        self.rtf_cond = self.model.rtf_cond_proj(torch.cat([z_dir, r], dim=-1))
        if geometry.ndim == 3:
            self.geo_center = geometry[target_seat, 0, :]
        else:
            self.geo_center = geometry[idx, target_seat, 0]
        self.state = StreamingState(b, 1, target_seat.device)
        self.rtf_update_frames = max(1, int(rtf_update_frames))
        self.state.h_sel_hold = self.h_center
        self.state.alpha_hold = torch.ones(
            (b, 1), device=self.h_center.device, dtype=self.h_center.real.dtype)
        self.att_decay = ema_decay_from_window(attention_window_ms)
        self.target_decay = ema_decay_from_window(target_window_ms)
        self.noise_decay = ema_decay_from_window(noise_window_ms)

    def step(self, X, rtf_mask=None):
        """Consume current complex STFT frames [B,C,F,T] and emit Y frames."""
        # RTF-first path: use the unmasked mixture covariance, never the mask.
        uniform_mask = torch.ones(
            X.shape[0], 2, X.shape[2], X.shape[3],
            device=X.device, dtype=X.real.dtype)
        h_cov = _stream_cov_rtf(self.model, X, uniform_mask, self.state, self.target_decay)
        h_res = self.model.rtf_net(h_cov, self.h_center, self.rtf_cond)
        scale = torch.sigmoid(self.model.rtf_net.residual_scale)
        h_ref = h_cov * complex_rtf_gate(h_res, scale)
        # RTF now guides mask estimation through spatial beamformer evidence.
        mask_base = _stream_mask_net(
            self.model, X, self.mask_cond, self.state, rtf=h_ref)
        mask_ref = self.model.mask_refiner(mask_base)
        Y = _stream_mvdr(
            self.model, X, mask_ref, h_ref, self.state, self.noise_decay)
        alpha = torch.ones(X.shape[0], 1, X.shape[-1], device=X.device,
                           dtype=h_ref.real.dtype)
        beta = torch.ones(X.shape[0], X.shape[-1], device=X.device,
                          dtype=h_ref.real.dtype)
        return {
            'Y': Y, 'mask_base': mask_base, 'mask_ref': mask_ref,
            'h_sel': self.h_center[:, :, :, None].expand(-1, -1, -1, X.shape[-1]),
            'h_pred': h_ref, 'h_ref': h_ref,
            'h_hat': h_cov, 'alpha': alpha, 'beta': beta,
        }


def stream_separation_forward(model, registration, mix, target_seat, geometry,
                              chunk_frames=10, attention_window_ms=500,
                              target_window_ms=500, noise_window_ms=750,
                              window=None, rtf_update_frames=10, rtf_mask=None):
    """Training helper: run the exact same incremental session over a full clip."""
    if window is None:
        window = model.win
    Xall = causal_stft(mix, window)
    session = StreamingMVDRSession(
        model, registration, target_seat, geometry,
        attention_window_ms=attention_window_ms,
        target_window_ms=target_window_ms,
        noise_window_ms=noise_window_ms,
        rtf_update_frames=rtf_update_frames,
    )
    ys, masks_base, masks, hpreds, hrefs, hcovs, alphas = [], [], [], [], [], [], []
    for start, end in chunks(Xall.shape[-1], chunk_frames):
        mask_chunk = rtf_mask[..., start:end] if rtf_mask is not None else None
        out = session.step(Xall[..., start:end], rtf_mask=mask_chunk)
        ys.append(out['Y'])
        masks_base.append(out['mask_base'])
        masks.append(out['mask_ref'])
        hpreds.append(out['h_pred'])
        hrefs.append(out['h_ref'])
        hcovs.append(out['h_hat'])
        alphas.append(out['alpha'])
    Yall = torch.cat(ys, -1)
    mask_all = torch.cat(masks, -1)
    y = causal_istft(Yall, mix.shape[-1], window)
    Ymask = mask_all[:, 0] * Xall[:, 0]
    y_mask = causal_istft(Ymask, mix.shape[-1], window)
    return {
        'y': y, 'y_mvdr': y, 'y_mask': y_mask, 'Y': Yall,
        'mask_base': torch.cat(masks_base, -1), 'mask_ref': mask_all,
        'h_pred': torch.cat(hpreds, -1), 'h_ref': torch.cat(hrefs, -1),
        'h_hat': torch.cat(hcovs, -1), 'alpha': torch.cat(alphas, 1),
    }


class StreamingAudioMVDR:
    """Complete block-fed audio inference frontend with causal Hamming STFT/OLA."""

    def __init__(self, model, registration, target_seat, geometry, window,
                 attention_window_ms=500, target_window_ms=500,
                 noise_window_ms=750, rtf_update_frames=10):
        self.window = window
        self.stft = StreamingCausalSTFT(window)
        self.istft = StreamingOLAISTFT(window)
        self.session = StreamingMVDRSession(
            model, registration, target_seat, geometry,
            attention_window_ms=attention_window_ms,
            target_window_ms=target_window_ms,
            noise_window_ms=noise_window_ms,
            rtf_update_frames=rtf_update_frames,
        )
        self.total_input = 0

    def reset(self):
        """Reset all utterance-level streaming state and overlap buffers."""
        self.stft = StreamingCausalSTFT(self.window)
        self.istft = StreamingOLAISTFT(self.window)
        self.session = StreamingMVDRSession(
            self.session.model,
            self.session.registration,
            self.session.target_seat,
            self.session.geometry,
            attention_window_ms=self.session.attention_window_ms,
            target_window_ms=self.session.target_window_ms,
            noise_window_ms=self.session.noise_window_ms,
            rtf_update_frames=self.session.rtf_update_frames,
        )
        self.total_input = 0

    def _run(self, X):
        if X.numel() == 0:
            return self.window.new_empty(0)
        out = self.session.step(X)
        return self.istft.push(out['Y'])

    def process(self, audio):
        if audio.ndim == 2:
            audio = audio[:, None]
        self.total_input += audio.shape[-1]
        return self._run(self.stft.push(audio))

    def finish(self):
        tail = self._run(self.stft.finish())
        final = self.istft.finish(self.total_input)
        if tail.numel() and final.numel():
            return torch.cat([tail, final], -1)
        return tail if tail.numel() else final

