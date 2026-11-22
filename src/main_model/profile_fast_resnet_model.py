#!/usr/bin/env python3
"""Joint multi-mic seat-profile + fast offline separation model.

Profile branch:
    four 3-s, 8-mic enrollment clips -> shared ResNet -> 4 x PROFILE_DIM-D seat tokens
    -> relation Transformer -> seat logits + pair relation logits.

Separation branch:
    3-s, 8-mic mixture complex STFT -> lightweight non-causal 2-D ResNet
    conditioned on target-seat token and relation token -> target waveform.

All branches are differentiable and intended for joint training.
"""
from __future__ import annotations
import math, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F

N_FFT = 512
HOP = 160
SR = 16000
PROFILE_DIM = 512
PROFILE_RAW_DIM = 1536


def stft_mag(x: torch.Tensor, win: torch.Tensor):
    """[B,C,T] -> complex [B,C,F,Frames]."""
    b, c, t = x.shape
    X = torch.stft(x.reshape(b * c, t), n_fft=N_FFT, hop_length=HOP,
                   window=win, return_complex=True, center=True)
    return X.reshape(b, c, X.shape[-2], X.shape[-1])


def profile_features(x: torch.Tensor, win: torch.Tensor):
    """8-channel log-mag + IPD + ILD, plus an utterance-level RTF proxy."""
    X = stft_mag(x, win)
    mag = torch.log1p(X.abs())
    ipd = X[:, :-1] * X[:, 1:].conj()
    ipd = ipd / (ipd.abs() + 1e-6)
    ild = mag[:, 1:] - mag[:, :1]
    # Relative transfer function proxy against reference microphone.
    ref = X[:, :1]
    cross = (X[:, 1:] * ref.conj()).mean(-1)
    power = (ref.abs().square()).mean(-1).clamp_min(1e-6)
    rtf = cross / power
    rtf = torch.cat([rtf.real, rtf.imag], 1).flatten(1)
    return torch.cat([mag, ipd.real, ipd.imag, ild], dim=1), rtf



def estimate_rtf(x: torch.Tensor, win: torch.Tensor):
    """Estimate [B,7,F] relative transfer function against mic 0."""
    X = stft_mag(x, win)
    ref = X[:, :1]
    cross = (X[:, 1:] * ref.conj()).mean(-1)
    power = ref.abs().square().mean(-1).clamp_min(1e-6)
    return cross / power

class BNReLU(nn.Sequential):
    def __init__(self, c):
        super().__init__(nn.BatchNorm2d(c), nn.ReLU(inplace=True))


class ResBlock(nn.Module):
    def __init__(self, cin, cout, stride=(1, 1)):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)
        self.short = None
        if cin != cout or stride != (1, 1):
            self.short = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride=stride, bias=False),
                nn.BatchNorm2d(cout),
            )

    def forward(self, x):
        s = x if self.short is None else self.short(x)
        x = self.act(self.bn1(self.c1(x)))
        x = self.bn2(self.c2(x))
        return self.act(x + s)


class ProfileEncoder(nn.Module):
    """Shared non-streaming seat encoder for one 8-mic enrollment clip."""
    def __init__(self, dim=PROFILE_DIM, raw_dim=None, blocks=(2, 2, 2, 2), widths=(96, 192, 384, 768)):
        raw_dim = dim if raw_dim is None else raw_dim
        super().__init__()
        self.register_buffer('win', torch.hann_window(N_FFT), persistent=False)
        self.rtf_proj = nn.Sequential(
            nn.LayerNorm(14 * (N_FFT // 2 + 1)),
            nn.Linear(14 * (N_FFT // 2 + 1), raw_dim),
            nn.GELU(), nn.LayerNorm(raw_dim),
        )
        self.stem = nn.Sequential(
            nn.Conv2d(29, widths[0], 7, stride=(2, 2), padding=3, bias=False),
            nn.BatchNorm2d(widths[0]), nn.ReLU(inplace=True),
        )
        stages = []
        cin = widths[0]
        strides = [(1, 1), (2, 2), (2, 1), (1, 1)]
        for i, (cout, n, st) in enumerate(zip(widths, blocks, strides)):
            for j in range(n):
                stages.append(ResBlock(cin if j == 0 else cout, cout, st if j == 0 else (1, 1)))
            cin = cout
        self.stages = nn.Sequential(*stages)
        self.out = nn.Sequential(
            nn.LayerNorm(widths[-1] * 2),
            nn.Linear(widths[-1] * 2, raw_dim),
            nn.GELU(),
            nn.LayerNorm(raw_dim),
        )

    def forward(self, x):
        f, rtf = profile_features(x, self.win)
        h = self.stages(self.stem(f))
        h = torch.cat([h.mean(dim=(2, 3)), h.std(dim=(2, 3)).clamp_min(1e-5)], dim=1)
        return self.out(h) + self.rtf_proj(rtf)


class SeatClassifier(nn.Module):
    """Residual token classifier for the four physical seat identities."""
    def __init__(self, dim=PROFILE_DIM, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, 2 * dim)
        self.fc2 = nn.Linear(2 * dim, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(dim, 4)

    def forward(self, z):
        h = self.norm(z)
        h = self.fc2(self.drop(self.act(self.fc1(h))))
        return self.out(self.norm(z + h))


class PairRelationHead(nn.Module):
    """Symmetric pair classifier using sum, distance, and product features."""
    def __init__(self, dim=PROFILE_DIM, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(3 * dim)
        self.net = nn.Sequential(
            nn.Linear(3 * dim, 2 * dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2 * dim, dim), nn.GELU(), nn.LayerNorm(dim),
            nn.Linear(dim, 3),
        )

    def forward(self, left, right):
        pair = torch.cat([
            left + right,
            (left - right).abs(),
            left * right,
        ], -1)
        return self.net(self.norm(pair))


class ProfileRelation(nn.Module):
    def __init__(self, dim=PROFILE_DIM, heads=8, layers=2):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4,
            dropout=0.1, activation='gelu', batch_first=True,
            norm_first=True,
        )
        self.tr = nn.TransformerEncoder(layer, num_layers=layers)
        self.seat_head = SeatClassifier(dim)
        self.pair_head = PairRelationHead(dim)

    def forward(self, z):
        # z: [B,4,D]
        r = self.tr(z)
        seat_logits = self.seat_head(z)                 # [B,4,4], no position shortcut
        pair_logits = []
        for i in range(4):
            for j in range(4):
                if i != j:
                    pair_logits.append(self.pair_head(r[:, i], r[:, j]))
        pair_logits = torch.stack(pair_logits, 1)       # [B,12,3]
        return r, seat_logits, pair_logits


class FiLM(nn.Module):
    def __init__(self, dim, ch):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2 * ch))
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h, cond):
        gb = self.net(cond)[:, :, None, None]
        g, b = gb.chunk(2, 1)
        return h * (1 + g) + b


class SepBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.block = ResBlock(cin, cout, stride)

    def forward(self, x):
        return self.block(x)




class RTFAggregator(nn.Module):
    """Attention-pool multiple enrollment RTF estimates without scalar complex averaging."""
    def __init__(self):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(14), nn.Linear(14, 32), nn.GELU(), nn.Linear(32, 1)
        )

    def forward(self, rtf):
        # rtf: [B,K,7,F]
        b, k, _, f = rtf.shape
        feat = torch.cat([rtf.real, rtf.imag], 2).permute(0, 1, 3, 2)  # B,K,F,14
        score = self.score(feat).squeeze(-1)                        # B,K,F
        alpha = torch.softmax(score, dim=1)
        return (alpha[:, :, None, :] * rtf).sum(1)                 # B,7,F


class RTFSelector(nn.Module):
    """Select/weight enrollment RTF candidates using the target separation token."""
    def __init__(self, dim=PROFILE_DIM):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Linear(14, 32), nn.GELU(), nn.Linear(32, 1)
        )
        self.cond = nn.Linear(dim, 1)
        nn.init.zeros_(self.feat[-1].weight)
        nn.init.zeros_(self.feat[-1].bias)
        nn.init.zeros_(self.cond.weight)
        nn.init.zeros_(self.cond.bias)

    def forward(self, rtf, z):
        # rtf [B,K,7,F], z [B,D]
        pooled = torch.cat([rtf.real.mean(-1), rtf.imag.mean(-1)], -1)
        score = self.feat(pooled).squeeze(-1) + self.cond(z)
        alpha = torch.softmax(score, dim=1)
        return (alpha[:, :, None, None] * rtf).sum(1), alpha


class MixtureAwareRTFSelector(nn.Module):
    """Select RTF candidates using mixture, predicted mask, and z_sep."""
    def __init__(self, dim=PROFILE_DIM):
        super().__init__()
        self.feat = nn.Sequential(nn.Linear(4, 32), nn.GELU(), nn.Linear(32, 1))
        self.cond = nn.Linear(dim, 1)
        nn.init.zeros_(self.feat[-1].weight)
        nn.init.zeros_(self.feat[-1].bias)
        nn.init.zeros_(self.cond.weight)
        nn.init.zeros_(self.cond.bias)

    def forward(self, xbfs, target_rough, X0, rtf_candidates, cond):
        # xbfs[B,K,F,T], target_rough[B,F,T], X0[B,F,T], rtf[B,K,7,F]
        et = target_rough.abs().square().mean((-1, -2)).clamp_min(1e-8)
        ek = xbfs.abs().square().mean((-1, -2)).clamp_min(1e-8)
        corr = (xbfs * target_rough[:, None].conj()).sum((-1, -2)).real / (torch.sqrt(et[:, None] * ek) + 1e-6)
        e0 = X0.abs().square().mean((-1, -2)).clamp_min(1e-8)
        energy_ratio = torch.log((ek + 1e-8) / (e0[:, None] + 1e-8))
        cross = (xbfs * X0[:, None].conj()).sum((-1, -2)).abs().square()
        coherence = cross / (ek * e0[:, None] * xbfs.shape[-1] * xbfs.shape[-2] + 1e-8)
        rtf_mag = rtf_candidates.abs().square().mean((-1, -2))
        feat = torch.stack([corr, energy_ratio, coherence, torch.log1p(rtf_mag)], -1)
        score = self.feat(feat).squeeze(-1) + self.cond(cond)
        alpha = torch.softmax(score, dim=1)
        selected = (alpha[:, :, None, None] * rtf_candidates).sum(1)
        return selected, alpha

class NeuralRTFBeamformer(nn.Module):
    """Lightweight target RTF beamformer; z_sep controls a complex correction."""
    def __init__(self, dim=PROFILE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(14, 32, 3, padding=1), nn.PReLU(),
            nn.Conv1d(32, 16, 1),
        )
        self.cond = nn.Linear(dim, 16)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, X, rtf, cond):
        # X [B,8,F,T], rtf [B,7,F], cond [B,D]
        b, c, f, t = X.shape
        base = torch.cat([torch.ones(b, 1, f, device=X.device, dtype=X.dtype), rtf], 1)
        rf = self.net(torch.cat([rtf.real, rtf.imag], 1))
        gate = 1 + torch.tanh(self.cond(cond))[:, :, None]  # B,16,1
        corr = rf[:, :8] * gate[:, :8, :] + 1j * rf[:, 8:] * gate[:, 8:, :]
        w = base + corr
        den = w.abs().square().sum(1).clamp_min(1e-6)  # [B,F]
        xbf = (w.conj()[..., None] * X).sum(1) / den[:, None, :, None]
        return xbf[:, 0]


class MixtureRTFRefiner(nn.Module):
    """Refine enrollment RTF using a predicted target mask on the mixture."""
    def __init__(self, dim=PROFILE_DIM):
        super().__init__()
        self.smooth = nn.Conv1d(14, 14, 3, padding=1, groups=14)
        self.gate = nn.Linear(dim, 1)
        # Identity initialisation: start from enrollment RTF, learn correction slowly.
        nn.init.zeros_(self.smooth.weight)
        nn.init.zeros_(self.smooth.bias)
        with torch.no_grad():
            for i in range(14):
                self.smooth.weight[i, 0, 1] = 1.0
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.alpha = nn.Parameter(torch.tensor(-4.0))

    def forward(self, X, mask, rtf0, cond):
        # X [B,8,F,T], mask [B,2,F,T], rtf0 [B,7,F]
        b, c, f, t = X.shape
        m = torch.complex(mask[:, 0].float(), mask[:, 1].float())
        weight = m.abs().square().clamp(0, 1)
        ref = X[:, :1]
        cross = (X[:, 1:] * ref.conj() * weight[:, None]).sum(-1)
        power = (ref.abs().square() * weight[:, None]).sum(-1)  # B,1,F
        h1 = cross / (power + 1e-6)
        # Smooth complex RTF across frequency.
        hv = torch.cat([h1.real, h1.imag], 1)
        hv = self.smooth(hv)
        h1 = hv[:, :7] + 1j * hv[:, 7:]
        alpha = torch.sigmoid(self.gate(cond.float()))[:, :, None] * torch.sigmoid(self.alpha)
        # Low-confidence masks must fall back toward the enrollment RTF.
        confidence = weight.mean(-1).clamp(0, 1)[:, None, :]
        valid = (power > 1e-6).to(h1.real.dtype)
        update = valid * confidence * alpha
        h = rtf0 + update * (h1 - rtf0)
        return h, h1

class FastFiLMResNetSeparator(nn.Module):
    """Non-causal, convolution-only separator. No recurrent or attention ops."""
    def __init__(self, dim=PROFILE_DIM, widths=(64, 128, 256, 256)):
        super().__init__()
        self.register_buffer('win', torch.hann_window(N_FFT), persistent=False)
        self.cond = nn.Sequential(
            nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim),
            nn.GELU(), nn.LayerNorm(dim),
        )
        self.stem = nn.Sequential(
            nn.Conv2d(18, widths[0], 7, padding=3, bias=False),
            nn.BatchNorm2d(widths[0]), nn.ReLU(inplace=True),
        )
        self.stages = nn.ModuleList([
            SepBlock(widths[0], widths[0], (1, 1)),
            SepBlock(widths[0], widths[1], (2, 1)),
            SepBlock(widths[1], widths[2], (2, 1)),
            SepBlock(widths[2], widths[3], (1, 1)),
        ])
        self.films = nn.ModuleList([FiLM(dim, c) for c in widths])
        self.dropouts = nn.ModuleList([nn.Dropout2d(0.1) for _ in widths])
        self.head = nn.Conv2d(widths[-1], 2, 1)
        self.beamformer = NeuralRTFBeamformer(dim)
        self.mvdr = MaskBasedMVDR()
        self.rtf_refiner = MixtureRTFRefiner(dim)
        self.mix_rtf_selector = MixtureAwareRTFSelector(dim)
        self.token_attn = nn.MultiheadAttention(dim, 4, batch_first=True, dropout=0.0)
        self.token_norm = nn.LayerNorm(dim)

    def forward(self, mix, z_target, r_target, rtf_target=None, token_seq=None, rtf_candidates=None):
        # mix: [B,8,T]
        b, c, t = mix.shape
        X = stft_mag(mix, self.win)
        cond = self.cond(torch.cat([z_target, r_target], -1))
        if token_seq is not None:
            q = cond[:, None, :]
            ctx, _ = self.token_attn(q, token_seq, token_seq, need_weights=False)
            cond = self.token_norm(cond + ctx[:, 0])
        feat = torch.cat([X.real, X.imag], 1)
        if rtf_target is not None:
            xbf = self.beamformer(X, rtf_target, cond)
            feat = torch.cat([feat, xbf.real[:, None], xbf.imag[:, None]], 1)
        h = self.stem(feat)
        for block, film, drop in zip(self.stages, self.films, self.dropouts):
            h = drop(film(block(h), cond))
        m = torch.tanh(self.head(h))
        m = F.interpolate(m, size=X.shape[-2:], mode='bilinear', align_corners=False)
        Ym = (m[:, 0] + 1j * m[:, 1]) * X[:, 0]
        y_mask = torch.istft(Ym, n_fft=N_FFT, hop_length=HOP,
                             window=self.win, length=t, center=True)
        rtf_refined = rtf_target
        rtf_sel = rtf_target
        if rtf_target is not None:
            if rtf_candidates is not None:
                # One beamformer output per enrollment RTF candidate.
                xbfs=[]
                for kk in range(rtf_candidates.shape[1]):
                    xbfs.append(self.beamformer(X,rtf_candidates[:,kk],cond))
                xbfs=torch.stack(xbfs,1)
                target_rough=(m[:,0]+1j*m[:,1])*X[:,0]
                rtf_sel, _ = self.mix_rtf_selector(xbfs,target_rough,X[:,0],rtf_candidates,cond)
            rtf_refined, _ = self.rtf_refiner(X, m, rtf_sel, cond)
            Ym_mvdr = self.mvdr(X, m, rtf_refined)
            y_mvdr = torch.istft(Ym_mvdr, n_fft=N_FFT, hop_length=HOP,
                                 window=self.win, length=t, center=True)
        else:
            Ym_mvdr, y_mvdr = Ym, y_mask
        return y_mask, Ym, m, y_mvdr, rtf_refined, rtf_sel


def supcon_loss(z, labels, group_ids=None, temperature=0.07, hard_negative_weight=2.0):
    """Supervised contrastive loss for profile embeddings.

    labels: integer class ids, normally car_id * 4 + seat_id.
    group_ids: car ids; same-car/different-seat pairs are hard negatives.
    """
    z = F.normalize(z.float(), dim=-1)
    b = z.shape[0]
    sim = torch.matmul(z, z.t()) / temperature
    self_mask = torch.eye(b, dtype=torch.bool, device=z.device)
    same = labels[:, None].eq(labels[None, :]) & ~self_mask
    weights = torch.ones_like(sim)
    if group_ids is not None:
        hard = group_ids[:, None].eq(group_ids[None, :]) & ~same & ~self_mask
        weights = torch.where(hard, weights * hard_negative_weight, weights)
    sim = sim.masked_fill(self_mask, -1e4)
    sim_max = sim.max(dim=1, keepdim=True).values.detach()
    exp_sim = torch.exp(sim - sim_max)
    denom = (exp_sim * weights).sum(dim=1)
    pos_exp = (exp_sim * same.float()).sum(dim=1)
    valid = same.any(dim=1)
    if valid.sum() == 0:
        return z.sum() * 0.0
    loss = -(torch.log(pos_exp[valid] + 1e-8) - torch.log(denom[valid] + 1e-8))
    return loss.mean()


def seat_classification_loss(seat_logits, seat_labels):
    """seat_logits [B,4,4], seat_labels [B,4]."""
    return F.cross_entropy(seat_logits.reshape(-1, 4), seat_labels.reshape(-1))


def make_pair_targets(seat_order, device='cpu'):
    """Return [B,12,3] labels using physical seat ids at token positions.

    seat_order[b, i] is the physical seat id represented by token i.
    """
    ids = seat_order.to(device).long()
    b = ids.shape[0]
    row = ((ids[:, :, None] < 2) & (ids[:, None, :] < 2)) | ((ids[:, :, None] >= 2) & (ids[:, None, :] >= 2))
    side = (ids[:, :, None] % 2) == (ids[:, None, :] % 2)
    front = (ids[:, :, None] < 2) & (ids[:, None, :] >= 2)
    pairs = []
    for i in range(4):
        for j in range(4):
            if i != j:
                pairs.append(torch.stack([row[:, i, j], side[:, i, j], front[:, i, j]], -1))
    return torch.stack(pairs, 1).float()  # [B,12,3]


def pair_relation_loss(pair_logits, pair_targets):
    """pair_logits [B,12,3], pair_targets [12,3] or [B,12,3]."""
    if pair_targets.ndim == 2:
        pair_targets = pair_targets[None].expand(pair_logits.shape[0], -1, -1)
    return F.binary_cross_entropy_with_logits(pair_logits.float(), pair_targets.float())


def wrong_seat_hinge(sdr_correct, sdr_wrong, margin=0.2):
    return F.relu(margin - sdr_correct + sdr_wrong).mean()



class MaskBasedMVDR(nn.Module):
    """Diagonal-loaded mask-based MVDR using enrollment RTF."""
    def forward(self, X, mask, rtf, diag=1e-3):
        # X [B,8,F,T], mask [B,2,F,T], rtf [B,7,F]
        b, c, f, t = X.shape
        X = X.to(torch.complex64)
        target_power = mask.float()[:, 0].clamp(0, 1)
        h = torch.cat([
            torch.ones(b, 1, f, device=X.device, dtype=X.dtype), rtf.to(torch.complex64)
        ], 1)
        target_est = h[..., None] * (target_power * X[:, 0])[:, None]
        noise = X - target_est
        R = torch.einsum('bcft,bdft->bfcd', noise, noise.conj()) / max(t, 1)
        eye = torch.eye(c, device=X.device, dtype=torch.complex64)[None, None]
        R = R + diag * eye
        rhs = h.permute(0, 2, 1)
        w = torch.linalg.solve(R, rhs)
        den = torch.einsum('bfc,bfc->bf', w.conj(), rhs).abs().clamp_min(1e-6)
        w = w / den[:, :, None]
        return torch.einsum('bfc,bcft->bft', w.conj(), X)


class JointSeatProfileSeparator(nn.Module):
    def __init__(self, dim=PROFILE_DIM):
        super().__init__()
        self.profile = ProfileEncoder(dim=dim)
        self.id_proj = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim), nn.LayerNorm(dim)
        )
        self.seat_proj = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim), nn.LayerNorm(dim)
        )
        self.sep_proj = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim), nn.LayerNorm(dim)
        )
        self.sep_drop = nn.Dropout(0.1)
        self.rtf_agg = RTFAggregator()
        self.rtf_selector = RTFSelector(dim)
        self.relation = ProfileRelation(dim=dim)
        self.separator = FastFiLMResNetSeparator(dim=dim)

    def encode_profiles(self, enroll):
        # enroll: [B,4,K,8,T], or [B,4,8,T] for K=1.
        if enroll.ndim == 4:
            enroll = enroll[:, :, None]
        b, nseat, k, c, t = enroll.shape
        raw = self.profile(enroll.reshape(b * nseat * k, c, t)).reshape(b, nseat, k, -1)
        z_id = F.normalize(self.id_proj(raw), dim=-1)
        z_seat = self.seat_proj(raw)
        z_sep = self.sep_drop(self.sep_proj(raw))
        return raw, z_id, z_seat, z_sep

    def forward(self, enroll, mix, target_seat):
        # enroll: [B,4,K,8,T], mix: [B,8,T], target_seat: [B]
        if enroll.ndim == 4:
            enroll = enroll[:, :, None]
        b, nseat, k, c, t = enroll.shape
        raw, z_id, z_seat, z_sep = self.encode_profiles(enroll)
        z_semantic = z_seat.mean(2)
        r, seat_logits, pair_logits = self.relation(z_semantic)
        idx = torch.arange(b, device=raw.device)
        z_target = F.normalize(z_sep[idx, target_seat].mean(1), dim=-1)
        r_target = r[idx, target_seat]
        rtf_all = estimate_rtf(enroll.reshape(b * nseat * k, c, t), self.profile.win)
        rtf_tokens = rtf_all.reshape(b, nseat, k, 7, -1)
        # Learn a target-conditioned soft selection over enrollment RTF candidates.
        selected=[]
        for s in range(nseat):
            zs=F.normalize(z_sep[:, s].mean(1), dim=-1)
            hs,_=self.rtf_selector(rtf_tokens[:, s], zs)
            selected.append(hs)
        rtf_agg=torch.stack(selected,1)
        rtf_target = rtf_agg[idx, target_seat]
        sep_tokens = F.normalize(z_sep[idx, target_seat], dim=-1)
        rtf_candidates = rtf_tokens[idx, target_seat]
        y_mask, Ym, mask, y_mvdr, rtf_refined, rtf_sel = self.separator(mix, z_target, r_target, rtf_target, sep_tokens, rtf_candidates)
        return {
            'y': y_mvdr, 'y_mask': y_mask, 'y_mvdr': y_mvdr,
            'rtf_refined': rtf_refined, 'rtf_sel': rtf_sel,
            'Ym': Ym, 'mask': mask,
            'raw': raw, 'z_id': z_id, 'z_seat': z_seat, 'z_sep': z_sep,
            'z_semantic': z_semantic, 'r': r, 'rtf': rtf_tokens, 'rtf_agg': rtf_agg,
            'seat_logits': seat_logits,
            'pair_logits': pair_logits,
        }


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def smoke_test():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    model = JointSeatProfileSeparator().to(dev)
    b = int(os.environ.get('SMOKE_BATCH', '2'))
    enroll = torch.randn(b, 4, 8, 48000, device=dev)
    mix = torch.randn(b, 8, 48000, device=dev)
    target = torch.tensor([i % 4 for i in range(b)], device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    if dev == 'cuda': torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(dev == 'cuda')):
        out = model(enroll, mix, target)
        # synthetic profile labels: one class per car-seat; two views per class
        labels = torch.arange(b, device=dev).repeat_interleave(2)
        z_dummy = torch.cat([out['z_id'][:, 0, 0], out['z_id'][:, 0, 0]], 0)
        groups = torch.arange(b, device=dev).repeat_interleave(2)
        seat_order = torch.arange(4, device=dev)[None].expand(b, -1)
        pair_targets = make_pair_targets(seat_order, dev)
        # smoke objective exercises every loss branch
        loss = out['y'].float().square().mean()
        loss = loss + 0.01 * seat_classification_loss(out['seat_logits'], torch.arange(4, device=dev)[None].expand(b, -1))
        loss = loss + 0.01 * pair_relation_loss(out['pair_logits'], pair_targets)
        loss = loss + 0.01 * supcon_loss(z_dummy, labels, groups)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    grad_norm = lambda mod: sum(float(p.grad.detach().float().norm()) for p in mod.parameters() if p.grad is not None)
    pgrad, rgrad, sgrad = grad_norm(model.profile), grad_norm(model.relation), grad_norm(model.separator)
    opt.step()
    if dev == 'cuda':
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1e9
    else:
        peak = 0.0
    print('profile_params', count_params(model.profile))
    print('relation_params', count_params(model.relation))
    print('separator_params', count_params(model.separator))
    print('total_params', count_params(model))
    print('shapes', tuple(out['y'].shape), tuple(out['z_id'].shape),
          tuple(out['seat_logits'].shape), tuple(out['pair_logits'].shape))
    print('finite', all(bool(torch.isfinite(v.float()).all()) for k, v in out.items()
                        if torch.is_tensor(v) and v.is_floating_point()))
    print('z_shapes', tuple(out['z_id'].shape), tuple(out['z_seat'].shape))
    print('losses', float(loss.detach()), float(supcon_loss(z_dummy, labels, groups).detach()))
    print('grad_norms', round(pgrad, 4), round(rgrad, 4), round(sgrad, 4))
    print('step_seconds', round(time.time() - t0, 3))
    print('peak_gpu_gb', round(peak, 3))


if __name__ == '__main__':
    smoke_test()
