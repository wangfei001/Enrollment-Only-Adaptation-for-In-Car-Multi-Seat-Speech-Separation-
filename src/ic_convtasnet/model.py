#!/usr/bin/env python3
"""IC Conv-TasNet with one registration FiLM layer.

Upstream implementation:
  donghoney0416/IC_Conv-TasNet
  "Inter-channel Conv-TasNet for multichannel speech enhancement"

The upstream source is kept unchanged.  This adapter subclasses its TasNet and
inserts one FiLM layer after the shared waveform encoder.  It remains offline
(non-causal), matching the upstream default.
"""
from pathlib import Path
import sys

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "baselines" / "IC_Conv-TasNet"
sys.path.insert(0, str(UPSTREAM))

from config import config as upstream_config  # noqa: E402
from conv_tasnet_ic import TasNet as UpstreamICTasNet  # noqa: E402


class FiLM(nn.Module):
    def __init__(self, cond_dim: int, channels: int, cond_mode: str = "registration", num_seats: int = 4):
        super().__init__()
        self.cond_mode = cond_mode
        if cond_mode == "seat_id":
            self.seat_embedding = nn.Embedding(num_seats, cond_dim)
            self.norm = None
        elif cond_mode == "registration":
            self.seat_embedding = None
            self.norm = nn.LayerNorm(cond_dim)
        else:
            raise ValueError(f"unknown cond_mode: {cond_mode}")
        self.proj = nn.Linear(cond_dim, channels * 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if self.cond_mode == "seat_id":
            c = self.seat_embedding(cond.long().reshape(-1))
        else:
            c = self.norm(cond.float())
        gamma, beta = self.proj(c).chunk(2, dim=-1)
        return x * (1.0 + gamma[:, None, :, None]) + beta[:, None, :, None]


class FiLMICTasNet(UpstreamICTasNet):
    def __init__(
        self,
        mic_num: int = 8,
        enc_dim: int = 512,
        feature_dim: int = 128,
        ch_dim: int = 8,
        win_ms: int = 16,
        layer: int = 8,
        stack: int = 1,
        kernel: int = 3,
        cond_dim: int = 512,
        cond_mode: str = "registration",
        num_seats: int = 4,
        causal: bool = False,
    ):
        # The upstream constructor reads these values from its global config.
        # Set them only for construction; the upstream source itself is untouched.
        upstream_config.model_mic_num = int(mic_num)
        upstream_config.model_enc_dim = int(enc_dim)
        upstream_config.model_feature_dim = int(feature_dim)
        upstream_config.model_ch_dim = int(ch_dim)
        upstream_config.model_win = int(win_ms)
        upstream_config.model_layer = int(layer)
        upstream_config.model_stack = int(stack)
        upstream_config.model_kernel = int(kernel)
        upstream_config.model_num_spk = 1
        upstream_config.model_causal = bool(causal)
        super().__init__()
        self.film = FiLM(cond_dim, self.enc_dim, cond_mode=cond_mode, num_seats=num_seats)

    def forward(self, input: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        output, rest = self.pad_signal(input)
        batch_size = output.size(0)
        num_ch = output.size(1)
        enc = self.encoder(output.reshape(batch_size * num_ch, 1, -1))
        enc = enc.reshape(batch_size, num_ch, self.enc_dim, -1)
        enc = self.film(enc, cond)

        mask = torch.sigmoid(self.TCN(enc)).reshape(
            batch_size, self.num_spk, self.enc_dim, -1,
        )
        ref = upstream_config.reference_channel_idx
        masked = enc[:, ref:ref + 1] * mask
        out = self.decoder(masked.reshape(batch_size * self.num_spk, self.enc_dim, -1))
        out = out[:, :, self.stride:-(rest + self.stride)].contiguous()
        return out.reshape(batch_size, self.num_spk, -1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
