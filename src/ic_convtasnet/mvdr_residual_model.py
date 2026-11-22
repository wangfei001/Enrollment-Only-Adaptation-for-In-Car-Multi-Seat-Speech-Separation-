#!/usr/bin/env python3
"""IC Conv-TasNet with registration/position embedding, MVDR auxiliary input and residual output."""
from __future__ import annotations
import sys
from pathlib import Path
import torch
import torch.nn as nn
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from .model import FiLMICTasNet

class ICConvTasNetMVDRResidual(nn.Module):
    def __init__(self,mic_num=8,residual_bound=0.5,enc_dim=512,feature_dim=128,
                 ch_dim=9,win_ms=16,layer=8,stack=1,kernel=3,cond_dim=512,cond_mode='registration'):
        super().__init__(); self.residual_bound=float(residual_bound)
        if cond_mode != 'registration':
            raise ValueError('MVDR residual is registration-conditioned only; run seat_id without MVDR')
        # The ninth channel is the frozen/calibration MVDR waveform.
        self.net=FiLMICTasNet(mic_num=mic_num+1,enc_dim=enc_dim,feature_dim=feature_dim,
                              ch_dim=ch_dim,win_ms=win_ms,layer=layer,stack=stack,
                              kernel=kernel,cond_dim=cond_dim,cond_mode=cond_mode,
                              num_seats=4,causal=False)
        # Start exactly at the frozen MVDR output and learn the residual.
        nn.init.zeros_(self.net.decoder.weight)
    def forward(self,mix,mvdr,cond):
        x=torch.cat([mix,mvdr[:,None]],dim=1)
        raw=self.net(x,cond).squeeze(1)
        mask=self.residual_bound*torch.tanh(raw)
        y=mvdr+mask*mix[:,0]
        return y,mask,raw

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
