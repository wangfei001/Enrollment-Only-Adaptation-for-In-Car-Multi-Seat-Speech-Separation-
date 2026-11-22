#!/usr/bin/env python3
"""Run the adapted Conv-TasNet-IC baselines in offline 3-s chunks."""
from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf
import torch

from common import FS, default_geometry, load_demo, load_state, load_teacher, selected_mask_cond
from src.main_model.mvdr_stream_model import stream_separation_forward
from src.ic_convtasnet.model import FiLMICTasNet
from src.ic_convtasnet.mvdr_residual_model import ICConvTasNetMVDRResidual


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--demo', default='demo_data/nissan_qashqai/set1_main_mix_interf2')
    p.add_argument('--teacher', default='checkpoints/teacher_msfe.pt')
    p.add_argument('--checkpoint', default='checkpoints/ic_convtasnet_registration.pt')
    p.add_argument('--variant', choices=['registration', 'seat_id'], default='registration')
    p.add_argument('--chunk-sec', type=float, default=3.0)
    p.add_argument('--out', default='outputs/ic_convtasnet.wav')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


@torch.inference_mode()
def main():
    a = parse_args()
    device = torch.device(a.device)
    meta, enroll, mix = load_demo(a.demo)
    enroll, mix = enroll.to(device), mix.to(device)
    seat = torch.tensor([int(meta['target_seat']) - 1], device=device)
    if a.variant == 'seat_id':
        model = FiLMICTasNet(mic_num=8, enc_dim=512, feature_dim=128, ch_dim=8,
                             win_ms=16, layer=8, stack=1, kernel=3, cond_dim=512,
                             cond_mode='seat_id', num_seats=4, causal=False).to(device)
    else:
        model = ICConvTasNetMVDRResidual(mic_num=8, residual_bound=0.5, enc_dim=512,
                                         feature_dim=128, ch_dim=9, win_ms=16,
                                         layer=8, stack=1, kernel=3, cond_dim=512,
                                         cond_mode='registration').to(device)
    model.load_state_dict(load_state(a.checkpoint, device)); model.eval()
    teacher, window = load_teacher(a.teacher, device)
    reg = teacher.encode_registration(enroll)
    cond = selected_mask_cond(teacher, reg, seat)
    geo = default_geometry(device)
    chunk = int(a.chunk_sec * FS)
    outputs = []
    for start in range(0, mix.shape[-1], chunk):
        x = mix[..., start:min(start + chunk, mix.shape[-1])].float()
        if a.variant == 'seat_id':
            y = model(x, seat).squeeze(1).float()
        else:
            tout = stream_separation_forward(teacher, reg, x, seat, geo,
                                             chunk_frames=10, window=window,
                                             attention_window_ms=500,
                                             target_window_ms=500,
                                             noise_window_ms=750,
                                             rtf_update_frames=10)
            y = model(x, tout['y'], cond).float()
        outputs.append(y.cpu())
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    y = torch.cat(outputs, -1)[0, :mix.shape[-1]].numpy()
    sf.write(str(out), y, FS, subtype='PCM_16')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
