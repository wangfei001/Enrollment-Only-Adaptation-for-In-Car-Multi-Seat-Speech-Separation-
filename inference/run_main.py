#!/usr/bin/env python3
"""Run MSFE and SRSN-Z on one aligned 8-channel demo recording."""
from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf
import torch

from common import FS, default_geometry, load_demo, load_srsn_z, load_teacher
from src.main_model.mvdr_stream_model import StreamingMVDRSession
from src.main_model.streaming_hamming_frontend import StreamingCausalSTFT, StreamingOLAISTFT
from src.main_model.student3_crn_model import Student3StreamingState


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--demo', default='demo_data/nissan_qashqai/set1_main_mix_interf2')
    p.add_argument('--teacher', default='checkpoints/teacher_msfe.pt')
    p.add_argument('--srsn', default='checkpoints/srsn_z.pt')
    p.add_argument('--out-dir', default='outputs/main')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


@torch.inference_mode()
def main():
    a = parse_args()
    device = torch.device(a.device)
    teacher, window = load_teacher(a.teacher, device)
    student = load_srsn_z(a.srsn, device)
    meta, enroll, mix = load_demo(a.demo)
    enroll, mix = enroll.to(device), mix.to(device)
    reg = teacher.encode_registration(enroll)
    seat = torch.tensor([int(meta['target_seat']) - 1], device=device)
    cond0 = torch.zeros(1, 512, device=device)
    session = StreamingMVDRSession(teacher, reg, seat, default_geometry(device),
                                   attention_window_ms=500, target_window_ms=500,
                                   noise_window_ms=750, rtf_update_frames=10)
    state = Student3StreamingState(1, device)
    stft = StreamingCausalSTFT(window)
    ola_t, ola_m, ola_s = StreamingOLAISTFT(window), StreamingOLAISTFT(window), StreamingOLAISTFT(window)
    outs_t, outs_m, outs_s = [], [], []
    total = mix.shape[-1]
    for start in range(0, total, FS):
        x = mix[..., start:min(start + FS, total)]
        X = stft.push(x)
        if X.numel() == 0 or X.shape[-1] == 0:
            continue
        out = session.step(X)
        Y = out['Y'].to(torch.complex64)
        M = out['mask_ref'][:, 0] * X[:, 0]
        d = student.step(X, Y, cond0, state)
        Z = Y + d[:, 0] * X[:, 0]
        for ola, src, dst in ((ola_t, Y, outs_t), (ola_m, M, outs_m), (ola_s, Z, outs_s)):
            y = ola.push(src)
            if y.numel():
                dst.append(y.cpu())
    X = stft.finish()
    if X.numel() and X.shape[-1]:
        out = session.step(X)
        Y = out['Y'].to(torch.complex64)
        M = out['mask_ref'][:, 0] * X[:, 0]
        d = student.step(X, Y, cond0, state)
        Z = Y + d[:, 0] * X[:, 0]
        for ola, src, dst in ((ola_t, Y, outs_t), (ola_m, M, outs_m), (ola_s, Z, outs_s)):
            y = ola.push(src)
            if y.numel():
                dst.append(y.cpu())
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for name, ola, outs in (('msfe', ola_t, outs_t), ('mask', ola_m, outs_m), ('srsn_z', ola_s, outs_s)):
        outs.append(ola.finish(total).cpu())
        y = torch.cat([v for v in outs if v.numel()], -1)[..., :total]
        sf.write(out_dir / f'{meta["scene"]}_{name}.wav', y[0].numpy(), FS, subtype='PCM_16')
    print(f'wrote {out_dir}')


if __name__ == '__main__':
    main()
