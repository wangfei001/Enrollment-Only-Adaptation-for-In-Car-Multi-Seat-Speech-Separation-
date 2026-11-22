#!/usr/bin/env python3
"""Run the adapted So-DAS baseline in offline 3-s chunks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf
import torch

from common import FS, load_demo, load_state, load_teacher, selected_mask_cond
from src.sodas.sodas_model_adapter import MISO2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--demo', default='demo_data/nissan_qashqai/set1_main_mix_interf2')
    p.add_argument('--teacher', default='checkpoints/teacher_msfe.pt')
    p.add_argument('--checkpoint', default='checkpoints/sodas_reg_phase.pt')
    p.add_argument('--condition', choices=['reg_phase', 'gt_doa'], default='reg_phase')
    p.add_argument('--doa', default='', help='JSON file containing [x, y] DOA values for gt_doa')
    p.add_argument('--chunk-sec', type=float, default=3.0)
    p.add_argument('--out', default='outputs/sodas.wav')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


@torch.inference_mode()
def main():
    a = parse_args()
    device = torch.device(a.device)
    meta, enroll, mix = load_demo(a.demo)
    enroll, mix = enroll.to(device), mix.to(device)
    seat = torch.tensor([int(meta['target_seat']) - 1], device=device)
    model = MISO2(use_dense=False, use_att=[True, True, True, True, False],
                  use_rnn=[True, True, False, False, False], width=32,
                  num_layers=1, dropout_rate=0.4, causal_conf=False,
                  num_spks=1, frame_len=512, frame_hop=128,
                  target_ref_only=True, condition_mode=a.condition).to(device)
    model.load_state_dict(load_state(a.checkpoint, device)); model.eval()
    teacher = reg_cond = None
    if a.condition == 'reg_phase':
        teacher, _ = load_teacher(a.teacher, device)
        reg = teacher.encode_registration(enroll)
        reg_cond = selected_mask_cond(teacher, reg, seat)
    elif not a.doa:
        raise ValueError('--doa is required for the GT-DOA checkpoint')
    doa = torch.tensor(json.loads(Path(a.doa).read_text()), dtype=torch.float32, device=device)
    chunk = int(a.chunk_sec * FS)
    outputs = []
    for start in range(0, mix.shape[-1], chunk):
        x = mix[..., start:min(start + chunk, mix.shape[-1])].float()
        cond = reg_cond if a.condition == 'reg_phase' else doa
        y = model(x, cond).float()[:, 0, 0]
        outputs.append(y.cpu())
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    y = torch.cat(outputs, -1)[0, :mix.shape[-1]].numpy()
    sf.write(str(out), y, FS, subtype='PCM_16')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
