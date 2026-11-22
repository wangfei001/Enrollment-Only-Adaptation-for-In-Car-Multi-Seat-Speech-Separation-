#!/usr/bin/env python3
"""Shared inference utilities for the SPR-SepNet release package."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.main_model.mvdr_stream_model import (  # noqa: E402
    AlignedSeatProfileMVDR,
    load_compatible_state_dict,
    set_model_window,
)
from src.main_model.student3_crn_model import Student3CRN  # noqa: E402

FS = 16000
DEFAULT_PEAK = 0.9


def load_state(path: str | Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device)
    return checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint


def peak_normalize(x: np.ndarray, target_peak: float = DEFAULT_PEAK):
    peak = float(np.max(np.abs(x)))
    if peak <= 1e-8:
        raise ValueError('empty or silent audio')
    scale = target_peak / peak
    return (x * scale).astype(np.float32)


def load_demo(demo_dir: str | Path, target_peak: float = DEFAULT_PEAK):
    """Load one demo scene and return normalized enrollment and mixture tensors."""
    demo_dir = Path(demo_dir)
    meta = json.loads((demo_dir / 'metadata.json').read_text(encoding='utf-8'))
    mix_channels = [sf.read(demo_dir / f'mix/ch{i}.wav', dtype='float32')[0] for i in range(1, 9)]
    n = min(map(len, mix_channels))
    mix = np.stack([x[:n] for x in mix_channels], axis=1)
    mix = peak_normalize(mix, target_peak)

    clips = []
    for seat in range(1, 5):
        seat_clips = []
        for k in range(1, 6):
            x, sr = sf.read(demo_dir / f'enrollment/seat{seat}/clip{k}.wav',
                            dtype='float32', always_2d=True)
            if sr != FS or x.shape[1] != 8:
                raise RuntimeError(f'bad enrollment clip: seat{seat}/clip{k}.wav')
            seat_clips.append(peak_normalize(x.T, target_peak))
        clips.append(np.stack(seat_clips, axis=0))
    enroll = np.stack(clips, axis=0)
    return meta, torch.from_numpy(enroll[None]), torch.from_numpy(mix.T[None])


def default_geometry(device: torch.device):
    """Coarse seat-to-mic geometry prior used by the deployed inference session."""
    g = [
        [[-0.24370122, 0.08749496, -0.02344567, 0.25999108]] * 5,
        [[-0.24370122, -0.09183399, -0.02344567, 0.26148319]] * 5,
        [[-0.51006615, 0.08749496, -0.02344567, 0.51804686]] * 5,
        [[-0.51006615, -0.09183399, -0.02344567, 0.51879734]] * 5,
    ]
    return torch.tensor(g, dtype=torch.float32, device=device)[None]


def load_teacher(checkpoint: str | Path, device: torch.device):
    teacher = AlignedSeatProfileMVDR(rtf_decay=0.99, rtf_tau=5.0, rtf_stride=10).to(device)
    state = load_state(checkpoint, device)
    load_compatible_state_dict(teacher, state)
    teacher.eval()
    window = set_model_window(teacher, 'hamming', device)
    return teacher, window


def load_srsn_z(checkpoint: str | Path, device: torch.device):
    model = Student3CRN(cond_dim=512, residual_bound=0.5).to(device)
    model.load_state_dict(load_state(checkpoint, device))
    model.eval()
    return model


def selected_mask_cond(teacher, registration, target_seat):
    zb = registration['z_prof'].shape[0]
    b = target_seat.shape[0]
    if zb == 1:
        zp = registration['z_prof'][0, target_seat]
        zd = registration['z_dir'][0, target_seat]
        rr = registration['r'][0, target_seat]
    elif zb == b:
        idx = torch.arange(b, device=target_seat.device)
        zp = registration['z_prof'][idx, target_seat]
        zd = registration['z_dir'][idx, target_seat]
        rr = registration['r'][idx, target_seat]
    else:
        raise ValueError('registration batch does not match target batch')
    return teacher.mask_cond_proj(torch.cat([zp, zd, rr], dim=-1))
