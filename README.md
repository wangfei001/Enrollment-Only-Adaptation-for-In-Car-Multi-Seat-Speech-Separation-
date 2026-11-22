# SPR-SepNet Inference Package

Code-only release for:

**Enrollment-Only Adaptation for In-Car Multi-Seat Speech Separation via Seat Acoustic Registration**

No audio files and no checkpoint weights are included.

## Included methods

- SPR-SepNet main chain: MSFE + SRSN-Z
- Adapted Conv-TasNet-IC: seat-identifier and registration variants
- Adapted So-DAS: GT-DOA and RegPhase variants

The internal SPR-SepNet methods are causal and support streaming inference.
Conv-TasNet-IC and So-DAS are adapted with non-causal configurations and are
evaluated offline on complete 3-s chunks.

## Directory layout

```text
inference/
  run_main.py       # MSFE + SRSN-Z
  run_ic.py         # adapted Conv-TasNet-IC
  run_sodas.py      # adapted So-DAS
  common.py
src/
  main_model/       # SPR-SepNet model definitions
  ic_convtasnet/    # IC Conv-TasNet adapter
  sodas/            # adapted So-DAS model code
  baselines/        # minimal upstream IC Conv-TasNet dependencies
checkpoints/        # place released weights here; not included
demo_data/          # place the demo recording here; not included
```

See `checkpoints/README.md` and `demo_data/README.md` for the expected file
names and input layout.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the main model

```bash
python inference/run_main.py \
  --demo demo_data/nissan_qashqai/set1_main_mix_interf2 \
  --teacher checkpoints/teacher_msfe.pt \
  --srsn checkpoints/srsn_z.pt \
  --out-dir outputs/main
```

Outputs:

```text
outputs/main/<scene>_msfe.wav
outputs/main/<scene>_mask.wav
outputs/main/<scene>_srsn_z.wav
```

## Run Conv-TasNet-IC

Registration-conditioned residual version:

```bash
python inference/run_ic.py \
  --demo demo_data/nissan_qashqai/set1_main_mix_interf2 \
  --teacher checkpoints/teacher_msfe.pt \
  --checkpoint checkpoints/ic_convtasnet_registration.pt \
  --variant registration \
  --out outputs/ic_registration.wav
```

Seat-identifier version:

```bash
python inference/run_ic.py \
  --demo demo_data/nissan_qashqai/set1_main_mix_interf2 \
  --checkpoint checkpoints/ic_convtasnet_seat_id.pt \
  --variant seat_id \
  --out outputs/ic_seat_id.wav
```

## Run So-DAS

Registration-phase version:

```bash
python inference/run_sodas.py \
  --demo demo_data/nissan_qashqai/set1_main_mix_interf2 \
  --teacher checkpoints/teacher_msfe.pt \
  --checkpoint checkpoints/sodas_reg_phase.pt \
  --condition reg_phase \
  --out outputs/sodas_reg_phase.wav
```

GT-DOA version requires a JSON file containing `[x, y]` direction values:

```bash
python inference/run_sodas.py \
  --demo demo_data/nissan_qashqai/set1_main_mix_interf2 \
  --checkpoint checkpoints/sodas_gt_doa.pt \
  --condition gt_doa \
  --doa path/to/doa.json \
  --out outputs/sodas_gt_doa.wav
```

## Input contract

- Sample rate: 16 kHz
- 8 channels per recording
- Enrollment: 4 seats x 5 clips x 3 s
- Target seat in demo metadata: seat 1 (driver)
- Interfering seat in demo metadata: seat 2 (front passenger)

See `THIRD_PARTY_NOTICES.md` before publishing adapted third-party code.
