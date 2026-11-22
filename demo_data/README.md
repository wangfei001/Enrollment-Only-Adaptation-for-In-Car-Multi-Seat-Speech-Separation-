# Demo Data

Audio is intentionally not included in the repository.

Expected structure for one aligned Nissan Qashqai 10-minute recording:

```text
demo_data/nissan_qashqai/set1_main_mix_interf2/
  metadata.json
  mix/ch1.wav ... ch8.wav
  enrollment/seat1/clip1.wav ... clip5.wav
  enrollment/seat2/clip1.wav ... clip5.wav
  enrollment/seat3/clip1.wav ... clip5.wav
  enrollment/seat4/clip1.wav ... clip5.wav
  transcript/asr.json
  transcript/asr.txt
```

All WAV files are 16 kHz. Each enrollment clip is 3 s and all four seats are
required because the registration encoder jointly processes the four seat
profiles and their relations.
