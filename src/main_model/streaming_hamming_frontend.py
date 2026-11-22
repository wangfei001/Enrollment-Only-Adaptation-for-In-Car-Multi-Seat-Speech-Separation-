"""Streaming-compatible Hamming STFT/ISTFT with symmetric zero padding.

Convention used everywhere in the low-latency fine-tune:
    left-pad by N_FFT - HOP zeros
    optionally right-pad to the next HOP boundary
    torch.stft(..., center=False)
    torch.istft(..., center=False), then trim the left padding
The first streaming frame can be produced after one HOP of real audio.
"""
import torch
import torch.nn.functional as F

from profile_fast_resnet_model import N_FFT, HOP


def causal_stft(x, window):
    """[B,C,T] or [B,T] -> complex [B,C,F,Frames]."""
    if x.ndim == 2:
        x = x[:, None]
    pad = N_FFT - HOP
    right = (HOP - x.shape[-1] % HOP) % HOP
    xp = F.pad(x, (pad, right))
    b, c, t = xp.shape
    X = torch.stft(
        xp.reshape(b * c, t), n_fft=N_FFT, hop_length=HOP,
        window=window, return_complex=True, center=False,
    )
    return X.reshape(b, c, X.shape[-2], X.shape[-1])


def causal_istft(Y, length, window):
    """Inverse of causal_stft for [B,F,T] complex spectra."""
    if Y.ndim == 2:
        Y = Y[None]
    pad = N_FFT - HOP
    padded_len = length + pad
    y = torch.istft(
        Y, n_fft=N_FFT, hop_length=HOP, window=window,
        length=padded_len, center=False,
    )
    return y[..., pad:pad + length]


class StreamingCausalSTFT:
    """Block-fed STFT using the same symmetric zero-padding convention."""

    def __init__(self, window):
        self.window = window
        self.pad = N_FFT - HOP
        self.buf = None
        self.total = 0
        self.next_frame = 0

    def _emit_frame(self, start):
        frame = self.buf[..., start:start + N_FFT] * self.window
        b, c, n = frame.shape
        spec = torch.fft.rfft(frame.reshape(b * c, n), dim=-1)
        return spec.reshape(b, c, spec.shape[-1])

    def push(self, audio):
        if audio.ndim == 2:
            audio = audio[:, None]
        if self.buf is None:
            self.buf = self.window.new_zeros(audio.shape[0], audio.shape[1], self.pad)
        self.buf = torch.cat([self.buf, audio], -1)
        self.total += audio.shape[-1]
        out = []
        while self.next_frame * HOP + N_FFT <= self.buf.shape[-1]:
            out.append(self._emit_frame(self.next_frame * HOP))
            self.next_frame += 1
        return torch.stack(out, -1) if out else self.window.new_empty(0)

    def finish(self):
        if self.buf is None:
            return self.window.new_empty(0)
        add = (HOP - self.total % HOP) % HOP
        if add:
            self.buf = F.pad(self.buf, (0, add))
        out = []
        while self.next_frame * HOP + N_FFT <= self.buf.shape[-1]:
            out.append(self._emit_frame(self.next_frame * HOP))
            self.next_frame += 1
        return torch.stack(out, -1) if out else self.window.new_empty(0)


class StreamingOLAISTFT:
    """Block-fed overlap-add ISTFT for the same symmetric zero padding."""

    def __init__(self, window):
        self.window = window
        self.pad = N_FFT - HOP
        self.ola = None
        self.wsum = None
        self.next_frame = 0
        self.emit_padded = self.pad

    def _ensure(self, end, batch, device, dtype):
        if self.ola is None:
            self.ola = torch.zeros(batch, end, device=device, dtype=dtype)
            self.wsum = torch.zeros(batch, end, device=device, dtype=torch.float32)
        elif self.ola.shape[1] < end:
            n = end - self.ola.shape[1]
            self.ola = F.pad(self.ola, (0, n))
            self.wsum = F.pad(self.wsum, (0, n))

    def _emit_ready(self, end):
        end = min(end, self.ola.shape[1])
        if end <= self.emit_padded:
            return self.window.new_empty(0)
        y = self.ola[:, self.emit_padded:end] / self.wsum[:, self.emit_padded:end].clamp_min(1e-8)
        self.emit_padded = end
        return y

    def push(self, Y):
        if Y.shape[-1] == 0:
            return self.window.new_empty(0)
        outputs = []
        for j in range(Y.shape[-1]):
            m = self.next_frame
            frame = torch.fft.irfft(Y[..., j], n=N_FFT, dim=-1) * self.window
            start = m * HOP
            self._ensure(start + N_FFT, frame.shape[0], frame.device, frame.dtype)
            self.ola[:, start:start + N_FFT] += frame
            self.wsum[:, start:start + N_FFT] += self.window.square()
            self.next_frame += 1
            outputs.append(self._emit_ready(start))
        outputs = [x for x in outputs if x.shape[-1] > 0]
        return torch.cat(outputs, -1) if outputs else self.window.new_empty(0)

    def finish(self, length):
        target = self.pad + length
        self._ensure(target, self.ola.shape[0], self.ola.device, self.ola.dtype)
        return self._emit_ready(target)
