"""Ep05 voice treatment: tame harshness and sibilance so the narration sits with the darker music.
Measured on voice_edit.wav (1/3-octave LTAS, re 400 Hz-1 kHz) against the music:
  voice is +7..+12 dB brighter at 1.6-4 kHz, +14..+21 dB at 5-8 kHz, narrow resonance cluster 7.0-7.4 kHz,
  and 12.8 % of speech frames are sibilant (5-10 kHz within 3 dB of 0.3-3 kHz).
Treatment (STFT, 1024/256, scipy):
  1. static EQ: -2.5 dB bell 3.3 kHz (Q 1.4), -3 dB bell 7.2 kHz (Q 2.5), -2 dB high shelf from 9 kHz
  2. dynamic de-esser: when 5-10 kHz rises above (0.3-3 kHz - 6 dB), pull 4.5-12 kHz down by up to 8 dB
     (attack ~5 ms, release ~60 ms), only on those moments: consonants keep their shape, no lisp.
  voicefx05.py  -> $MT_ROOT/audio/voice_fx.wav (+ report)"""
import os, sys, subprocess
import numpy as np
from scipy.signal import stft, istft
ROOT = os.environ.get('MT_ROOT', '/home/claude/ep05')
SR = 48000
NF, HOP = 1024, 256

def load(p):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', p, '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-'], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).astype(np.float64)

def bell(f, f0, g, q):
    x = np.log2(np.maximum(f, 1) / f0) * q * 2
    return g * np.exp(-x * x)

def shelf(f, f0, g, w=0.5):
    return g / (1 + np.exp(-np.log2(np.maximum(f, 1) / f0) / w * 4))

def treat(x, eq=True, deess=True, thr=-6.0, slope=0.75, maxred=8.0):
    f, t, Z = stft(x, SR, nperseg=NF, noverlap=NF - HOP)
    P = np.abs(Z) ** 2
    G = np.zeros_like(f)
    if eq:
        G += bell(f, 3300, -2.5, 1.4) + bell(f, 7200, -3.0, 2.5) + shelf(f, 9000, -2.0)
    Zo = Z * (10 ** (G / 20))[:, None]
    red = np.zeros(len(t))
    if deess:
        hi = P[(f > 5000) & (f < 10000)].sum(0); mid = P[(f > 300) & (f < 3000)].sum(0)
        R = 10 * np.log10((hi + 1e-12) / (mid + 1e-12))
        lvl = 10 * np.log10(P.sum(0) + 1e-12)
        speech = lvl > lvl.max() - 45
        target = np.clip((R - thr) * slope, 0, maxred) * speech
        # attack/release smoothing (frames of 5.3 ms)
        a_att, a_rel = np.exp(-1 / 1.0), np.exp(-1 / 11.0)
        g = 0.0
        for i, v in enumerate(target):
            a = a_att if v > g else a_rel
            g = a * g + (1 - a) * v
            red[i] = g
        band = np.clip((f - 3500) / 1000, 0, 1) * np.clip((14000 - f) / 2000, 0, 1)      # 4.5-12 kHz, soft edges
        Zo = Zo * 10 ** (-(band[:, None] * red[None, :]) / 20)
    _, y = istft(Zo, SR, nperseg=NF, noverlap=NF - HOP)
    return y[:len(x)], red, t

def report(x, tag):
    f, t, Z = stft(x, SR, nperseg=4096, noverlap=3072)
    P = np.abs(Z) ** 2
    lv = 10 * np.log10(P.sum(0) + 1e-12); P = P[:, lv > lv.max() - 35]
    def b(lo, hi): return P[(f >= lo) & (f < hi)].mean() if ((f >= lo) & (f < hi)).any() else 1e-12
    ref = b(400, 1000)
    hi = P[(f > 5000) & (f < 10000)].sum(0); mid = P[(f > 300) & (f < 3000)].sum(0)
    r = 10 * np.log10(hi / (mid + 1e-12) + 1e-12)
    bands = [(1600, 2500), (2500, 4000), (4000, 6300), (6300, 9000), (9000, 12500)]
    s = '  '.join(f'{lo/1000:.1f}-{hi_/1000:.1f}k {10*np.log10(b(lo, hi_)/ref):6.1f}' for lo, hi_ in bands)
    print(f'{tag:8s} {s} | sibilant frames {100*np.mean(r > -3):4.1f}%  95th {np.percentile(r, 95):5.1f} dB')

if __name__ == '__main__':
    x = load(f'{ROOT}/audio/voice_edit.wav')
    y, red, t = treat(x)
    report(x, 'before'); report(y, 'after')
    print(f'de-esser: active {100*np.mean(red > 1):.1f}% of frames, mean reduction when active {red[red > 1].mean():.1f} dB, max {red.max():.1f} dB')
    y = y * (np.sqrt(np.mean(x ** 2)) / np.sqrt(np.mean(y ** 2)))      # keep the level
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'f64le', '-ar', str(SR), '-ac', '1', '-i', '-', '-c:a', 'pcm_s24le',
                    f'{ROOT}/audio/voice_fx.wav'], input=y.astype(np.float64).tobytes(), check=True)
    print('->', f'{ROOT}/audio/voice_fx.wav')
