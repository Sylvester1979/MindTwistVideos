"""Episode 3 mix: voice + music (ducked, dip after «Κανείς τους δεν γύρισε», natural outro) + SFX by shot
-> audio/mix.wav (-14 LUFS, TP -1.5).  usage: MT_ROOT=/home/user/ep03 python3 mix.py"""
import json, subprocess, numpy as np, os
ROOT = os.environ.get('MT_ROOT', '/home/user/ep03')
SR = 48000
TL = json.load(open(f'{ROOT}/timeline.json'))
T, VOFF = TL['T'], TL.get('voff', 0.6)
SH = {s['k']: s for s in TL['shots']}
N = int((T + 1) * SR)


def S0(k): return SH[k]['t0']
def S1(k): return SH[k]['t1']


def load(path, ch=2):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-ac', str(ch), '-ar', str(SR), '-f', 'f32le', '-'],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, ch).copy()


def db(x): return 10 ** (x / 20)


def env(points, n=N):
    t = np.arange(n) / SR
    ts, gs = zip(*points)
    return np.interp(t, ts, gs).astype(np.float32)


out = np.zeros((N, 2), np.float32)


def place(x, t, gain=1.0, fin=0.0, fout=0.0, dur=None):
    if isinstance(x, str) and not os.path.exists(x): return
    a = load(x) if isinstance(x, str) else x.copy()
    if dur is not None: a = a[:int(dur * SR)]
    a = a * gain
    if fin > 0: n = min(len(a), int(fin * SR)); a[:n] *= np.linspace(0, 1, n)[:, None]
    if fout > 0: n = min(len(a), int(fout * SR)); a[-n:] *= np.linspace(1, 0, n)[:, None]
    i0 = int(t * SR)
    if i0 < 0: a = a[-i0:]; i0 = 0
    i1 = min(N, i0 + len(a)); out[i0:i1] += a[:i1 - i0]


def bed(path, t0, t1, gain, fin=0.8, fout=1.0):
    """sfx bed from t0 to t1 (looped with a crossfade if the file is shorter)."""
    if not os.path.exists(path): return
    a = load(path); n = int((t1 - t0) * SR)
    if n <= 0: return
    x = 1.0 * SR
    b = a.copy()
    while len(b) < n:
        xf = int(min(x, len(a) // 3))
        w = np.linspace(0, 1, xf)[:, None]
        b = np.concatenate([b[:-xf], b[-xf:] * (1 - w) + a[:xf] * w, a[xf:]])
    place(b[:n], t0, gain, fin, fout)


S = f'{ROOT}/sfx'
# ---------------- voice
voice = load(f'{ROOT}/audio/voice_edit.wav', 1)[:, 0]
vo = np.zeros(N, np.float32)
i0 = int(VOFF * SR); n = min(N - i0, len(voice)); vo[i0:i0 + n] = voice[:n]
out += vo[:, None]

# ---------------- music: head from 0, crossfade into the natural outro so it ends with the video
mus = load(f'{ROOT}/audio/music.wav')
L = len(mus) / SR
OUT_SRC = 147.0                       # sparse outro starts here in the track
tB = T + 0.5 - (L - 1.5 - OUT_SRC)    # outro placed so its decay lands at the end
XF = 3.0
m = np.zeros((N, 2), np.float32)
a_end = int((tB + XF) * SR)
m[:min(N, a_end)] = mus[:min(N, a_end)]
w = np.clip((np.arange(N) / SR - tB) / XF, 0, 1)[:, None].astype(np.float32)
m *= (1 - w)
b = mus[int(OUT_SRC * SR):]
j0 = int(tB * SR); nb = min(N - j0, len(b))
m[j0:j0 + nb] += b[:nb] * w[j0:j0 + nb]
# ducking under the voice
va = np.convolve(np.abs(vo), np.ones(SR // 20) / (SR // 20), 'same')
act = (va > 0.01).astype(np.float32)
k = int(0.35 * SR); act = np.convolve(act, np.ones(k) / k, 'same')
duck = 1 - 0.45 * np.clip(act * 1.6, 0, 1)
tN = S1('F03_empty') - 0.3             # after «Κανείς τους δεν γύρισε.»
tE = S0('H01_flowers')
auto = env([(0, 0.0), (0.05, 0.0), (1.5, 1.0),
            (tN - 2.2, 1.0), (tN - 1.4, db(-12)), (tN + 0.6, db(-12)), (tN + 1.8, 1.0),
            (tE, 1.0), (T - 3.0, 1.25), (T - 0.3, 1.25), (T + 0.5, 0.0), (N / SR, 0.0)])
m *= (duck * auto)[:, None] * db(-16)
out += m

# ---------------- SFX (quiet, realistic; missing files are skipped)
f = lambda name: f'{S}/{name}.mp3'
bed(f('breeze'), 0.0, S1('A03_guard'), db(-24), 1.2, 1.0)                           # morning, window
place(f('water'), S0('A02_window') + 0.5, db(-15), 0.3, 1.0, dur=S1('A02_window') - S0('A02_window'))
place(f('pen'), S0('A04_diary') + 1.2, db(-17), 0.3, 1.2, dur=7.0)                   # «Εκείνο το βράδυ γράφει»
bed(f('hall'), S0('B02_orphans'), S1('B02_orphans'), db(-23), 0.8, 1.0)             # dining hall
place(f('whistle'), S0('C04_rails') + 0.8, db(-16), 0.5, 1.5, dur=S1('C04_rails') - S0('C04_rails') + 1.0)
bed(f('steps'), S0('E01_march') + 0.2, S1('E02_column'), db(-19), 1.0, 1.5)        # the march
bed(f('breeze'), S0('H01_flowers'), T + 0.5, db(-22), 1.5, 2.0)                     # the empty window

np.clip(out, -1, 1).astype(np.float32).tofile(f'{ROOT}/audio/mix_raw.f32')
subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'f32le', '-ar', str(SR), '-ac', '2', '-i', f'{ROOT}/audio/mix_raw.f32',
                '-af', 'loudnorm=I=-14:TP=-1.5:LRA=11,alimiter=limit=0.84:attack=1:release=60:level=disabled',
                '-ar', str(SR), f'{ROOT}/audio/mix.wav'], check=True)
print('mix ok', T)
