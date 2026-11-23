"""Ep05 mix: voice + music (one continuous take from 0: its quiet dip at 128-133 s falls on the countdown and
its swell on the reveal) + quiet realistic SFX placed on the frame events -> audio/mix.wav (-14 LUFS, TP -1.5).
A missing SFX file is skipped, so the mix works with or without the user's effects.
  usage: MT_ROOT=/home/claude/ep05 python3 mix05.py"""
import json, subprocess, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.environ.get('MT_ROOT', '/home/claude/ep05')
SR = 48000
TL = json.load(open(f'{ROOT}/timeline.json'))
T = TL['_end'][0]
N = int((T + 0.5) * SR)
def b0(k): return TL[k][0]
def b1(k): return TL[k][1]
def db(x): return 10 ** (x / 20)

def load(path, ch=2):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-ac', str(ch), '-ar', str(SR), '-f', 'f32le', '-'],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, ch).copy()

def load_norm(path, target_db=-20.0):
    a = load(path)
    fr = SR // 20; n = len(a) // fr
    if n == 0: return a
    r = np.sqrt((a[:n * fr] ** 2).mean(1).reshape(n, fr).mean(1) + 1e-12)
    act = np.sort(r)[int(n * 0.4):]
    return a * (db(target_db) / max(float(np.sqrt((act ** 2).mean())), 1e-6))

def env(points, n=N):
    t = np.arange(n) / SR
    ts, gs = zip(*points)
    return np.interp(t, ts, gs).astype(np.float32)

out = np.zeros((N, 2), np.float32)
LOG = []
SFX_TRIM = db(-10)          # channel taste (ep04): effects are background, not events

def onset(a, thr_db=-30):
    """first sample where the effect actually starts (align the *event*, not the file start)."""
    e = np.sqrt(np.convolve((a ** 2).mean(1), np.ones(SR // 200) / (SR // 200), 'same'))
    idx = np.where(e > db(thr_db) * e.max() / db(-6))[0]
    return idx[0] / SR if len(idx) else 0.0

def slow(a, k=2.0):
    """play k times slower: longer, an octave (k=2) deeper."""
    n = int(len(a) * k); x = np.linspace(0, len(a) - 1, n)
    return np.stack([np.interp(x, np.arange(len(a)), a[:, c]) for c in range(a.shape[1])], 1).astype(np.float32)

def place(path, t, gain_db=0.0, fin=0.0, fout=0.0, dur=None, skip=0.0, at_onset=True, k=1.0):
    if not os.path.exists(path): LOG.append(f'skip {os.path.basename(path)}'); return
    a = load_norm(path)
    if k != 1.0: a = slow(a, k)
    if at_onset: skip += onset(a)
    a = a[int(skip * SR):]
    if dur is not None: a = a[:int(dur * SR)]
    a = a * db(gain_db) * SFX_TRIM
    if fin > 0: n = min(len(a), int(fin * SR)); a[:n] *= np.linspace(0, 1, n)[:, None]
    if fout > 0: n = min(len(a), int(fout * SR)); a[-n:] *= np.linspace(1, 0, n)[:, None]
    i0 = int(t * SR)
    if i0 < 0: a = a[-i0:]; i0 = 0
    i1 = min(N, i0 + len(a)); out[i0:i1] += a[:i1 - i0]
    LOG.append(f'{os.path.basename(path):14s} {t:7.2f}s {len(a) / SR:5.1f}s {gain_db:6.1f} dB')

def bed(path, t0, t1, gain_db, fin=0.6, fout=0.8, skip=0.0):
    """effect from t0 to t1, looped with crossfades if the file is shorter."""
    if not os.path.exists(path): LOG.append(f'skip {os.path.basename(path)}'); return
    a = load_norm(path)[int(skip * SR):]
    n = int((t1 - t0) * SR)
    if n <= 0: return
    b = a.copy()
    while len(b) < n:
        xf = int(min(0.8 * SR, len(a) // 3)); w = np.linspace(0, 1, xf)[:, None]
        b = np.concatenate([b[:-xf], b[-xf:] * (1 - w) + a[:xf] * w, a[xf:]])
    b = b[:n] * db(gain_db) * SFX_TRIM
    if fin > 0: k = min(n, int(fin * SR)); b[:k] *= np.linspace(0, 1, k)[:, None]
    if fout > 0: k = min(n, int(fout * SR)); b[-k:] *= np.linspace(1, 0, k)[:, None]
    i0 = int(t0 * SR); i1 = min(N, i0 + n); out[i0:i1] += b[:i1 - i0]
    LOG.append(f'{os.path.basename(path):14s} {t0:7.2f}-{t1:6.2f}s       {gain_db:6.1f} dB')

# ---------------------------------------------------------------- voice (voice_edit.wav already starts at 0)
VFX = f'{ROOT}/audio/voice_fx.wav'                       # treated voice (voicefx05.py) if present
voice = load(VFX if os.path.exists(VFX) else f'{ROOT}/audio/voice_edit.wav', 1)[:, 0]
vo = np.zeros(N, np.float32); vo[:min(N, len(voice))] = voice[:N]
out += vo[:, None]

# ---------------------------------------------------------------- music
mus = load(f'{ROOT}/music/music_ep05.wav')
m = np.zeros((N, 2), np.float32); m[:min(N, len(mus))] = mus[:N]
va = np.convolve(np.abs(vo), np.ones(SR // 20) / (SR // 20), 'same')
act = (va > 0.01).astype(np.float32)
k = int(0.4 * SR); act = np.convolve(act, np.ones(k) / k, 'same')
duck = 1 - 0.6 * np.clip(act * 1.6, 0, 1)                       # about -8 dB under the voice, never silent
tc = b1('q_true') + 0.15                                       # countdown start
auto = env([(0, 0.0), (0.05, 0.6), (1.5, 1.0),
            (b1('a_93') + 0.2, 1.0), (b1('a_93') + 0.7, 1.35), (b0('b_1912') - 0.3, 1.35), (b0('b_1912') + 0.2, 1.0),   # bridge: card A
            (b1('b_oath') + 0.1, 1.0), (b1('b_oath') + 0.5, 1.3), (b0('q_so') - 0.1, 1.0),                           # bridge: card B
            (tc - 0.2, 1.0), (tc + 0.3, 1.5), (tc + 5.0, 1.5), (b0('r_true') + 0.2, 1.0),                              # the countdown
            (b1('r_cta'), 1.0), (b1('r_cta') + 0.6, 1.3), (T - 1.3, 1.3), (T, 0.0), (N / SR, 0.0)])
m *= (duck * auto)[:, None] * db(-12)
out += m

# ---------------------------------------------------------------- SFX
S = f'{ROOT}/sfx'
f = lambda name: next((f'{S}/{name}.{e}' for e in ('wav', 'mp3') if os.path.exists(f'{S}/{name}.{e}')), f'{S}/{name}.wav')
# cards and paper props sliding in
for t in (b0('hook1') - 0.25, b0('hook1') + 0.05, b0('a_who') - 0.35, b0('a_who') + 0.15, b0('a_trip') - 0.3, b0('a_home') - 0.4,
          b0('a_aug9') - 0.3, b0('a_boss') - 0.4, b1('a_93') + 0.5, b0('b_name') - 0.3, b0('b_again') - 0.3, b0('b_lusi') + 0.6,
          b1('b_oath') + 0.3, b0('r_only') + 0.6, b0('r_never') - 0.4, b0('r_never') - 0.15, b0('r_never') + 0.1,
          b0('r_legend') - 0.3, b0('r_legend')):
    place(f('paper'), t, -4, 0.0, 0.3)
# typewriter: dates, names, the cards being typed, time stamps
for t, d in ((b0('a_date') + 0.05, 1.4), (b0('a_who') + 0.5, 1.6), (b0('a_onebomb') + 0.05, 1.8), (b0('a_3km'), 0.6),
             (b1('a_93') + 0.6, 1.6), (b0('b_name') - 0.1, 1.3), (b0('b_ice') + 0.2, 0.5), (b0('b_fog') + 0.2, 0.5),
             (b0('b_torp') + 0.2, 0.5), (b1('b_oath') + 0.5, 1.4), (b1('r_lists') - 0.6, 0.7)):
    place(f('typewriter'), t, -6, 0.0, 0.25, dur=d)
# pen drawing
for t0, t1 in ((b0('a_japan') - 0.15, b1('a_date')), (b0('a_who') + 0.2, b1('a_trip') - 0.6), (b0('a_home') + 1.2, b1('a_home')),
               (b0('a_3km') - 0.3, b0('a_3km') + 0.9), (b0('a_nobelieve') + 0.1, b0('a_nobelieve') + 1.0), (b0('a_again') - 0.5, b1('a_93')),
               (b0('b_stoker') - 0.4, b0('b_stoker') + 1.4), (b0('r_two') + 0.6, b0('r_two') + 1.2)):
    bed(f('pen'), t0, t1, -10, 0.15, 0.25)
# the watch, ticking up to 8:15 and stopping at the flash
tf = b0('a_flash') + 0.38
bed(f('watch'), b0('a_815') - 0.75, tf, -4, 0.3, 0.02)
# the flash: a deep muffled swell (no bang), and the second light in Nagasaki, lower
place(f('rumble'), tf, 0, 0.0, 2.0, k=2.0)
place(f('rumble'), b1('a_light') + 0.05, -4, 0.0, 1.6, k=2.0)
place(f('rumble'), b0('a_burn') - 0.4, -8, 0.4, 1.4, k=2.0)
# the train home
bed(f('train'), b0('a_train') - 0.2, b1('a_train') + 0.3, -8, 0.6, 0.8)
# stoker's furnace
bed(f('furnace'), b0('b_stoker') + 0.3, b1('b_name') + 0.2, -8, 0.6, 0.8)
# the iceberg, the collision in the fog, the torpedo
place(f('hull'), b0('b_ice') + 0.72 * (b1('b_ice') - b0('b_ice')), -3, 0.0, 1.0)
place(f('foghorn'), b0('b_fog') + 0.1, -6, 0.2, 1.0)
place(f('hull'), b0('b_hit') + 0.62 * (b1('b_hit') - b0('b_hit')), -5, 0.0, 1.0)
place(f('torpedo'), b1('b_torp') - 0.25, -4, 0.0, 0.8)
# the sea
for t0, t1, g in ((b0('b_ice') - 0.3, b1('b_saved1') + 0.3, -16), (b0('b_fog') - 0.3, b1('b_saved2') + 0.3, -16),
                  (b0('b_18') - 0.4, b1('b_1200') + 0.3, -16), (b0('b_swim') - 0.4, b1('b_swim') + 0.3, -12),
                  (b0('b_oath') - 0.4, b1('b_oath') + 0.4, -10), (b0('r_sea') - 0.5, b1('r_sea') + 0.5, -10)):
    bed(f('waves'), t0, t1, g, 0.5, 0.8)
# the hanko and the stamp of truth
tfo, dfo = b0('a_forgot'), b1('a_forgot') - b0('a_forgot')
STAMP_HIT = 0.64                                               # main thud in stamp.wav (light touch at 0.30 s is skipped)
place(f('stamp'), tfo + 0.86 * dfo, -6, 0.0, 0.2, skip=STAMP_HIT, at_onset=False)
place(f('stamp'), b0('r_true') + 0.55 * (b1('r_true') - b0('r_true')), 0, 0.0, 0.3, skip=STAMP_HIT, at_onset=False)
# the countdown: one tick per second
for k in range(5):
    place(f('watch'), tc + k, -2, 0.0, 0.05, dur=0.25)

for l in LOG: print(l)
np.clip(out, -1, 1).astype(np.float32).tofile(f'{ROOT}/audio/mix_raw.f32')
RAW = ['-f', 'f32le', '-ar', str(SR), '-ac', '2', '-i', f'{ROOT}/audio/mix_raw.f32']
LIM = 'alimiter=limit=0.79:attack=1:release=60:level=disabled'
txt = subprocess.run(['ffmpeg', '-hide_banner', '-nostats'] + RAW + ['-af', f'{LIM},loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json',
                     '-f', 'null', '-'], capture_output=True, text=True).stderr
ms = json.loads(txt[txt.rindex('{'):txt.rindex('}') + 1])
ln = (f"loudnorm=I=-14:TP=-1.5:LRA=11:linear=true:measured_I={ms['input_i']}:measured_TP={ms['input_tp']}"
      f":measured_LRA={ms['input_lra']}:measured_thresh={ms['input_thresh']}:offset={ms['target_offset']}")
subprocess.run(['ffmpeg', '-v', 'error', '-y'] + RAW + ['-af', f'{LIM},{ln},{LIM}', '-ar', str(SR), f'{ROOT}/audio/mix.wav'], check=True)
chk = subprocess.run(['ffmpeg', '-hide_banner', '-nostats', '-i', f'{ROOT}/audio/mix.wav', '-af', 'ebur128=peak=true', '-f', 'null', '-'],
                     capture_output=True, text=True).stderr
print('mix ok', T, ' | ', ' '.join(l.strip() for l in chk.splitlines()[-12:] if 'I:' in l or 'Peak:' in l))
