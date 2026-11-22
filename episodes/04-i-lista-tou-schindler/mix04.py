"""Episode 4 mix: voice + music (one continuous take, ducked, never silent) + quiet realistic SFX by shot
-> audio/mix.wav (-14 LUFS, TP -1.5).  usage: MT_ROOT=/home/user/ep04 python3 mix04.py
SFX files are loudness-matched first (each to the same RMS), so the gains below are relative and readable.
A missing SFX file is simply skipped (the user's four extra effects can arrive later)."""
import json, subprocess, numpy as np, os
ROOT = os.environ.get('MT_ROOT', '/home/user/ep04')
SR = 48000
TL = json.load(open(f'{ROOT}/timeline.json'))
T, VOFF = TL['T'], TL.get('voff', 0.5)
SH = {s['k']: s for s in TL['shots']}
N = int((T + 1) * SR)


def S0(k): return SH[k]['t0']
def S1(k): return SH[k]['t1']
def db(x): return 10 ** (x / 20)


def load(path, ch=2):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-ac', str(ch), '-ar', str(SR), '-f', 'f32le', '-'],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, ch).copy()


def load_norm(path, target_db=-20.0):
    """load and bring the 'active' part (loudest 60% of 50 ms frames) to target RMS."""
    a = load(path)
    fr = SR // 20
    n = len(a) // fr
    if n == 0: return a
    r = np.sqrt((a[:n * fr] ** 2).mean(1).reshape(n, fr).mean(1) + 1e-12)
    act = np.sort(r)[int(n * 0.4):]
    rms = float(np.sqrt((act ** 2).mean()))
    return a * (db(target_db) / max(rms, 1e-6))


def env(points, n=N):
    t = np.arange(n) / SR
    ts, gs = zip(*points)
    return np.interp(t, ts, gs).astype(np.float32)


out = np.zeros((N, 2), np.float32)
LOG = []
SFX_TRIM = db(-10)           # user, 22/11: every effect -4, then -3, then -3 dB (music and voice unchanged)


def place(x, t, gain=1.0, fin=0.0, fout=0.0, dur=None, name=''):
    if isinstance(x, str):
        if not os.path.exists(x): LOG.append(f'skip {os.path.basename(x)}'); return
        name = name or os.path.basename(x); x = load_norm(x)
    a = x.copy()
    if dur is not None: a = a[:int(dur * SR)]
    a = a * gain * SFX_TRIM
    if fin > 0: n = min(len(a), int(fin * SR)); a[:n] *= np.linspace(0, 1, n)[:, None]
    if fout > 0: n = min(len(a), int(fout * SR)); a[-n:] *= np.linspace(1, 0, n)[:, None]
    i0 = int(t * SR)
    if i0 < 0: a = a[-i0:]; i0 = 0
    i1 = min(N, i0 + len(a)); out[i0:i1] += a[:i1 - i0]
    LOG.append(f'{name:18s} {t:7.2f}s  {len(a) / SR:5.1f}s  {20 * np.log10(gain + 1e-9):6.1f} dB')


def bed(path, t0, t1, gain, fin=0.8, fout=1.0, skip=0.0):
    """sfx bed from t0 to t1 (looped with a crossfade if the file is shorter)."""
    if not os.path.exists(path): LOG.append(f'skip {os.path.basename(path)}'); return
    a = load_norm(path)[int(skip * SR):]
    n = int((t1 - t0) * SR)
    if n <= 0: return
    b = a.copy()
    while len(b) < n:
        xf = int(min(1.0 * SR, len(a) // 3)); w = np.linspace(0, 1, xf)[:, None]
        b = np.concatenate([b[:-xf], b[-xf:] * (1 - w) + a[:xf] * w, a[xf:]])
    place(b[:n], t0, gain, fin, fout, name=os.path.basename(path))


# ---------------- voice
voice = load(f'{ROOT}/audio/voice_edit.wav', 1)[:, 0]
vo = np.zeros(N, np.float32)
i0 = int(VOFF * SR); n = min(N - i0, len(voice)); vo[i0:i0 + n] = voice[:n]
out += vo[:, None]

# ---------------- music: one continuous take; its own decay ends with the video
mus = load(f'{ROOT}/music/music_ep04.wav')
e = np.sqrt(np.convolve(mus[:, 0] ** 2, np.ones(SR // 10) / (SR // 10), 'same'))
decay_end = (np.where(e > db(-45))[0][-1]) / SR
src0 = max(0.0, decay_end - (T + 0.3))
print('music: track', round(len(mus) / SR, 1), 's, decay end', round(decay_end, 1), 's, start at', round(src0, 2), 's')
m = np.zeros((N, 2), np.float32)
seg = mus[int(src0 * SR):]; m[:min(N, len(seg))] = seg[:N]
# ducking under the voice (about -6 dB while she speaks; never silent)
va = np.convolve(np.abs(vo), np.ones(SR // 20) / (SR // 20), 'same')
act = (va > 0.01).astype(np.float32)
k = int(0.35 * SR); act = np.convolve(act, np.ones(k) / k, 'same')
duck = 1 - 0.5 * np.clip(act * 1.6, 0, 1)
t_last = max(s['t1'] for s in TL['subs'])            # «Όλοι τους... χάρη σε μια λίστα.»
auto = env([(0, 0.0), (0.05, 0.25), (2.0, 1.0),
            (S0('Q30_goeth') - 0.3, 1.0), (S0('Q30_goeth') + 0.4, 1.12), (S1('Q30_goeth'), 1.0),   # the quote card
            (t_last, 1.0), (t_last + 0.8, 1.35), (T - 0.2, 1.35), (T + 0.5, 0.0), (N / SR, 0.0)])
m *= (duck * auto)[:, None] * db(-15)
out += m

# ---------------- SFX (quiet, realistic). Gains are relative to the loudness-matched file (-20 dB RMS).
S = f'{ROOT}/sfx'
f = lambda name: f'{S}/{name}.mp3'
bed(f('cemetery_b'), 0.0, S1('S03_twohands'), db(-12 - 6), 0.6, 1.5)   # opening: user wants it 6 dB under the rest                    # Mount Zion, film's final scene
place(f('stones_a'), 0.35, db(-6 - 6), 0.02, 0.8)                       # opening: -6 dB more                                          # stones on the grave
place(f('hammer'), S0('S06_shop') + 0.2, db(-8), 0.02, 0.5, dur=3.4)                 # notice nailed on the shop
bed(f('factory_a'), S0('S07_factory') - 0.2, S1('S08_pots') + 0.3, db(-14), 0.6, 1.0)  # enamelware factory
bed(f('crowd_b'), S0('S09_column'), S1('S09_column'), db(-15), 0.8, 1.0)            # the column of workers
# safe: the loud creak is 2-4.5 s into the file; keep it INSIDE the shot (it ran into «Πλάσοφ, 1944»)
bed(f('safe'), S0('S12_safe') + 0.05, S1('S12_safe') - 0.1, db(-12), 0.05, 0.5, skip=1.7)  # emptying his fortune
bed(f('train_a'), S0('S14_wagons') - 0.2, S1('S14_wagons') + 0.3, db(-13), 0.8, 1.2)  # deportations
bed(f('factory_a'), S0('S16_shells'), S1('S16_shells'), db(-17), 0.8, 1.0, skip=12.0)  # Brünnlitz shell plant
bed(f('typewriter_a'), S0('G17_list') + 0.1, S1('S19_pemper'), db(-12), 0.2, 1.0)   # the list is typed
place(f('clapper'), S0('S18_set_typewriter') + 0.08, db(-9), 0.0, 0.1)                # «Στην ταινία»
bed(f('train_b'), S0('S21_train'), S1('S21_train') + 0.2, db(-14), 0.8, 1.2)        # the train that saved them
bed(f('cemetery_a'), S0('S23_zion'), S1('S23_zion'), db(-13), 0.8, 1.0)             # buried on Mount Zion, 1974
# the bell itself rings 2.2 s into the file: start the file at 2.0 s so it rings as the writer walks in
bed(f('shopbell'), S0('S25_shop') - 0.05, S1('S25_shop'), db(-10), 0.05, 0.8, skip=2.0)   # Pfefferberg's shop
place(f('projector'), S0('S27_book_reel') + 0.1, db(-13), 0.4, 1.0, dur=S1('S27_book_reel') - S0('S27_book_reel'))
bed(f('crowd_a'), S0('S32_redcoat'), S1('S33_redcoat'), db(-15), 1.5, 2.0)          # the red coat in the crowd
bed(f('cemetery_a'), S0('S34_zion1993'), S1('S35_slab'), db(-13), 1.2, 1.5, skip=8.0)  # 1993, the survivors at the grave
place(f('stones_b'), S0('S35_slab') + 0.6, db(-9), 0.02, 0.8)

for l in LOG: print(l)
np.clip(out, -1, 1).astype(np.float32).tofile(f'{ROOT}/audio/mix_raw.f32')
# two-pass loudnorm (linear, so the mix balance is kept), then a true-peak safe limiter
RAW = ['-f', 'f32le', '-ar', str(SR), '-ac', '2', '-i', f'{ROOT}/audio/mix_raw.f32']
LIM = 'alimiter=limit=0.79:attack=1:release=60:level=disabled'
txt = subprocess.run(['ffmpeg', '-hide_banner', '-nostats'] + RAW + ['-af', f'{LIM},loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json',
                     '-f', 'null', '-'], capture_output=True, text=True).stderr
ms = json.loads(txt[txt.rindex('{'):txt.rindex('}') + 1])
ln = (f"loudnorm=I=-14:TP=-1.5:LRA=11:linear=true:measured_I={ms['input_i']}:measured_TP={ms['input_tp']}"
      f":measured_LRA={ms['input_lra']}:measured_thresh={ms['input_thresh']}:offset={ms['target_offset']}")
subprocess.run(['ffmpeg', '-v', 'error', '-y'] + RAW + ['-af', f'{LIM},{ln},{LIM}', '-ar', str(SR), f'{ROOT}/audio/mix.wav'], check=True)
print('mix ok', T)
