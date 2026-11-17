"""Final mix: voice + music (ducked, with the 11:00 dip) + SFX placed from the timeline -> audio/mix.wav (-14 LUFS)."""
import json, subprocess, numpy as np, os, sys
ROOT = os.environ.get('MT_ROOT', '/home/user/ww1')
SR = 48000
TL = json.load(open(f'{ROOT}/timeline.json'))
T = TL['T']; PC = {p['id']: p for p in TL['pieces']}
N = int(T * SR) + SR

def P0(k): return PC[k]['t0']
def P1(k): return PC[k]['t1']
def cue(pid, *subs, frac=0.5):
    p = PC[pid]
    for w in p['words']:
        if any(s.lower() in w['w'].lower() for s in subs): return w['s']
    return p['t0'] + frac * (p['t1'] - p['t0'])

def load(path, ch=2):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-ac', str(ch), '-ar', str(SR), '-f', 'f32le', '-'],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, ch).copy()

def db(x): return 10 ** (x / 20)

def env(points, n=N):
    """piecewise-linear gain envelope from [(t, gain), ...]"""
    t = np.arange(n) / SR
    ts, gs = zip(*points)
    return np.interp(t, ts, gs).astype(np.float32)

out = np.zeros((N, 2), np.float32)

def place(x, t, gain=1.0, fin=0.0, fout=0.0, dur=None, pan=0.0):
    if isinstance(x, str) and not os.path.exists(x): return
    a = load(x) if isinstance(x, str) else x
    if dur is not None: a = a[:int(dur * SR)]
    a = a * gain
    if fin > 0: n = min(len(a), int(fin * SR)); a[:n] *= np.linspace(0, 1, n)[:, None]
    if fout > 0: n = min(len(a), int(fout * SR)); a[-n:] *= np.linspace(1, 0, n)[:, None]
    if pan: a = a * np.array([1 - max(0, pan), 1 + min(0, pan)], np.float32)
    i0 = int(t * SR)
    if i0 < 0: a = a[-i0:]; i0 = 0
    i1 = min(N, i0 + len(a)); out[i0:i1] += a[:i1 - i0]

def loop(path, t0, t1, gain, fin=0.6, fout=0.8):
    if not os.path.exists(path): return
    a = load(path); n = int((t1 - t0) * SR)
    if n <= 0: return
    reps = int(np.ceil(n / len(a))) + 1
    b = np.concatenate([a] * reps)[:n].copy()
    place(b, t0, gain, fin, fout)

S = f'{ROOT}/sfx'
# ---------------- voice
voice = load(f'{ROOT}/audio/voice_edit.wav', 1)[:, 0]
vo = np.zeros(N, np.float32); vo[:min(N, len(voice))] = voice[:N]
out += vo[:, None] * db(0)

# ---------------- music: end aligned to T, ducked under voice, dip at 11:00
mus = load(f'{ROOT}/audio/music.mp3')
off = T + 0.2 - len(mus) / SR          # music start time (negative -> trimmed head)
m = np.zeros((N, 2), np.float32)
i0 = int(off * SR)
src = mus[max(0, -i0):]; d0 = max(0, i0)
m[d0:d0 + min(len(src), N - d0)] = src[:N - d0]
# voice-activity ducking (smooth)
va = np.convolve(np.abs(vo), np.ones(SR // 20) / (SR // 20), 'same')
act = (va > 0.01).astype(np.float32)
k = int(0.35 * SR); act = np.convolve(act, np.ones(k) / k, 'same')
duck = 1 - 0.45 * np.clip(act * 1.6, 0, 1)
tE = P0('eleven'); tG = P0('grave1')
auto = env([(0, 0.0), (0.05, 0.0), (1.2, 1.0),
            (tE - 0.25, 1.0), (tE + 0.05, db(-15)), (tE + 2.3, db(-15)), (tG - 0.4, 0.75), (tG + 0.8, 1.0),
            (T - 4.5, 1.0), (T - 3.8, 1.25), (T - 1.0, 1.25), (T, 0.0), (N / SR, 0.0)])
m *= (duck * auto)[:, None] * db(-17)
out += m

# ---------------- SFX
def f(name):
    w = f'{S}/{name}.wav'
    return w if os.path.exists(w) else f'{S}/{name}.mp3'
for nm, a, b in (('shot1', 0.60, 4.60), ('shot2', 5.00, 7.90)):
    if not os.path.exists(f'{S}/{nm}.wav') and os.path.exists(f'{S}/shot.mp3'):
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', str(a), '-to', str(b), '-i', f'{S}/shot.mp3', '-af', 'afade=t=out:st=%.2f:d=0.6' % (b - a - 0.6), '-ar', str(SR), f'{S}/{nm}.wav'], check=True)
# ticking: opening watch, 10:59 watch, seconds macro, up to 11:00 (then stops)
loop(f('tick'), 0.0, P0('carriage') + 0.2, db(-16), 0.05, 0.5)
loop(f('tick'), P0('at11'), P0('generals'), db(-20), 0.3, 0.4)
loop(f('tick'), cue('gun1', 'αμερικ') - 0.1, P0('gun2') + 0.2, db(-17), 0.2, 0.4)
loop(f('tick'), cue('gun4', 'πόλεμ') - 0.1, P0('gun5'), db(-11), 0.05, 0.1)
loop(f('tick'), P1('gun7') + 0.3, P0('eleven') + 0.03, db(-11), 0.4, 0.0)
# trench: artillery thumps
for k_, tt in enumerate((P0('attacks') + 0.5, P0('attacks') + 1.4, P0('attacks') + 2.1)):
    place(f('artillery'), tt, db(-9 - 2 * k_), 0.0, 0.2)
loop(f('rumble'), P0('attacks') - 0.2, P0('treb1'), db(-14), 0.4, 1.0)
loop(f('rain'), P0('attacks'), P0('lastday'), db(-16), 0.3, 0.6)
# Trébuchon: rain by the river, pen, the shot
loop(f('rain'), cue('treb1', 'ογκιστ', 'ωγκυ') - 0.2, P0('treb2'), db(-17), 0.5, 0.4)
loop(f('pen'), P0('treb2') + 0.1, P1('treb2') - 0.1, db(-14), 0.1, 0.2)
place(f('shot1'), P0('treb3') + 0.22 - 0.17, db(-11))
loop(f('rain'), P0('treb3') + 0.9, P0('price1'), db(-15), 0.4, 0.6)
# Price: sniper shot
place(f('shot1'), cue('price2', 'πυροβολ') - 0.02 - 0.17, db(-8))
# Gunther: letter paper, rain, MG burst
place(f('paper'), P0('gun2') + 0.1, db(-12))
loop(f('rain'), P0('gun3'), P0('gun5'), db(-18), 0.4, 0.4)
place(f('shot2'), cue('gun6', 'πυροβ') - 0.02 - 0.2, db(-8))
# silence -> bugle after 11:00
place(f('trumpet_user'), cue('eleven', 'όπλα', 'οπλα') + 0.45, db(1), 0.05, 0.8)
loop(f('wind'), cue('eleven', 'όπλα', 'οπλα'), P0('mons1'), db(-22), 1.0, 1.5)
loop(f('wind'), P0('mons1'), T, db(-24), 1.5, 2.0)

np.clip(out, -1, 1).astype(np.float32).tofile(f'{ROOT}/audio/mix_raw.f32')
subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'f32le', '-ar', str(SR), '-ac', '2', '-i', f'{ROOT}/audio/mix_raw.f32',
                '-af', 'loudnorm=I=-14:TP=-1.5:LRA=11,alimiter=limit=0.84:attack=1:release=60:level=disabled', '-ar', str(SR), f'{ROOT}/audio/mix.wav'], check=True)
print('mix ok', T)
