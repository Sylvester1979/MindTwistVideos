"""Episode 5 mix: voice + music (Suno «zakinthos ww2», id 8cbe7e4e, the user's track) ducked, never silent
+ quiet realistic SFX by shot -> audio/mix.wav (-14 LUFS, TP -1.5).
usage: MT_ROOT=<dir with timeline.json, audio/voice_edit.wav, music/zakinthos_ww2.wav, sfx/sfx_pack_ep05.wav> python3 mix05.py
All effects travel as ONE file (sfx_pack_ep05.wav, md5 f24048f4…): 14 ElevenLabs effects back to back, 0.5 s apart; PACK gives
[first sample, length] at 48 kHz. 8 are reused from the user's earlier set (pen … paper), 6 were made for this episode.
SFX files are loudness-matched first (each to the same RMS), so the gains below are relative and readable.
Every effect is placed by its audible event (onset, measured high-passed), never by the start of its file (LESSONS, ep. 4)."""
import json, subprocess, numpy as np, os
ROOT = os.environ.get('MT_ROOT', '/home/user/ep05')
SR = 48000
TL = json.load(open(f'{ROOT}/timeline.json'))
T, VOFF = TL['T'], TL.get('voff', 0.4)
SH = {s['k']: s for s in TL['shots']}
N = int((T + 1) * SR)


def S0(k): return SH[k]['t0']
def S1(k): return SH[k]['t1']
def db(x): return 10 ** (x / 20)


def load(path, ch=2):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-ac', str(ch), '-ar', str(SR), '-f', 'f32le', '-'],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, ch).copy()


def active_rms(a):
    fr = SR // 20; n = len(a) // fr
    r = np.sqrt((a[:n * fr] ** 2).reshape(n, fr, -1).mean((1, 2)) + 1e-12)
    act = np.sort(r)[int(n * 0.4):]
    return float(np.sqrt((act ** 2).mean()))


def env(points, n=N):
    t = np.arange(n) / SR
    ts, gs = zip(*points)
    return np.interp(t, ts, gs).astype(np.float32)


out = np.zeros((N, 2), np.float32)
LOG = []
SFX_TRIM = db(-10)           # the user's taste (ep. 4): effects are background, not events
PACK = {'pen': [0, 96000], 'waves': [120000, 384000], 'rumble': [528000, 96000], 'train': [648000, 96000],
        'typewriter': [768000, 96000], 'watch': [888000, 96000], 'stamp': [1008000, 96000], 'paper': [1128000, 147840],
        'fire': [1299840, 282240], 'embers': [1606080, 186240], 'boats_in': [1816320, 186240], 'boats_out': [2026560, 186240],
        'village': [2236800, 186240], 'footsteps': [2447040, 186240]}
_pack = None


def sfx(name, target_db=-20.0):
    """one effect out of the pack, its active part brought to target RMS (so gains below are relative)."""
    global _pack
    if _pack is None: _pack = load(f'{ROOT}/sfx/sfx_pack_ep05.wav')
    i, n = PACK[name]; a = _pack[i:i + n].copy()
    return a * (db(target_db) / max(active_rms(a), 1e-6))


def place(x, t, gain=1.0, fin=0.0, fout=0.0, dur=None, name='', skip=0.0):
    if isinstance(x, str): name = name or x; x = sfx(x)
    a = x[int(skip * SR):].copy()
    if dur is not None: a = a[:int(dur * SR)]
    a = a * gain * SFX_TRIM
    if fin > 0: n = min(len(a), int(fin * SR)); a[:n] *= np.linspace(0, 1, n)[:, None]
    if fout > 0: n = min(len(a), int(fout * SR)); a[-n:] *= np.linspace(1, 0, n)[:, None]
    i0 = int(t * SR)
    if i0 < 0: a = a[-i0:]; i0 = 0
    i1 = min(N, i0 + len(a)); out[i0:i1] += a[:i1 - i0]
    LOG.append(f'{name[:24]:24s} {t:7.2f}s  {len(a) / SR:5.1f}s  {20 * np.log10(gain + 1e-9):6.1f} dB')


def bed(name, t0, t1, gain, fin=0.8, fout=1.0, skip=0.0):
    """sfx bed from t0 to t1 (looped with a crossfade if the file is shorter)."""
    a = sfx(name)[int(skip * SR):]
    n = int((t1 - t0) * SR)
    if n <= 0: return
    b = a.copy()
    while len(b) < n:
        xf = int(min(0.4 * SR, len(a) // 3)); w = np.linspace(0, 1, xf)[:, None]
        b = np.concatenate([b[:-xf], b[-xf:] * (1 - w) + a[:xf] * w, a[xf:]])
    place(b[:n], t0, gain, fin, fout, name=name)


# ---------------- voice
voice = load(f'{ROOT}/audio/voice_edit.wav', 1)[:, 0]
vo = np.zeros(N, np.float32)
i0 = int(VOFF * SR); n = min(N - i0, len(voice)); vo[i0:i0 + n] = voice[:n]
out += vo[:, None]
v_rms = active_rms(voice[:, None])

# ---------------- music: «zakinthos ww2» (200 s) on a 179 s video. It enters at 14.9 s of the track (its first full phrase),
# so its warm lift (166.1 s) lands on «Τους έσωσε ένα ολόκληρο νησί» and its last chord (192.35 s) right after the last word.
MSRC = 14.9
mus = load(f'{ROOT}/music/zakinthos_ww2.wav')[int(MSRC * SR):]
m = np.zeros((N, 2), np.float32); m[:min(N, len(mus))] = mus[:N]
MUS_REL = -13.0              # music (unducked, active RMS) this many dB under the voice
g = db(MUS_REL) * v_rms / active_rms(mus)
# ducking under the voice (about -6 dB while he speaks; never silent)
va = np.convolve(np.abs(vo), np.ones(SR // 20) / (SR // 20), 'same')
act = (va > 0.01).astype(np.float32)
k = int(0.35 * SR); act = np.convolve(act, np.ones(k) / k, 'same')
duck = 1 - 0.5 * np.clip(act * 1.6, 0, 1)
auto = env([(0, 0.0), (0.25, 1.0), (T - 1.6, 1.0), (T, 0.0), (N / SR, 0.0)])     # short fade in, fade out on the last 1.6 s
m *= (duck * auto)[:, None] * g
out += m
print(f'music: from {MSRC}s of the track, gain {20 * np.log10(g):.1f} dB (voice active RMS {20 * np.log10(v_rms):.1f} dBFS);'
      f' lift at {166.1 - MSRC:.2f}s, last chord at {192.35 - MSRC:.2f}s of the video')

# ---------------- SFX (quiet, realistic). Gains are relative to the loudness-matched effect (-20 dB RMS).
# onset = the audible event inside each effect (high-passed, 12 dB under its peak), measured:
ON = {'pen': 0.25, 'train': 0.15, 'type': 0.25, 'stamp': 0.65, 'paper': 0.35, 'fire': 0.15}
# no pen on «Πήραν μια λίστα με δύο ονόματα»: the user (26/11) heard it as a disturbance in the narrator's voice
bed('waves', S0('G02_map'), S1('G02_map') + 0.3, db(-14), 0.8, 1.2)                     # the island on the map
bed('rumble', S0('S04') - 0.1, S1('S04') + 0.6, db(-9), 0.1, 0.8)                       # «Έρχονται οι Γερμανοί»
place('train', 23.46 - ON['train'], db(-14), 0.2, 0.8)                                  # «ήδη σταλεί στο Άουσβιτς»
bed('typewriter', 34.70 - ON['type'], S1('G09_page') - 0.05, db(-14), 0.02, 0.4)       # the heads typed on their words
bed('watch', S0('G11_clock') + 0.1, S1('G11_clock') - 0.05, db(-12), 0.3, 0.5)          # the clock to the next day
place('fire', S0('B3D') + 0.05 - ON['fire'], db(-10), 0.0, 0.6, dur=S1('B3D') - S0('B3D') + 0.3)  # «να κάψει τη λίστα»: lit, then burning
bed('footsteps', S0('S16') + 0.1, S1('S16'), db(-15), 0.3, 0.5)                         # the Metropolitan walks to the commander
for t in (70.04, 70.85):                                                                # «ξανά και ξανά»: a stamp on each sheet
    place('stamp', t - ON['stamp'], db(-14), 0.0, 0.3, dur=1.4)
# the paper slides across the desk in C between its frames ~95 and ~155 (video 74.7-76.7 s); the effect's sliding part
# (0.35-2.4 s into the file) covers that move. It was at 76.25 s, after the move (the user, 26/11)
place('paper', 74.70 - ON['paper'], db(-11), 0.05, 0.5)                                 # «παραδίδουν στον διοικητή ένα χαρτί»
bed('village', S0('S23'), S1('S23') + 0.3, db(-15), 0.5, 0.8)                           # hidden in the villages, at night
# no train whistle on «στέλνονται στο Άουσβιτς» (Κέρκυρα): the user (26/11) found it wrong for this video
bed('boats_in', S0('S27'), S1('S27'), db(-12), 2.0, 0.3)                                # «γερμανικά σκάφη έρχονται»: they come closer
bed('boats_out', S0('S28'), S1('S28') + 1.0, db(-12), 0.2, 2.2)                         # «Φεύγουν χωρίς κανέναν»: they fade away
bed('waves', S0('S28'), S1('S28') + 1.2, db(-17), 0.6, 1.5, skip=2.0)
bed('embers', S0('B3D_ash'), S1('B3D_ash'), db(-16), 0.4, 0.8)                          # the ash of the list
bed('typewriter', 167.26 - ON['type'], 169.48, db(-15), 0.02, 0.5)                     # the two names typed

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
