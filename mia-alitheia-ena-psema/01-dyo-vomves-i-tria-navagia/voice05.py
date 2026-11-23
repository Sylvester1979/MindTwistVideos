"""Ep05 voice: splice the re-recorded last phrase, find the beats in the narration, insert the edit pauses,
write voice_edit.wav and the real timeline.json.

No ASR in this sandbox (no whisper, no network), so beats are aligned to the *silences*:
speech islands from silencedetect are grouped into the 62 beats by dynamic programming, matching each
group's duration to the beat's expected length (letters of the text). Word times inside a beat are then
spread by letters (used only for subtitle chunks).
  voice05.py            -> $MT_ROOT/audio/voice_edit.wav, $MT_ROOT/timeline.json, prints the table"""
import os, sys, re, json, subprocess, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
from timeline05 import BEATS, ROOT

SR = 48000
V1 = f'{ROOT}/audio/voice_ep05_v1.mp3'
CTA = f'{ROOT}/audio/cta_v2.mp3'
CUT_CTA = 161.39            # middle of the silence before «Εσύ το βρήκες;» in v1
CTA_GAIN_DB = 0.4           # v1 -20.0 LUFS, cta -20.4 LUFS
VOFF = 0.40                 # narration starts here in the video
TAIL = 2.2                  # hold after the last word
# extra silence inserted after a beat (seconds): bridges, the countdown, the long pauses
INSERT = {'a_flash': 0.6, 'a_93': 2.6, 'b_oath': 1.2, 'q_true': 5.0, 'r_sea': 0.65}
# hand corrections where the silence grouping is ambiguous (checked on the island list and the energy curve):
#   'Η άλλη... όχι' has its own pause; a_home runs over two islands; a_road/a_forgot share one island,
#   split at the -41 dB dip at 25.27 s; 'Δύο ναυάγια, όχι τρία' has a comma pause.
OVERRIDE = {'hook4': (6.17, 7.35), 'hook5': (7.91, 9.02), 'a_home': (19.70, 23.46),
            'a_road': (23.97, 25.24), 'a_forgot': (25.30, 27.69),
            'r_two': (156.44, 158.28), 'r_sea': (158.88, 161.17)}

def load(path):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-f', 'f32le', '-ac', '1', '-ar', str(SR), '-'],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).copy()

def silences(x, thr_db=-40, dmin=0.12):
    """[(start, end)] of silences (RMS over 10 ms)."""
    hop = SR // 100
    n = len(x) // hop
    rms = np.sqrt(np.mean(x[:n * hop].reshape(n, hop) ** 2, 1) + 1e-12)
    db = 20 * np.log10(rms)
    q = db < thr_db
    out, i = [], 0
    while i < n:
        if q[i]:
            j = i
            while j < n and q[j]: j += 1
            if (j - i) / 100 >= dmin or i == 0 or j == n:
                out.append((i / 100, j / 100))
            i = j
        else:
            i += 1
    return out

def islands(sil, dur):
    sp, t = [], 0.0
    for a, b in sil:
        if a > t + 0.02: sp.append((t, a))
        t = b
    if t < dur - 0.02: sp.append((t, dur))
    return sp

def letters(txt):
    return len(re.sub(r'[^\wά-ώΆ-Ώ]', '', re.sub(r'\[[^\]]*\]', '', txt)))

def align(isl, beats):
    """group consecutive speech islands into beats. returns list of (i0, i1) island index ranges per beat."""
    n, m = len(isl), len(beats)
    L = np.array([letters(t) for _, t in beats], np.float64)
    speech = sum(b - a for a, b in isl)
    rate = speech / L.sum()                                # seconds per letter (speech only)
    exp = L * rate
    INF = 1e18
    D = np.full((m + 1, n + 1), INF); D[0, 0] = 0
    back = np.zeros((m + 1, n + 1), np.int32)
    for i in range(1, m + 1):
        for j in range(i, n - (m - i) + 1):
            best, arg = INF, -1
            for k in range(max(i - 1, j - 12), j):          # beat i = islands k..j-1 (at most 12 islands)
                if D[i - 1, k] >= INF: continue
                s0, s1 = isl[k][0], isl[j - 1][1]
                inner = sum(isl[q + 1][0] - isl[q][1] for q in range(k, j - 1))
                d = (s1 - s0) - inner * 0.6                 # pauses inside a beat count a little
                c = math.log(max(d, 0.05) / exp[i - 1]) ** 2
                # prefer cutting beats at longer silences
                if j < n:
                    gap = isl[j][0] - isl[j - 1][1]
                    c -= 0.15 * min(gap, 0.6)
                if D[i - 1, k] + c < best: best, arg = D[i - 1, k] + c, k
            D[i, j], back[i, j] = best, arg
    res, j = [], n
    for i in range(m, 0, -1):
        k = back[i, j]; res.append((k, j - 1)); j = k
    return res[::-1]

def main():
    v1 = load(V1); cta = load(CTA) * 10 ** (CTA_GAIN_DB / 20)
    cut = int(CUT_CTA * SR)
    # splice: 10 ms fades on both sides of the join so nothing clicks; the pause before the new
    # phrase is lengthened in INSERT['r_sea'] (the original gap was only 0.3 s)
    head = v1[:cut].copy(); f = int(0.010 * SR)
    head[-f:] *= np.linspace(1, 0, f, dtype=np.float32)
    cta = cta.copy(); cta[:f] *= np.linspace(0, 1, f, dtype=np.float32)
    voice = np.concatenate([head, np.zeros(int(0.08 * SR), np.float32), cta])
    dur = len(voice) / SR
    isl = islands(silences(voice), dur)
    groups = align(isl, BEATS)
    beats = {}
    for (key, txt), (k, j) in zip(BEATS, groups):
        beats[key] = list(OVERRIDE.get(key, (isl[k][0], isl[j][1])))
    # insert the edit pauses (in the middle of the silence after the beat)
    pieces, t_prev, shifts = [], 0.0, []
    keys = [k for k, _ in BEATS]
    for key, extra in sorted(INSERT.items(), key=lambda kv: beats[kv[0]][1]):
        i = keys.index(key)
        t_end = beats[key][1]
        t_next = beats[keys[i + 1]][0] if i + 1 < len(keys) else dur
        at = 0.5 * (t_end + t_next)
        shifts.append((at, extra))
    out, last = [np.zeros(int(VOFF * SR), np.float32)], 0
    for at, extra in sorted(shifts):
        s = int(at * SR)
        out += [voice[last:s], np.zeros(int(extra * SR), np.float32)]; last = s
    out += [voice[last:], np.zeros(int(TAIL * SR), np.float32)]
    edit = np.concatenate(out)
    def remap(t):
        return VOFF + t + sum(e for a, e in shifts if a <= t)
    tl = {k: [round(remap(a), 3), round(remap(b), 3)] for k, (a, b) in beats.items()}
    tl['_end'] = [round(len(edit) / SR, 3)] * 2
    json.dump(tl, open(f'{ROOT}/timeline.json', 'w'), ensure_ascii=False, indent=1)
    p = subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'f32le', '-ar', str(SR), '-ac', '1', '-i', '-',
                        '-c:a', 'pcm_s16le', f'{ROOT}/audio/voice_edit.wav'], input=edit.tobytes())
    rate_chk = []
    for (key, txt), (k, j) in zip(BEATS, groups):
        a, b = beats[key]
        print(f'{key:12s} {remap(a):7.2f} {remap(b):7.2f}  {b - a:5.2f}s  isl {k:3d}-{j:3d}  {letters(txt) / max(0.1, b - a):5.1f} let/s  {txt[:50]}')
        rate_chk.append(letters(txt) / max(0.1, b - a))
    print(f'islands {len(isl)}, voice {dur:.2f}s, edit {len(edit) / SR:.2f}s, letters/s median {np.median(rate_chk):.1f} min {min(rate_chk):.1f} max {max(rate_chk):.1f}')

if __name__ == '__main__':
    main()
