"""Ep05 (Zakynthos) voice edit: splice the two re-recorded phrases into the full take (script v7),
cut «Σε λίγες εβδομάδες, φεύγουν κι οι Γερμανοί» (planned cut if > 3′), tighten pauses (keep L/2 each side),
loudness-match the inserts to the phrases they replace. Times come from align05z.py (silences, no ASR).
-> audio/voice_edit.wav, audio/splice_*.mp3 previews, prints the report."""
import subprocess, json, os
import numpy as np
SR = 48000
A = '/home/claude/ep05z/audio'
FADE = int(0.010 * SR)

def load(p):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', p, '-f', 'f32le', '-ac', '1', '-ar', str(SR), '-'],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).copy()

def save(p, x):
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'f32le', '-ar', str(SR), '-ac', '1', '-i', '-', p],
                   input=x.astype(np.float32).tobytes(), check=True)

def lufs(x):
    save('/tmp/_l.wav', x)
    out = subprocess.run(['ffmpeg', '-hide_banner', '-i', '/tmp/_l.wav', '-af', 'ebur128', '-f', 'null', '-'],
                         capture_output=True, text=True).stderr
    return float([l for l in out.splitlines() if l.strip().startswith('I:')][-1].split()[1])

def seg(x, t0, t1):
    s = x[int(t0 * SR):int(t1 * SR)].copy()
    n = min(FADE, len(s) // 2)
    s[:n] *= np.linspace(0, 1, n); s[-n:] *= np.linspace(1, 0, n)
    return s

full, fa, fb = load(f'{A}/voice_full.wav'), load(f'{A}/fix_A.wav'), load(f'{A}/fix_B.wav')

# speech spans (silencedetect / align05z): old phrase in the full take -> new take
OLD_B = (73.66, 76.14)   # «Και τότε, οι δυο τους του δίνουν ένα χαρτί.»
NEW_B = (0.06, 5.47)     # «Και τότε, ο δήμαρχος και ο Μητροπολίτης παραδίδουν στον διοικητή ένα χαρτί.»
OLD_A = (99.27, 101.70)  # «Οι αντάρτες τους προειδοποιούν για τις εφόδους.»
NEW_A = (0.03, 2.92)     # «Οι αντάρτες, τους προειδοποιούν για τις εφόδους.»
CUT = (118.57, 121.86)   # short pause + «Σε λίγες εβδομάδες, φεύγουν κι οι Γερμανοί.» (the long pause after it stays)

gb = lufs(seg(full, *OLD_B)) - lufs(seg(fb, *NEW_B))
ga = lufs(seg(full, *OLD_A)) - lufs(seg(fa, *NEW_A))
ins_b = seg(fb, *NEW_B) * 10 ** (gb / 20)
ins_a = seg(fa, *NEW_A) * 10 ** (ga / 20)

parts = [seg(full, 0, OLD_B[0]), ins_b, seg(full, OLD_B[1], OLD_A[0]), ins_a,
         seg(full, OLD_A[1], CUT[0]), seg(full, CUT[1], len(full) / SR)]
marks = []
t = 0.0
for i, p in enumerate(parts):
    if i in (1, 3, 5): marks.append(t)
    t += len(p) / SR
x = np.concatenate(parts)

# pause tightening: every silence > cap shrinks to cap, removed from its middle
CAP_S, CAP_L = 0.55, 0.85     # ordinary pause / long pause (> 0.85 s in the take)
hop = SR // 100
n = len(x) // hop
db = 20 * np.log10(np.sqrt(np.mean(x[:n * hop].reshape(n, hop) ** 2, 1) + 1e-12))
q = db < -40
sil, i = [], 0
while i < n:
    if q[i]:
        j = i
        while j < n and q[j]: j += 1
        if (j - i) >= 25 and i > 0 and j < n: sil.append((i / 100, j / 100))
        i = j
    else: i += 1
keep, t_prev, removed = [], 0.0, 0.0
shift = []                                  # (time in x, cumulative removed) for remapping the splice marks
for a, b in sil:
    L = b - a; cap = CAP_L if L > 0.85 else CAP_S
    if L <= cap: continue
    cut0, cut1 = a + cap / 2, b - cap / 2
    keep.append(seg(x, t_prev, cut0)); t_prev = cut1
    removed += cut1 - cut0; shift.append((cut1, removed))
keep.append(seg(x, t_prev, len(x) / SR))
y = np.concatenate(keep)
remap = lambda tm: tm - max([r for c, r in shift if c <= tm], default=0.0)
save(f'{A}/voice_edit.wav', y)

marks = [remap(m) for m in marks]
for k, (name, m) in enumerate(zip(['B (χαρτί)', 'A (αντάρτες)', 'κόψιμο «Σε λίγες εβδομάδες»'], marks)):
    clip = y[int(max(0, m - 4.5) * SR):int((m + 6.5) * SR)]
    save(f'{A}/splice_{k + 1}.mp3', clip)
rep = dict(full=len(full) / SR, edited=len(y) / SR, gain_B_dB=round(gb, 2), gain_A_dB=round(ga, 2),
           pauses_removed=round(removed, 2), splice_at=[round(m, 2) for m in marks])
json.dump(rep, open(f'{A}/edit_report.json', 'w'), indent=1)
print(json.dumps(rep, indent=1))
