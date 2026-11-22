"""Episode 4 narration edit.
  new hook (audio/hook_v2.mp3)  +  original take (audio/voice_ep04_v1.mp3) from «Χίλια εννιακόσια τριάντα εννιά»,
  minus three approved sentences, pauses capped at CAP, longer silences at the section breaks.
Writes audio/voice_edit.wav and audio/align_edit.json (whisper words moved to the edited timeline).
Needs audio/align_hook.json and audio/align_v1.json from tools/align.py.
usage: MT_ROOT=/home/user/ep04 python3 edit_voice04.py"""
import json, os, re, subprocess, unicodedata, numpy as np
ROOT = os.environ.get('MT_ROOT', '/home/user/ep04')
A = f'{ROOT}/audio'
SR = 48000
CAP = 0.36                      # longest ordinary pause (s)
# (previous word, word) that ends a section -> total silence after it (s)
BREAKS = {(('επιζωντες', 'ειναι'), 1): 0.9,          # «…είναι αληθινοί.»  (anchor words, +n -> word that ends the section)
          (('οπως', 'ζητησε'), 2): 0.9,              # «…στην Ιερουσαλήμ.»
          (('το', 'κανει'), 1): 0.8,                 # «…το κάνει ταινία.»
          (('δεν', 'εκαναν'), 1): 0.9,               # «…δεν έκαναν τίποτα.»
          (('διακοσιων', 'κατατρεγμενων'), 1): 1.0}  # «…κατατρεγμένων Εβραίων».
# original take: keep these ranges (s). Removed: «Τις πουλάει στον γερμανικό στρατό.»,
# «Δεν τους πληρώνει. Πληρώνει νοίκι στα Ες Ες.», «Ο κόσμος έμαθε την ιστορία του χάρη σε έναν χαρτοφύλακα.»
KEEP = [(14.0, 32.40), (34.55, 39.42), (42.47, 101.55), (105.25, 149.76), (155.51, None)]
HOOK_END = 11.40
# re-recorded sentence (user, 22/11): «Για τον Σπίλμπεργκ, ήταν μια κόκκινη σημαία μπροστά σε έναν κόσμο που δεν ήθελε να δει.»
# replaces v1 149.76-155.51 (cut points = middles of the silences around the old take). +0.8 dB to match v1.
RF = ('redflag_v2.mp3', 'align_redflag.json', 149.76, 0.25, 5.42, 10 ** (0.8 / 20))   # file, align, v1 cut, lead pad, use until, gain


def load(f):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', f, '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-'],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).copy()


def norm(w):
    w = unicodedata.normalize('NFD', w.lower())
    w = ''.join(c for c in w if unicodedata.category(c) != 'Mn')
    return re.sub(r'[^\w]', '', w)


def words(path):
    return [w for s in json.load(open(path)) for w in s['words']]


hook, v1 = load(f'{A}/hook_v2.mp3'), load(f'{A}/voice_ep04_v1.mp3')
hook_gain = 10 ** (0.9 / 20)                       # hook take was 0.9 LU quieter
pieces = [(hook[:int(HOOK_END * SR)] * hook_gain, 0.0, 'hook')]
src_map = [('hook', 0.0, HOOK_END, 0.0)]           # (src, s0, s1, edit offset)
off = HOOK_END
rf = load(f'{A}/{RF[0]}')
for s0, s1 in KEEP:
    s1 = s1 or len(v1) / SR
    pieces.append((v1[int(s0 * SR):int(s1 * SR)], s0, 'v1'))
    src_map.append(('v1', s0, s1, off))
    off += s1 - s0
    if abs(s1 - RF[2]) < 1e-6:                    # the re-recorded sentence goes here
        clip = np.concatenate([np.zeros(int(RF[3] * SR), np.float32), rf[:int(RF[4] * SR)] * RF[5],
                               np.zeros(int(0.1 * SR), np.float32)])
        pieces.append((clip, 0.0, 'rf'))
        src_map.append(('rf', -RF[3], RF[4], off))
        off += len(clip) / SR
x = np.concatenate([p[0] for p in pieces])

# words on the assembled timeline (words inside removed ranges are dropped)
ww = []
for src, path in (('hook', f'{A}/align_hook.json'), ('v1', f'{A}/align_v1.json'), ('rf', f'{A}/{RF[1]}')):
    for w in words(path):
        for s, a, b, o in src_map:
            if s == src and a <= (w['s'] + w['e']) / 2 < b:
                ww.append({'w': w['w'], 's': w['s'] - a + o, 'e': w['e'] - a + o, 'p': w['p']})
ww.sort(key=lambda w: w['s'])

# silences of the assembled take -> new lengths
tmp = f'{A}/_asm.wav'
x.astype(np.float32).tofile(tmp + '.f32')
subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'f32le', '-ar', str(SR), '-ac', '1', '-i', tmp + '.f32', tmp], check=True)
log = subprocess.run(['ffmpeg', '-i', tmp, '-af', 'silencedetect=n=-42dB:d=0.18', '-f', 'null', '-'],
                     capture_output=True, text=True).stderr
sil = list(zip([float(v) for v in re.findall(r'silence_start: ([0-9.]+)', log)],
               [float(v) for v in re.findall(r'silence_end: ([0-9.]+)', log)]))
nw = [norm(w['w']) for w in ww]
want = {}
for (key, plus), L in BREAKS.items():
    n = len(key)
    hits = [i for i in range(len(nw) - n + 1) if tuple(nw[i:i + n]) == key]
    assert len(hits) == 1, f'break {key}: {len(hits)} matches'
    t = ww[hits[0] + n - 1 + plus]['e']
    a, b = min(sil, key=lambda s: abs(s[0] - t))
    assert abs(a - t) < 0.6, f'no silence after {key}'
    want[(a, b)] = L
edits = []                                         # (a, b, new length)
for a, b in sil:
    if b >= len(x) / SR - 0.01: continue           # trailing silence handled below
    L = want.get((a, b), min(b - a, CAP))
    if (a, b) in want or b - a > CAP + 0.01:
        edits.append((a, b, L))
missing = set(want) - {(a, b) for a, b, _ in edits}
assert not missing, missing

out, prev, shift = [], 0, []
for a, b, L in edits:
    m = (a + b) / 2
    d = L - (b - a)                                # <0 shorten, >0 lengthen
    if d < 0:
        # keep L/2 after the last word and L/2 before the next one, drop the middle.
        # (the old 'cut at the middle, skip -d' overshot into the next word when the pause was > 2*L:
        #  that clipped the onset of «Είναι το τέλος της ταινίας»)
        out.append(x[int(prev * SR):int((a + L / 2) * SR)])
        prev = b - L / 2
    else:
        out.append(x[int(prev * SR):int(m * SR)])
        out.append(np.zeros(int(d * SR), np.float32)); prev = m
    shift.append((m, d))
out.append(x[int(prev * SR):])
y = np.concatenate(out)


def mv(t):
    return t + sum(d for m, d in shift if m <= t)


for w in ww:
    w['s'], w['e'] = round(mv(w['s']), 3), round(mv(w['e']), 3)
y.astype(np.float32).tofile(f'{A}/voice_edit.f32')
subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'f32le', '-ar', str(SR), '-ac', '1', '-i', f'{A}/voice_edit.f32',
                f'{A}/voice_edit.wav'], check=True)
json.dump([{'start': ww[0]['s'], 'end': ww[-1]['e'], 'text': '', 'words': ww}], open(f'{A}/align_edit.json', 'w'),
          ensure_ascii=False)
print('voice_edit', round(len(y) / SR, 2), 's  (assembled', round(len(x) / SR, 2), 's)  edits', len(edits),
      ' breaks', sorted((round(a, 2), L) for (a, b), L in want.items()))
