"""Real timeline from the recorded narration (replaces timeline_est.py output).
Aligns the script words (SEG in timeline_est.py) to the whisper words (tools/align.py JSON) with difflib,
then: shot t0 = first word of its text - LEAD, subtitle chunks timed on their own words.
usage: MT_ROOT=/home/user/ep03 python3 timeline_voice.py audio/align_voice.json
Writes $MT_ROOT/timeline.json  {T, voff, shots, subs}."""
import json, os, re, sys, unicodedata, difflib, importlib.util, subprocess

ROOT = os.environ.get('MT_ROOT', '/home/user/ep03')
HERE = os.path.dirname(os.path.abspath(__file__))
EST_SRC = open(os.path.join(HERE, 'timeline_est.py')).read()
# reuse SEG / sentences / chunks without running the estimator's main code
ns = {}
exec(EST_SRC.split('\nt = 0.4')[0], ns)
SEG, sentences, chunks = ns['SEG'], ns['sentences'], ns['chunks']

VOFF = 0.6          # narration starts here in the video
LEAD = 0.35         # cut to a shot this long before its first word
TAIL = 3.2          # hold after the last word
DISPLAY = [         # subtitle display: digits where episode 2 used digits
    ('χίλια εννιακόσια σαράντα δύο', '1942'),
    ('Σχεδόν διακόσια παιδιά', 'Σχεδόν 200 παιδιά'),
    ('τέσσερις χιλιάδες παιδιά', '4.000 παιδιά'),
]


def norm(w):
    w = unicodedata.normalize('NFD', w.lower())
    w = ''.join(c for c in w if unicodedata.category(c) != 'Mn')
    return re.sub(r'[^\w]', '', w)


al = json.load(open(sys.argv[1] if os.path.isabs(sys.argv[1]) else f'{ROOT}/{sys.argv[1]}'))
ww = [w for s in al for w in s['words']]
wn = [norm(w['w']) for w in ww]

# script words, tagged with (seg index, chunk index)
script = []
seg_chunks = []
for si, (k, text, p) in enumerate(SEG):
    cl = [c for s in sentences(text) for c in chunks(s)]
    seg_chunks.append(cl)
    for ci, c in enumerate(cl):
        for w in c.split():
            script.append((si, ci, norm(w)))
sn = [x[2] for x in script]

sm = difflib.SequenceMatcher(a=sn, b=wn, autojunk=False)
m = {}
for a, b, n in sm.get_matching_blocks():
    for i in range(n):
        m[a + i] = b + i
matched = len(m) / len(sn)
print(f'matched {len(m)}/{len(sn)} script words ({matched:.0%})')

# whisper word edges drift into silences: snap them to the silence borders of the actual audio
AUDIO = os.environ.get('VOICE', f'{ROOT}/audio/voice_edit.wav')
log = subprocess.run(['ffmpeg', '-i', AUDIO, '-af', 'silencedetect=n=-42dB:d=0.18', '-f', 'null', '-'],
                     capture_output=True, text=True).stderr
SIL = list(zip([float(v) for v in re.findall(r'silence_start: ([0-9.]+)', log)],
               [float(v) for v in re.findall(r'silence_end: ([0-9.]+)', log)]))


def snap(s, e):
    for a, b in SIL:
        if a - 0.02 <= s < b: s = b
        if a < e <= b + 0.02: e = a
    return s, max(e, s + 0.1)


# time for every script word: matched -> whisper time; unmatched -> interpolate
times = [None] * len(sn)
for i, j in m.items():
    times[i] = snap(ww[j]['s'], ww[j]['e'])
known = [i for i in range(len(sn)) if times[i]]
for i in range(len(sn)):
    if times[i] is None:
        lo = max([k for k in known if k < i], default=None)
        hi = min([k for k in known if k > i], default=None)
        if lo is None: times[i] = (times[hi][0], times[hi][0])
        elif hi is None: times[i] = (times[lo][1], times[lo][1])
        else:
            f = (i - lo) / (hi - lo)
            t = times[lo][1] + f * (times[hi][0] - times[lo][1])
            times[i] = (t, t)

# group
subs, first = [], {}
idx = 0
for si, cl in enumerate(seg_chunks):
    for ci, c in enumerate(cl):
        n = len(c.split())
        s0, s1 = times[idx][0], times[idx + n - 1][1]
        txt = c
        for a, b in DISPLAY: txt = txt.replace(a, b)
        subs.append({'t0': round(s0 + VOFF, 2), 't1': round(s1 + VOFF + 0.15, 2), 'text': txt})
        first.setdefault(si, s0)
        idx += n
# subtitle never overlaps the next one
for a, b in zip(subs, subs[1:]):
    a['t1'] = min(a['t1'], b['t0'] - 0.02)

voice_end = ww[-1]['e']
shots = []
for si, (k, _, _) in enumerate(SEG):
    t0 = 0.0 if si == 0 else first[si] + VOFF - LEAD
    shots.append({'k': k, 't0': round(t0, 2)})
for a, b in zip(shots, shots[1:]):
    a['t1'] = b['t0']
T = round(voice_end + VOFF + TAIL, 2)
shots[-1]['t1'] = T
json.dump({'T': T, 'voff': VOFF, 'shots': shots, 'subs': subs, 'estimated': False},
          open(f'{ROOT}/timeline.json', 'w'), ensure_ascii=False, indent=1)
print('T =', T)
for s in shots: print(f"{s['t0']:7.2f} {s['t1'] - s['t0']:5.2f}  {s['k']}")
