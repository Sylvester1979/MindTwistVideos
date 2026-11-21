"""Lengthen the pauses the TTS shortened ([long pause] tags are often ignored by v4).
Inserts silence at the silence nearest each cue time, writes audio/voice_edit.wav and a shifted
copy of the alignment (audio/align_edit.json) so timeline_voice.py can use it directly.
usage: MT_ROOT=/home/user/ep03 python3 edit_voice.py"""
import json, os, re, subprocess, numpy as np
ROOT = os.environ.get('MT_ROOT', '/home/user/ep03')
SR = 48000
SRC = f'{ROOT}/audio/voice.mp3'
# (approx. time in the original take, extra seconds) — section breaks of the script
ADD = [
    (46.35, 0.8),   # after Stefa                       [long pause]
    (57.55, 0.3),   # «Περιμένει να φύγει.»
    (66.60, 1.2),   # after Treblinka                   [long pause]
    (78.00, 0.4),   # after «μπογαλάκι»                 [pause]
    (91.30, 0.6),   # after the witness quote           [long pause]
    (102.20, 0.6),  # after «ούτε για ένα λεπτό»        [long pause]
    (106.60, 1.0),  # after «Κανείς τους δεν γύρισε»    [long pause]
    (119.85, 0.8),  # after «με τα λουλούδια»           [long pause]
    (122.00, 0.6),  # «Τα λουλούδια τα πότισε.» … «Τα παιδιά δεν τα άφησε.»
]

raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', SRC, '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-'],
                     capture_output=True, check=True).stdout
x = np.frombuffer(raw, np.float32).copy()
log = subprocess.run(['ffmpeg', '-i', SRC, '-af', 'silencedetect=n=-42dB:d=0.18', '-f', 'null', '-'],
                     capture_output=True, text=True).stderr
st = [float(v) for v in re.findall(r'silence_start: ([0-9.]+)', log)]
en = [float(v) for v in re.findall(r'silence_end: ([0-9.]+)', log)]
sil = list(zip(st, en))

cuts = []
for t, d in ADD:
    a, b = min(sil, key=lambda s: abs((s[0] + s[1]) / 2 - t))
    assert abs((a + b) / 2 - t) < 0.8, f'no silence near {t}'
    cuts.append(((a + b) / 2, d))
cuts.sort()

parts, prev = [], 0
for t, d in cuts:
    i = int(t * SR)
    parts += [x[prev:i], np.zeros(int(d * SR), np.float32)]
    prev = i
parts.append(x[prev:])
y = np.concatenate(parts)
y.astype(np.float32).tofile(f'{ROOT}/audio/voice_edit.f32')
subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'f32le', '-ar', str(SR), '-ac', '1', '-i', f'{ROOT}/audio/voice_edit.f32',
                f'{ROOT}/audio/voice_edit.wav'], check=True)


def shift(t):
    return t + sum(d for c, d in cuts if c <= t)


al = json.load(open(f'{ROOT}/audio/align_voice.json'))
for s in al:
    s['start'], s['end'] = shift(s['start']), shift(s['end'])
    for w in s['words']:
        w['s'], w['e'] = shift(w['s']), shift(w['e'])
json.dump(al, open(f'{ROOT}/audio/align_edit.json', 'w'), ensure_ascii=False)
print('voice_edit', round(len(y) / SR, 2), 's; cuts', [(round(c, 2), d) for c, d in cuts])
