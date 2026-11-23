"""Ep05 subtitles: chunks of <= 6 words per beat, digits for numbers (like ep04).
Times: estimated inside each beat (by characters) until the voice is aligned.
  subs05.py ass out.ass      -> ASS file (preview burn-in with ffmpeg/libass)"""
import os, sys, re, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from timeline05 import BEATS, ROOT

DISPLAY = [('χίλια εννιακόσια σαράντα πέντε', '1945'), ('Χίλια εννιακόσια δεκαπέντε', '1915'), ('χίλια εννιακόσια δώδεκα', '1912'),
           ('Έξι Αυγούστου', '6 Αυγούστου'), ('Εννιά Αυγούστου', '9 Αυγούστου'), ('Οχτώ και δεκαπέντε', '8:15'),
           ('είκοσι εννιά', '29'), ('Τρία χιλιόμετρα', '3 χιλιόμετρα'), ('ενενήντα τρία', '93'),
           ('χίλιοι πεντακόσιοι', '1.500'), ('χίλιοι διακόσιοι', '1.200'), ('από χίλιοι', 'από 1.000'),
           ('δεκατέσσερα λεπτά', '14 λεπτά'), ('Δεκαοχτώ λεπτά', '18 λεπτά')]
WEAK = {'τα', 'το', 'η', 'ο', 'οι', 'την', 'τον', 'της', 'των', 'του', 'από', 'σε', 'για', 'με',
        'και', 'να', 'ένα', 'ένας', 'μια', 'στο', 'στην', 'στα', 'στον', 'προς', 'που', 'ή', 'ότι'}

def clean(txt):
    t = re.sub(r'\[[^\]]*\]', ' ', txt)
    for a, b in DISPLAY: t = t.replace(a, b)
    return re.sub(r'\s+', ' ', t).strip()

def chunks(sent, maxw=6):
    out = []
    for part in re.split(r'(?<=[,.;])\s+', sent):
        w = part.split()
        if not w: continue
        n = -(-len(w) // maxw); size = -(-len(w) // n)
        out += [' '.join(w[i:i + size]) for i in range(0, len(w), size)]
    for i in range(len(out) - 1):
        w = out[i].split()
        while len(w) > 1 and w[-1].lower().strip('«»') in WEAK:
            out[i + 1] = w.pop() + ' ' + out[i + 1]
        out[i] = ' '.join(w)
    return [c for c in out if c]

def build(tl=None):
    tl = tl or json.load(open(f'{ROOT}/timeline.json'))
    subs = []
    for key, txt in BEATS:
        t0, t1 = tl[key]
        cs = chunks(clean(txt))
        tot = sum(len(c) for c in cs) or 1
        t = t0
        for c in cs:
            d = (t1 - t0) * len(c) / tot
            subs.append({'t0': round(t, 2), 't1': round(t + d + 0.15, 2), 'text': c})
            t += d
    for a, b in zip(subs, subs[1:]):
        a['t1'] = min(a['t1'], b['t0'] - 0.02)
    return subs

def ass_time(t):
    h = int(t // 3600); m = int(t % 3600 // 60); s = t % 60
    return f'{h}:{m:02d}:{s:05.2f}'

def write_ass(path, subs, y=1610):
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Sub,EB Garamond,58,&H00F6F2EC,&H00F6F2EC,&H00101010,&H90101010,0,0,0,0,100,100,0,0,1,2.2,2,2,40,40,{1920 - y - 30},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    with open(path, 'w') as f:
        f.write(head)
        for s in subs:
            f.write(f"Dialogue: 0,{ass_time(s['t0'])},{ass_time(s['t1'])},Sub,,0,0,0,,{{\\fad(80,80)}}{s['text']}\n")

if __name__ == '__main__':
    if sys.argv[1] == 'ass':
        subs = build()
        write_ass(sys.argv[2], subs)
        for s in subs[:12]: print(s)
        print(len(subs), 'subs')
