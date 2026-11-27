"""Episode 5 timeline from the edited narration.
Word times come from audio/align_edit.json (tools/align.py on voice_edit.wav) when it exists. Until then they are
interpolated inside each phrase of audio/beats_edit.json (silence-based beats from align05z.py) by letter count:
good to within a few tenths of a second, enough for a draft; the final timeline uses the whisper words.
Each shot owns a span of the script; shot t0 = its first word - LEAD. Subtitles are timed on their own words.
usage: MT_ROOT=/home/user/ep05 python3 timeline05.py   -> $MT_ROOT/timeline.json"""
import json, os, re, unicodedata, difflib, subprocess
ROOT = os.environ.get('MT_ROOT', '/home/user/ep05')

# (shot, narration it covers): script v8 as recorded, minus «Σε λίγες εβδομάδες, φεύγουν κι οι Γερμανοί» (cut in the edit)
SEG = [
    ('C3D_hook',  'Οι Γερμανοί ζήτησαν μια λίστα με όλους τους Εβραίους του νησιού. Πήραν μια λίστα με δύο ονόματα.'),
    ('G02_map',   'Ζάκυνθος. Σεπτέμβριος του χίλια εννιακόσια σαράντα τρία.'),
    ('S03',       'Οι Ιταλοί, που κρατούσαν το νησί, συνθηκολογούν.'),
    ('S04',       'Έρχονται οι Γερμανοί.'),
    ('S05',       'Στη Ζάκυνθο ζουν διακόσιοι εβδομήντα πέντε Εβραίοι.'),
    ('S06',       'Οι Εβραίοι της Θεσσαλονίκης έχουν ήδη σταλεί στο Άουσβιτς, το στρατόπεδο εξόντωσης.'),
    ('S07',       'Στα τέλη της χρονιάς, ο Γερμανός διοικητής του νησιού καλεί τον δήμαρχο,'),
    ('S08',       'τον Λουκά Καρρέρ. Θέλει τη λίστα των Εβραίων.'),
    ('G09_page',  'Ονόματα, διευθύνσεις, περιουσίες.'),
    ('S10',       'Μια τέτοια λίστα σημαίνει ένα πράγμα: εκτοπισμό.'),
    ('G11_clock', 'Αν ο δήμαρχος δεν τη φέρει ως την επόμενη μέρα, θα το πληρώσει με τη ζωή του.'),
    ('S12',       'Ο δήμαρχος έχει ήδη στα χέρια του τα ονόματα.'),
    ('S13',       'Πηγαίνει στον Μητροπολίτη του νησιού,'),
    ('S14',       'τον Χρυσόστομο. Η συμβουλή του είναι μία:'),
    ('B3D',       'να κάψει τη λίστα. Και η λίστα καίγεται.'),
    ('S16',       'Ο Μητροπολίτης πηγαίνει ο ίδιος στον διοικητή.'),
    ('S17',       'Οι Εβραίοι του νησιού, του λέει, είναι Έλληνες πολίτες και δεν έβλαψαν κανέναν.'),
    ('G18_orders', 'Ο διοικητής επιμένει. Ζητά τη λίστα, ξανά και ξανά.'),
    ('C3D',       'Και τότε, ο δήμαρχος και ο Μητροπολίτης παραδίδουν στον διοικητή ένα χαρτί. Πάνω του, στα ελληνικά και στα γερμανικά, δύο ονόματα: του Μητροπολίτη και του δημάρχου. Τίποτ\' άλλο.'),
    ('S20',       'Αν οι Γερμανοί ήθελαν Εβραίους, θα έπρεπε να πάρουν πρώτα αυτούς τους δύο.'),
    ('S21',       'Ο διοικητής δεν εκτελεί την απειλή του.'),
    ('S22',       'Στο μεταξύ, με οδηγία του Μητροπολίτη,'),
    ('S23',       'οι Εβραίοι κρύβονται σε χριστιανικά σπίτια, στα χωριά.'),
    ('S24',       'Οι αντάρτες, τους προειδοποιούν για τις εφόδους.'),
    ('S25',       'Και ούτε ένας χωρικός δεν τους προδίδει.'),
    ('G26_corfu', 'Ιούνιος του σαράντα τέσσερα. Οι Εβραίοι της Κέρκυρας, χίλιοι οχτακόσιοι, στέλνονται στο Άουσβιτς.'),
    ('S27',       'Τον Αύγουστο, γερμανικά σκάφη έρχονται για τους Εβραίους της Ζακύνθου.'),
    ('S28',       'Φεύγουν χωρίς κανέναν.'),
    ('G29_greece', 'Στην Ελλάδα, οι ναζί δολοφόνησαν πάνω από οχτώ στους δέκα Εβραίους.'),
    ('S30',       'Από τη Ζάκυνθο, κανείς δεν παραδόθηκε. Κανείς δεν μπήκε σε πλοίο.'),
    ('C3D_still', 'Το χαρτί με τα δύο ονόματα το διηγήθηκαν οι ίδιοι, και κάποιοι ερευνητές αμφιβάλλουν.'),
    ('B3D_ash',   'Βέβαιο είναι ότι η λίστα των Εβραίων δεν παραδόθηκε ποτέ.'),
    ('S33',       'Και ότι τους έσωσαν πολλοί. Οι χωρικοί, οι αντάρτες.'),
    ('S34',       'Κάποιοι ιστορικοί υποστηρίζουν ότι ίσως βοήθησε κι ο ίδιος ο Γερμανός διοικητής, καθυστερώντας τον εκτοπισμό.'),
    ('S35',       'Τους έσωσε ένα ολόκληρο νησί.'),
    ('G36_plaque', 'Το χίλια εννιακόσια εβδομήντα οχτώ, το Ισραήλ τιμά τον δήμαρχο και τον Μητροπολίτη ως Δίκαιους των Εθνών: τίτλος που δόθηκε σε όσους έσωσαν Εβραίους κατά τον Δεύτερο Παγκόσμιο Πόλεμο.'),
    ('G37_names', 'Μερικές λίστες σώζουν ανθρώπους με τα ονόματα που γράφουν.'),
    ('D3D',       'Στη Ζάκυνθο, η σωτηρία γράφτηκε μέσα από τις στάχτες μιας λίστας.'),
]
NB = '¦'          # joins the words of a number so it is never split (not whitespace, not a letter)
VOFF = 0.4          # narration starts here in the video (voice + 0.4 s, as in the storyboard)
LEAD = 0.35         # cut to a shot this long before its first word
HOLD = 2.6          # hold on the last image after the last word
DISPLAY = [         # subtitles: digits like the earlier episodes
    ('χίλια εννιακόσια σαράντα τρία', '1943'), ('σαράντα τέσσερα', '1944'), ('χίλια εννιακόσια εβδομήντα οχτώ', '1978'),
    ('διακόσιοι εβδομήντα πέντε', '275'), ('χίλιοι οχτακόσιοι', '1.800'), ('πάνω από οχτώ στους δέκα', 'πάνω από 8 στους 10'),
]


def sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.;»:])\s+(?=[«Α-ΩΆ-Ώα-ωά-ώ])', text) if s.strip()]


def chunks(sent, maxw=6):
    out = []
    for part in re.split(r'(?<=,)\s+', sent):
        w = part.split()
        n = -(-len(w) // maxw)
        size = -(-len(w) // n)
        out += [' '.join(w[i:i + size]) for i in range(0, len(w), size)]
    WEAK = {'τα', 'το', 'η', 'ο', 'οι', 'την', 'τη', 'τον', 'της', 'των', 'του', 'τους', 'από', 'σε', 'για', 'με',
            'και', 'να', 'ένα', 'ένας', 'μια', 'στο', 'στη', 'στην', 'στα', 'στον', 'στους', 'προς', 'που', 'κι'}
    for i in range(len(out) - 1):
        w = out[i].split()
        while len(w) > 1 and w[-1].lower() in WEAK:
            out[i + 1] = w.pop() + ' ' + out[i + 1]
        out[i] = ' '.join(w)
    return [c for c in out if c]


def norm(w):
    w = unicodedata.normalize('NFD', w.lower())
    w = ''.join(c for c in w if unicodedata.category(c) != 'Mn')
    return re.sub(r'[^\w]', '', w)


def words_from_beats():
    """no ASR yet: spread each beat's words over its span by letter count (+1 per word for the gap)."""
    out = []
    for b in json.load(open(f'{ROOT}/audio/beats_edit.json')):
        ws = b['text'].split()
        L = [len(norm(w)) + 1 for w in ws]; tot = sum(L); t = b['t0']; span = b['t1'] - b['t0']
        for w, l in zip(ws, L):
            d = span * l / tot
            out.append({'w': w, 's': t, 'e': t + d * 0.92}); t += d
    return out


if __name__ == '__main__':
    p = f'{ROOT}/audio/align_edit.json'
    if os.path.exists(p):
        ww = [w for s in json.load(open(p)) for w in s['words']]; src = 'whisper'
    else:
        ww = words_from_beats(); src = 'beats (draft)'
    print('word times from', src)
    wn = [norm(w['w']) for w in ww]
    script, seg_chunks = [], []
    for si, (k, text) in enumerate(SEG):
        for a, _ in DISPLAY: text = text.replace(a, a.replace(' ', NB))      # a number is never split across subtitles
        cl = [c for s in sentences(text) for c in chunks(s)]
        seg_chunks.append(cl)
        for c in cl:
            for w in c.replace(NB, ' ').split():
                if norm(w): script.append((si, norm(w), w))
    sn = [x[1] for x in script]
    sm = difflib.SequenceMatcher(a=sn, b=wn, autojunk=False)
    m, rep = {}, []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            for k in range(i2 - i1): m[i1 + k] = j1 + k
        elif tag == 'replace': rep.append((i1, i2, j1, j2))    # misheard (Ζάκινθος) or a number heard as digits (1943)
    print(f'matched {len(m)}/{len(sn)} script words ({len(m) / len(sn):.0%})')
    print('heard differently:', ['/'.join(x[2] for x in script[i1:i2]) + ' = ' + '/'.join(w['w'].strip() for w in ww[j1:j2])
                                 for i1, i2, j1, j2 in rep])
    inrep = {i for i1, i2, _, _ in rep for i in range(i1, i2)}
    print('script words not heard:', [script[i][2] for i in range(len(sn)) if i not in m and i not in inrep])

    SIL = []
    if src == 'whisper':
        log = subprocess.run(['ffmpeg', '-i', f'{ROOT}/audio/voice_edit.wav', '-af', 'silencedetect=n=-42dB:d=0.18', '-f', 'null', '-'],
                             capture_output=True, text=True).stderr
        SIL = list(zip([float(v) for v in re.findall(r'silence_start: ([0-9.]+)', log)],
                       [float(v) for v in re.findall(r'silence_end: ([0-9.]+)', log)]))

    def snap(s, e):
        for a, b in SIL:
            if s >= a - 0.02 and e <= b + 0.02:      # whisper put the whole word inside a pause: it was said just before it
                return max(a - (e - s), 0.0), a
        for a, b in SIL:
            if a - 0.02 <= s < b: s = b
            if a < e <= b + 0.02: e = a
        return s, max(e, s + 0.1)

    times = [None] * len(sn)
    for i, j in m.items(): times[i] = snap(ww[j]['s'], ww[j]['e'])
    for i1, i2, j1, j2 in rep:                       # share the heard span among the script words by letters
        s0, e0 = snap(ww[j1]['s'], ww[j1]['e'])[0], snap(ww[j2 - 1]['s'], ww[j2 - 1]['e'])[1]
        L = [len(script[i][1]) + 1 for i in range(i1, i2)]; t = s0
        for i, l in zip(range(i1, i2), L):
            d = (e0 - s0) * l / sum(L); times[i] = (t, t + d); t += d
    known = [i for i in range(len(sn)) if times[i]]
    for i in range(len(sn)):
        if times[i] is None:
            lo = max([k for k in known if k < i], default=None); hi = min([k for k in known if k > i], default=None)
            if lo is None: times[i] = (times[hi][0],) * 2
            elif hi is None: times[i] = (times[lo][1],) * 2
            else:
                t = times[lo][1] + (i - lo) / (hi - lo) * (times[hi][0] - times[lo][1]); times[i] = (t, t)
    subs, first, idx = [], {}, 0
    for si, cl in enumerate(seg_chunks):
        for c in cl:
            n = len([w for w in c.replace(NB, ' ').split() if norm(w)])
            s0, s1 = times[idx][0], times[idx + n - 1][1]
            txt = c.replace(NB, ' ')
            for a, b in DISPLAY: txt = txt.replace(a, b)
            subs.append({'t0': round(s0 + VOFF, 2), 't1': round(s1 + VOFF + 0.15, 2), 'text': txt, 'k': SEG[si][0]})
            first.setdefault(si, s0)
            idx += n
    for a, b in zip(subs, subs[1:]): a['t1'] = min(a['t1'], b['t0'] - 0.02)
    shots = [{'k': k, 't0': 0.0 if si == 0 else round(first[si] + VOFF - LEAD, 2)} for si, (k, _) in enumerate(SEG)]
    for a, b in zip(shots, shots[1:]): a['t1'] = b['t0']
    T = round(times[-1][1] + VOFF + HOLD, 2)
    shots[-1]['t1'] = T
    json.dump({'T': T, 'voff': VOFF, 'src': src, 'shots': shots, 'subs': subs}, open(f'{ROOT}/timeline.json', 'w'),
              ensure_ascii=False, indent=1)
    print('T =', T)
    for s in shots: print(f"{s['t0']:7.2f} {s['t1'] - s['t0']:5.2f}  {s['k']}")
