"""Episode 4 timeline from the edited narration (audio/align_edit.json from edit_voice04.py).
Each shot owns a span of the script; shot t0 = its first word - LEAD. Subtitles timed on their own words.
Also prints the word-by-word check: script words that whisper did not hear.
usage: MT_ROOT=/home/user/ep04 python3 timeline04.py   -> $MT_ROOT/timeline.json"""
import json, os, re, unicodedata, difflib, subprocess
ROOT = os.environ.get('MT_ROOT', '/home/user/ep04')

# (shot, narration it covers) — the final approved script after the three cuts
SEG = [
    ('S01_hands',        'Επιζώντες του Ολοκαυτώματος αφήνουν πέτρες στον τάφο ενός ναζί.'),
    ('S02_crew',         'Είναι το τέλος της ταινίας «Η λίστα του Σίντλερ».'),
    ('S03_twohands',     'Και οι επιζώντες είναι αληθινοί.'),
    ('G04_map',          'Χίλια εννιακόσια τριάντα εννιά. Η Γερμανία εισβάλλει στην Πολωνία.'),
    ('S05f_schindler',   'Ο Όσκαρ Σίντλερ, μέλος του ναζιστικού κόμματος, φτάνει στην κατεχόμενη Κρακοβία για να πλουτίσει.'),
    ('S06_shop',         'Οι ναζί αρπάζουν τις επιχειρήσεις των Εβραίων.'),
    ('S07_factory',      'Ο Σίντλερ παίρνει ένα χρεοκοπημένο εργοστάσιο'),
    ('S08_pots',         'που φτιάχνει κατσαρόλες.'),
    ('S09_column',       'Αργότερα, για εργάτες, παίρνει τους πιο φτηνούς: Εβραίους από το γκέτο.'),
    ('S10_window',       'Όμως βλέπει τι κάνουν οι ναζί στους Εβραίους.'),
    ('S11_bribe',        'Και αρχίζει να τους προστατεύει. Με δωροδοκίες.'),
    ('S12_safe',         'Ξοδεύοντας όλη του την περιουσία.'),
    ('S13_plaszow',      'Χίλια εννιακόσια σαράντα τέσσερα. Ο σοβιετικός στρατός πλησιάζει, και το στρατόπεδο της Κρακοβίας αδειάζει.'),
    ('S14_wagons',       'Οι περισσότεροι κρατούμενοι, πάνω από είκοσι χιλιάδες, στέλνονται σε στρατόπεδα εξόντωσης.'),
    ('S15_desk',         'Ο Σίντλερ δωροδοκεί αξιωματικούς των Ες Ες. Τους πείθει ότι οι εργάτες του είναι απαραίτητοι για τον πόλεμο,'),
    ('S16_shells',       'και παίρνει άδεια να τους μεταφέρει σε ένα νέο εργοστάσιο πυρομαχικών.'),
    ('G17_list',         'Συντάσσεται μια λίστα με τα ονόματά τους.'),
    ('S18_set_typewriter', 'Στην ταινία, την υπαγορεύει ο ίδιος.'),
    ('S19_pemper',       'Στην πραγματικότητα, τη δακτυλογραφεί ένας Εβραίος κρατούμενος, ο Μίετεκ Πέμπερ.'),
    ('G20_count',        'Χίλια διακόσια ονόματα.'),
    ('S21_train',        'Χίλιοι διακόσιοι άνθρωποι, που γλιτώνουν από σχεδόν βέβαιο θάνατο.'),
    ('S22_old_schindler', 'Μετά τον πόλεμο, ο Σίντλερ ζει από τα χρήματα που του στέλνουν όσοι έσωσε. Πεθαίνει φτωχός, το χίλια εννιακόσια εβδομήντα τέσσερα.'),
    ('S23_zion',         'Και θάβεται, όπως ζήτησε, στην Ιερουσαλήμ.'),
    ('S24_beverly',      'Χίλια εννιακόσια ογδόντα. Μπέβερλι Χιλς.'),
    ('S25_shop',         'Ένας Αυστραλός συγγραφέας ψάχνει χαρτοφύλακα σε ένα μαγαζί.'),
    ('S26_poldek',       'Ο ιδιοκτήτης λέγεται Πόλντεκ Πφέφερμπεργκ. Και είναι ένας από τους χίλιους διακόσιους. Του λέει την ιστορία.'),
    ('S27_book_reel',    'Γράφεται ένα βιβλίο. Ο Σπίλμπεργκ το κάνει ταινία.'),
    ('S28_set_night',    'Στα γυρίσματα βρίσκεται και η Μίλα, η γυναίκα του Πόλντεκ, που ήταν κι εκείνη στη λίστα.'),
    ('S29_mila',         'Της συστήνουν τον ηθοποιό που παίζει τον Άμον Γκαιτ, τον διοικητή του στρατοπέδου της.'),
    ('Q30_goeth',        '«Όταν έβλεπες τον Γκαιτ», έλεγε ο Πόλντεκ, «έβλεπες τον θάνατο».'),
    ('S31_hands_tremble', 'Με τη στολή, ο ηθοποιός μοιάζει τόσο με τον αληθινό... που η Μίλα αρχίζει να τρέμει.'),
    ('S32_redcoat',      'Η ταινία είναι σχεδόν ολόκληρη ασπρόμαυρη. Μέσα της ξεχωρίζει ένα κοριτσάκι με κόκκινο παλτό. Για τον Σπίλμπεργκ, ήταν μια κόκκινη σημαία μπροστά σε έναν κόσμο που δεν ήθελε να δει.'),
    ('S33_redcoat',      '«Η Αμερική ήξερε», είπε. «Η Βρετανία ήξερε. Η Ρωσία ήξερε». Και δεν έκαναν τίποτα.'),
    ('S34_zion1993',     'Η τελευταία σκηνή γυρίζεται στον τάφο του Σίντλερ. Πάνω από εκατόν είκοσι επιζώντες περνούν, ένας ένας, και αφήνουν μια πέτρα.'),
    ('S35_slab',         'Στην πλάκα, στα γερμανικά, γράφει: «Ο αξέχαστος σωτήρας χιλίων διακοσίων κατατρεγμένων Εβραίων».'),
    ('G36_final',        'Οι απόγονοί τους υπολογίζονται σε πάνω από οχτώμισι χιλιάδες. Όλοι τους... χάρη σε μια λίστα.'),
]
VOFF = 0.5          # narration starts here in the video
LEAD = 0.35         # cut to a shot this long before its first word
HOLD = 2.6          # hold on the last image after the last word
DISPLAY = [         # subtitles: digits like the earlier episodes
    ('Χίλια εννιακόσια τριάντα εννιά', '1939'), ('Χίλια εννιακόσια σαράντα τέσσερα', '1944'),
    ('χίλια εννιακόσια εβδομήντα τέσσερα', '1974'), ('Χίλια εννιακόσια ογδόντα', '1980'),
    ('είκοσι χιλιάδες', '20.000'), ('Χίλια διακόσια', '1.200'), ('Χίλιοι διακόσιοι', '1.200'),
    ('χίλιους διακόσιους', '1.200'), ('χιλίων διακοσίων', '1.200'), ('εκατόν είκοσι', '120'),
    ('οχτώμισι χιλιάδες', '8.500'), ('Ες Ες', 'SS'),
]


def sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.;»:])\s+(?=[«Α-ΩΆ-Ώ])', text) if s.strip()]


def chunks(sent, maxw=6):
    out = []
    for part in re.split(r'(?<=,)\s+', sent):
        w = part.split()
        n = -(-len(w) // maxw)
        size = -(-len(w) // n)
        out += [' '.join(w[i:i + size]) for i in range(0, len(w), size)]
    WEAK = {'τα', 'το', 'η', 'ο', 'οι', 'την', 'τον', 'της', 'των', 'του', 'από', 'σε', 'για', 'με',
            'και', 'να', 'ένα', 'ένας', 'μια', 'στο', 'στην', 'στα', 'στον', 'προς', 'που'}
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


if __name__ == '__main__':
    al = json.load(open(f'{ROOT}/audio/align_edit.json'))
    ww = [w for s in al for w in s['words']]
    wn = [norm(w['w']) for w in ww]
    script, seg_chunks = [], []
    for si, (k, text) in enumerate(SEG):
        cl = [c for s in sentences(text) for c in chunks(s)]
        seg_chunks.append(cl)
        for c in cl:
            for w in c.split():
                if norm(w): script.append((si, norm(w), w))
    sn = [x[1] for x in script]
    sm = difflib.SequenceMatcher(a=sn, b=wn, autojunk=False)
    m = {}
    for a, b, n in sm.get_matching_blocks():
        for i in range(n): m[a + i] = b + i
    print(f'matched {len(m)}/{len(sn)} script words ({len(m) / len(sn):.0%})')
    miss = [script[i][2] for i in range(len(sn)) if i not in m]
    extra = [ww[j]['w'].strip() for j in range(len(wn)) if j not in set(m.values())]
    print('script words not heard:', miss)
    print('heard but not in script:', extra)

    log = subprocess.run(['ffmpeg', '-i', f'{ROOT}/audio/voice_edit.wav', '-af', 'silencedetect=n=-42dB:d=0.18', '-f', 'null', '-'],
                         capture_output=True, text=True).stderr
    SIL = list(zip([float(v) for v in re.findall(r'silence_start: ([0-9.]+)', log)],
                   [float(v) for v in re.findall(r'silence_end: ([0-9.]+)', log)]))

    def snap(s, e):
        for a, b in SIL:
            if a - 0.02 <= s < b: s = b
            if a < e <= b + 0.02: e = a
        return s, max(e, s + 0.1)

    times = [None] * len(sn)
    for i, j in m.items(): times[i] = snap(ww[j]['s'], ww[j]['e'])
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
            n = len([w for w in c.split() if norm(w)])
            s0, s1 = times[idx][0], times[idx + n - 1][1]
            txt = c
            for a, b in DISPLAY: txt = txt.replace(a, b)
            subs.append({'t0': round(s0 + VOFF, 2), 't1': round(s1 + VOFF + 0.15, 2), 'text': txt, 'k': SEG[si][0]})
            first.setdefault(si, s0)
            idx += n
    for a, b in zip(subs, subs[1:]): a['t1'] = min(a['t1'], b['t0'] - 0.02)
    shots = [{'k': k, 't0': 0.0 if si == 0 else round(first[si] + VOFF - LEAD, 2)} for si, (k, _) in enumerate(SEG)]
    for a, b in zip(shots, shots[1:]): a['t1'] = b['t0']
    T = round(times[-1][1] + VOFF + HOLD, 2)
    shots[-1]['t1'] = T
    json.dump({'T': T, 'voff': VOFF, 'shots': shots, 'subs': subs}, open(f'{ROOT}/timeline.json', 'w'),
              ensure_ascii=False, indent=1)
    print('T =', T)
    for s in shots: print(f"{s['t0']:7.2f} {s['t1'] - s['t0']:5.2f}  {s['k']}")
