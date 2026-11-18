"""Estimated timeline for the animatic (before the voice exists).
Each shot gets the narration text it covers; duration = words / WPS + pauses.
Writes $MT_ROOT/timeline.json  {T, shots:[{k,t0,t1}], subs:[{t0,t1,text}]}.
After the recording, timeline.py (alignment) replaces this file with real times."""
import json, os, re

ROOT = os.environ.get('MT_ROOT', '/home/user/ep03')
WPS = 2.25          # slow documentary Greek, words per second
PAUSE, LONG, SENT = 0.55, 1.3, 0.25
TAIL = 2.5          # hold after the last line

# (shot, text, pause after: 0 none / 1 pause / 2 long pause)
SEG = [
    ('A01_street',  'Τέσσερις Αυγούστου, χίλια εννιακόσια σαράντα δύο. Βαρσοβία.', 1),
    ('A02_window',  'Ένας γέρος γιατρός ποτίζει τα λουλούδια στο παράθυρό του.', 1),
    ('A03_guard',   'Απέναντι, ένας φρουρός με τουφέκι τον κοιτάζει.', 1),
    ('A04_diary',   'Εκείνο το βράδυ γράφει: «Το φαλακρό μου κεφάλι στο παράθυρο. Τι υπέροχος στόχος. Γιατί στέκεται και κοιτάζει ήρεμα;»', 2),
    ('B01_portrait','Τον λένε Γιάνους Κόρτσακ. Παιδίατρος, συγγραφέας παιδικών βιβλίων.', 1),
    ('B02_orphans', 'Διευθύνει ένα ορφανοτροφείο μέσα στο γκέτο. Σχεδόν διακόσια παιδιά.', 1),
    ('B03_wall',    'Φίλοι του, από την άλλη πλευρά του τείχους, του προσφέρουν ξανά και ξανά να τον φυγαδεύσουν.', 0),
    ('B04_refuses', 'Εκείνος αρνείται. Δεν αφήνει τα παιδιά.', 1),
    ('B05_stefa',   'Το ίδιο και η Στέφα, η συνεργάτιδά του για πάνω από τριάντα χρόνια.', 2),
    ('C01_play',    'Λίγες εβδομάδες πριν, βάζει τα παιδιά να ανεβάσουν ένα θεατρικό.', 1),
    ('C02_sickboy', 'Ένα άρρωστο αγόρι, κλεισμένο σε ένα δωμάτιο, περιμένει ένα γράμμα από τον βασιλιά. Περιμένει να φύγει.', 0),
    ('C03_watches', 'Ο Κόρτσακ ήθελε να τα προετοιμάσει.', 1),
    ('C04_rails',   'Τέσσερις μέρες μετά, φεύγουν τα πρώτα τρένα για την Τρεμπλίνκα.', 2),
    ('D01_gate',    'Ένα πρωί του Αυγούστου, έρχονται για τα παιδιά.', 1),
    ('D02_dress',   'Ο Κόρτσακ τους λέει να φορέσουν τα καλά τους ρούχα. Ότι πηγαίνουν στην εξοχή.', 1),
    ('D03_bundle',  'Τα πιο μικρά είναι δύο, τριών χρονών. Το καθένα κρατάει ένα μικρό μπογαλάκι.', 1),
    ('E01_march',   'Περπατούν σε σειρές μέσα στο γκέτο. Μπροστά, ο Κόρτσακ, κρατώντας ένα παιδί από το χέρι.', 1),
    ('E02_column',  'Ένας άνθρωπος που τους είδε έγραψε: «Δεν ήταν πορεία προς τα βαγόνια. Ήταν μια σιωπηλή διαμαρτυρία».', 2),
    ('F01_platform','Στην αποβάθρα, κάποιος προσπαθεί μια τελευταία φορά να τον βγάλει από εκεί. «Αρνήθηκε», θα πει αργότερα. «Δεν ήθελε να αφήσει τα παιδιά ούτε για ένα λεπτό».', 2),
    ('F02_wagon',   'Ανέβηκε στο τρένο μαζί τους.', 1),
    ('F03_empty',   'Κανείς τους δεν γύρισε.', 2),
    ('G01_count',   'Εκείνο το καλοκαίρι, περίπου τέσσερις χιλιάδες παιδιά από τα ορφανοτροφεία της Βαρσοβίας οδηγήθηκαν στην Τρεμπλίνκα.', 1),
    ('G02_hidden',  'Το ημερολόγιο του Κόρτσακ σώθηκε.', 0),
    ('G03_page',    'Η τελευταία του σελίδα τελειώνει με τα λουλούδια.', 2),
    ('H01_flowers', 'Τα λουλούδια τα πότισε. Τα παιδιά δεν τα άφησε.', 0),
]


def sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.;»:])\s+(?=[«Α-ΩΆ-Ώ])', text) if s.strip()]


def chunks(sent, maxw=6):
    """split a sentence into subtitle chunks: at commas first, then every <= maxw words."""
    out = []
    for part in re.split(r'(?<=,)\s+', sent):
        w = part.split()
        n = -(-len(w) // maxw)
        size = -(-len(w) // n)
        out += [' '.join(w[i:i + size]) for i in range(0, len(w), size)]
    # merge very short tails into the previous chunk
    merged = []
    for c in out:
        if merged and len(c.split()) <= 2 and len(merged[-1].split()) + len(c.split()) <= maxw + 1:
            merged[-1] += ' ' + c
        else:
            merged.append(c)
    return merged


t = 0.4
shots, subs = [], []
for k, text, p in SEG:
    t0 = t
    for s in sentences(text):
        for c in chunks(s):
            d = len(c.split()) / WPS
            subs.append({'t0': round(t, 2), 't1': round(t + d, 2), 'text': c})
            t += d
        t += SENT
    t += [0, PAUSE, LONG][p]
    shots.append({'k': k, 't0': round(t0, 2), 't1': round(t, 2)})
shots[-1]['t1'] = round(t + TAIL, 2)
T = shots[-1]['t1']
json.dump({'T': T, 'shots': shots, 'subs': subs, 'estimated': True},
          open(f'{ROOT}/timeline.json', 'w'), ensure_ascii=False, indent=1)
print('T =', T, 'shots', len(shots), 'subs', len(subs))
for s in shots: print(f"{s['t0']:6.1f} {s['t1']-s['t0']:5.1f}  {s['k']}")
