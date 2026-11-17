"""Assemble the edited narration from pieces (snapped to silences) and write timeline.json.
Each piece: id, source file, src start/end (seconds, None=auto-trim), gap after, subtitle chunks ('|' separated)."""
import json, subprocess, numpy as np, os, sys, re
ROOT = os.environ.get('MT_ROOT', '/home/user/ww1')
SR = 48000
TEMPO = float(os.environ.get('TEMPO', '1.04'))
LEAD = 0.45
FULL = f'{ROOT}/audio/voice_1803.mp3'
HERE = os.path.dirname(os.path.abspath(__file__))
_FIXF = next((p for p in (f'{HERE}/data/fixes.json', f'{ROOT}/audio/fixes.json') if os.path.exists(p)), None)
FIX = json.load(open(_FIXF)) if _FIXF else {}

P = [  # id, s0, s1, gap_after, subtitles
 ('date', 0.0, 4.313, .55, 'Έντεκα Νοεμβρίου 1918. | Πέντε το πρωί.'),
 ('carriage', 5.228, 10.286, .45, 'Σε ένα βαγόνι τρένου στη Γαλλία, | υπογράφεται το τέλος | του Πρώτου Παγκοσμίου Πολέμου.'),
 ('at11', 11.232, 13.993, .40, 'Όμως η ανακωχή | θα ισχύσει από τις έντεκα.'),
 ('generals', 14.722, 15.858, .30, 'Οι στρατηγοί το ξέρουν.'),
 ('attacks', 16.57, 19.195, .50, 'Κι όμως, | οι επιθέσεις συνεχίζονται.'),
 ('lastday', 20.071, 24.744, .85, 'Εκείνη την τελευταία μέρα... | σκοτώνονται πάνω από | δυόμισι χιλιάδες άνθρωποι.'),
 ('treb1', 26.132, 34.941, .35, 'Έντεκα παρά τέταρτο. Γαλλία. | Ο Ωγκυστέν Τρεμπυσόν, βοσκός, | σαράντα χρονών, | τρέχει να παραδώσει ένα σημείωμα | στους συντρόφους του:'),
 ('treb2', 35.479, 38.216, .45, '«Συγκέντρωση στις έντεκα και μισή, | για φαγητό».'),
 ('treb3', 39.063, 42.982, .80, 'Μια σφαίρα τον χτυπάει. | Πέφτει με το σημείωμα στο χέρι.'),
 ('price1', 44.262, 51.412, .40, 'Έντεκα παρά δύο. Βέλγιο. | Ο Καναδός Τζορτζ Πράις, | είκοσι πέντε χρονών, | βγαίνει από την πόρτα ενός σπιτιού.'),
 ('price2', 52.022, 55.554, .85, 'Ένας Γερμανός ελεύθερος σκοπευτής | τον πυροβολεί στο στήθος.'),
 ('gun1', 56.66, 61.064, .45, 'Έντεκα παρά ένα. | Ο Αμερικανός Χένρι Γκάνθερ, | είκοσι τριών χρονών.'),
 ('gun2', 61.552, 67.237, .50, 'Λίγους μήνες πριν, | τον είχαν υποβιβάσει, | γιατί σε ένα γράμμα έγραψε | την αλήθεια για τα χαρακώματα.'),
 ('gun3', 67.798, 71.617, .45, 'Τώρα σηκώνεται... | και τρέχει μόνος του | προς ένα γερμανικό πολυβόλο.'),
 ('gun4', 72.1, 76.64, .50, 'Οι Γερμανοί του φωνάζουν | να γυρίσει πίσω. | Ο πόλεμος τελειώνει σε δευτερόλεπτα.'),
 ('gun5', 77.249, 78.708, .35, 'Εκείνος συνεχίζει.'),
 ('gun6', 79.335, 80.166, .40, 'Τον πυροβολούν.'),
 ('gun7', 80.672, 84.492, 1.50, 'Θεωρείται ο τελευταίος στρατιώτης | που σκοτώθηκε σε αυτόν τον πόλεμο.'),
 ('eleven', 85.719, 88.086, 1.90, 'Έντεκα. | Τα όπλα σωπαίνουν.'),
 ('grave1', 89.242, 94.054, .50, 'Στον τάφο του Τρεμπυσόν γράφει: | «Πέθανε για τη Γαλλία, | δέκα Νοεμβρίου».'),
 ('grave2', 94.827, 95.692, .50, 'Δεν είναι αλήθεια.'),
 ('grave3', 96.474, 107.792, .90, 'Ίσως από ντροπή. | Ίσως για να μην μάθουν οι οικογένειες | ότι πέθαναν μέσα στις τελευταίες ώρες. | Κι έτσι, στους τάφους τους, | έγραψαν ότι πέθαναν | μία μέρα νωρίτερα.'),
 ('mons1', 109.093, 114.162, .45, 'Έξω από τη Μονς, στο Βέλγιο, | δύο τάφοι βρίσκονται | ο ένας απέναντι στον άλλον.'),
 ('mons2', 114.743, 120.223, .35, 'Στον έναν, αυτός που θεωρείται | ο πρώτος Βρετανός στρατιώτης | που σκοτώθηκε σε αυτόν τον πόλεμο.'),
 ('mons3', 120.743, 124.155, .70, 'Ένας έφηβος, | τον Αύγουστο του 1914.'),
 ('mons4', 124.765, 126.555, .55, 'Στον άλλον, ο τελευταίος Βρετανός.'),
 ('mons4b', 127.069, 130.087, .95, 'Σκοτώθηκε το πρωί της ανακωχής, | στα ίδια μέρη.'),
 ('end', 131.239, 134.144, 5.0, 'Ο πόλεμος τελείωσε... | εκεί που άρχισε.'),
]

def load(path):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-'],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).copy()

def autotrim(x, thr_db=-42, pad=0.06):
    env = np.sqrt(np.convolve(x ** 2, np.ones(480) / 480, 'same'))
    on = np.where(env > 10 ** (thr_db / 20))[0]
    a = max(0, on[0] - int(pad * SR)); b = min(len(x), on[-1] + int(pad * SR))
    return x[a:b], a / SR

def fade(x, ms=12):
    n = int(SR * ms / 1000); w = np.linspace(0, 1, n, dtype=np.float32)
    x = x.copy(); x[:n] *= w; x[-n:] *= w[::-1]; return x

def main():
    full = load(FULL)
    words_full = [w for s in json.load(open(f'{ROOT}/audio/align_1803.json')) for w in s['words']]
    out = [np.zeros(int(LEAD * TEMPO * SR), np.float32)]; t = LEAD
    pieces = []
    for pid, s0, s1, gap, sub in P:
        if pid in FIX:
            f = FIX[pid]
            x = load(f'{ROOT}/audio/{f["file"]}')
            if 's0' in f:
                x = x[int(f['s0'] * SR):int(f['s1'] * SR)]; off = f['s0']
            else:
                x, off = autotrim(x)
            words = [dict(w, s=w['s'] - off, e=w['e'] - off) for w in json.load(open(f'{ROOT}/audio/{f["words"]}'))] if 'words' in f else []
        else:
            x = full[int(s0 * SR):int(s1 * SR)]
            words = [dict(w, s=w['s'] - s0, e=w['e'] - s0) for w in words_full if w['s'] >= s0 - 0.05 and w['s'] < s1]
        x = fade(x)
        if pid in FIX and 'gain_db' in FIX[pid]:
            x = x * np.float32(10 ** (FIX[pid]['gain_db'] / 20))
        dur = len(x) / SR / TEMPO
        ws = [{'w': w['w'].strip(), 's': t + max(0, w['s']) / TEMPO, 'e': t + min(len(x) / SR, w['e']) / TEMPO} for w in words]
        pieces.append({'id': pid, 't0': t, 't1': t + dur, 'sub': [c.strip() for c in sub.split('|')], 'words': ws})
        out.append(x); out.append(np.zeros(int(gap * TEMPO * SR), np.float32))
        t += dur + gap
    y = np.concatenate(out)
    tmp = f'{ROOT}/audio/voice_cat.f32'
    y.astype(np.float32).tofile(tmp)
    af = f'atempo={TEMPO},highpass=f=70' if TEMPO != 1 else 'highpass=f=70'
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'f32le', '-ar', str(SR), '-ac', '1', '-i', tmp, '-af', af,
                    '-ar', str(SR), f'{ROOT}/audio/voice_edit.wav'], check=True)
    T = len(y) / SR / TEMPO
    # subtitles: map chunk boundaries by normalized character position onto word times
    subs = []
    for p in pieces:
        chunks = p['sub']; wl = p['words']
        L = sum(len(c) for c in chunks); acc = 0
        if wl:
            wc = np.cumsum([0] + [len(w['w']) + 1 for w in wl]).astype(float); wc /= wc[-1]
            ws_ = [w['s'] for w in wl] + [wl[-1]['e']]
        for i, c in enumerate(chunks):
            a = acc / L; b = (acc + len(c)) / L; acc += len(c)
            if wl:
                t0 = np.interp(a, wc, ws_); t1 = np.interp(b, wc, ws_)
            else:
                t0 = p['t0'] + a * (p['t1'] - p['t0']); t1 = p['t0'] + b * (p['t1'] - p['t0'])
            if i == 0: t0 = p['t0']
            if i == len(chunks) - 1: t1 = p['t1']
            subs.append({'t0': float(t0), 't1': float(t1), 'text': c, 'pid': p['id']})
    json.dump({'T': T, 'tempo': TEMPO, 'pieces': pieces, 'subs': subs}, open(f'{ROOT}/timeline.json', 'w'), ensure_ascii=False, indent=1)
    print('T=%.2f' % T)
    for p in pieces:
        print(f"{p['id']:9s} {p['t0']:7.2f} {p['t1']:7.2f}")

if __name__ == '__main__':
    main()
