"""Episode 2 renderer: 'Ένα λεπτό πριν την ειρήνη'. Usage:
   python3 render.py stills t1 t2 ...      -> test/still_<t>.jpg
   python3 render.py video f0 f1 out.mp4    -> frames [f0,f1)"""
import sys, os, json, math, subprocess, functools
import numpy as np, cv2
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'engine'))
from eng import *
from eng import _ink_noise
from gfx import *

TL = json.load(open(f'{ROOT}/timeline.json'))
T = TL['T']
PC = {p['id']: p for p in TL['pieces']}
SUBS = TL['subs']

def P0(pid): return PC[pid]['t0']
def P1(pid): return PC[pid]['t1']

def cue(pid, *subs, frac=None):
    """time of the first word in piece containing any of the substrings (case-insensitive)."""
    p = PC[pid]
    for w in p['words']:
        lw = w['w'].lower()
        if any(s.lower() in lw for s in subs):
            return w['s']
    if frac is None: frac = 0.5
    return p['t0'] + frac * (p['t1'] - p['t0'])

FILM = Film()
RAIN = Rain(n=700, alpha=0.16)
RAIN2 = Rain(n=500, seed=9, angle=0.1, speed=2200, alpha=0.12)
MOTES = Motes()

@functools.lru_cache(None)
def plate(k):
    return Plate(k)

def lerp3(a, b, u):
    return tuple(lerp(x, y, u) for x, y in zip(a, b))

# ------------------------------------------------------------------ overlays
SUBY = 1590

def subtitles(img, t):
    for s in SUBS:
        if s['t0'] - 0.05 <= t <= s['t1'] + 0.12:
            a = clamp01((t - s['t0'] + 0.05) / 0.12) * clamp01((s['t1'] + 0.12 - t) / 0.12)
            if a <= 0: continue
            al = sub_alpha(s['text'])
            img = band(img, SUBY - 50, SUBY + 50, 0.45 * a, 70)
            img = paste_alpha(img, al, W / 2, SUBY, (236, 241, 245), a, shadow=0.9, rough=False)
    return img

@functools.lru_cache(None)
def sub_alpha(text):
    size = 56
    while True:
        al = text_layer([text], 'serif', size, 560)
        if al.shape[1] < W - 70 or size <= 40: return al
        size -= 2

def time_chip(img, a, hhmm, sub=None, y=330):
    if a <= 0: return img
    img = band(img, y - 110, y + (90 if sub else 50), 0.62 * a, 90)
    al = text_layer([hhmm], 'type', 168, None)
    img = paste_alpha(img, al, W / 2, y - 20, (236, 241, 245), a, shadow=1.0)
    if sub:
        al2 = fit_text(sub, 50, 600, 6)
        img = paste_alpha(img, al2, W / 2, y + 95, (215, 222, 228), a, shadow=0.8)
    return img

@functools.lru_cache(None)
def fit_text(text, size, wght, track):
    while True:
        al = text_layer([text], 'serif', size, wght, track=track)
        if al.shape[1] < W - 60 or size <= 26: return al
        size -= 2

def name_chip(img, a, name, sub, y=1395):
    if a <= 0: return img
    img = band(img, y - 50, y + 60, 0.42 * a, 60)
    al = text_layer([name], 'serif', 64, 650)
    img = paste_alpha(img, al, W / 2, y - 18, (236, 241, 245), a, shadow=0.9)
    al2 = fit_text(sub, 46, 560, 2)
    img = paste_alpha(img, al2, W / 2, y + 46, (222, 228, 233), a, shadow=0.9)
    return img

def fade_win(t, t0, t1, fin=0.35, fout=0.35):
    return clamp01((t - t0) / fin) * clamp01((t1 - t) / fout)

def glow_at(img, x, y, r, amt, color=(200, 230, 255)):
    yy, xx = np.ogrid[0:H, 0:W]
    g = np.exp(-(((xx - x) ** 2 + (yy - y) ** 2) / (2 * r * r))).astype(np.float32)[..., None] * amt
    return img + g * np.array(color, np.float32)

def u_of(t, t0, t1):
    return clamp01((t - t0) / max(1e-3, t1 - t0))

# ------------------------------------------------------------------ shots
def plate_shot(key, cam0, cam1, par0=(0, 0, 0), par1=(0, 0, 0), t0=0, t1=1, **kw):
    def fn(t, fi):
        u = ease2(u_of(t, t0, t1))
        return plate(key).render(t, fi, lerp3(cam0, cam1, u), lerp3(par0, par1, u), **kw)
    return fn

SH = []
def add(t0, t1, fn, tr=0.5, kind='ink', seed=11, center=(0.5, 0.5)):
    SH.append(dict(t0=t0, t1=t1, fn=fn, tr=tr, kind=kind, seed=seed, center=center))

def build():
    SH.clear()
    t0 = t1 = 0.0
    # 1 — watch, 5:00
    tA, tB = 0.0, P0('carriage') + 0.15
    t5 = cue('date', 'πέντε', frac=0.75)
    def s_watch_open(t, fi, t0=t0, t1=t1):
        u = u_of(t, tA, tB)
        img = watch_scene(t, fi, 5 * 3600 + (t - t5), cam=(0.5, 0.47, lerp(0.92, 1.06, ease2(u)), lerp(-0.03, 0.0, u)))
        a = fade_win(t, 0.25, tB + 1, 0.6, 0.4)
        img = band(img, 120, 260, 0.35 * a, 80)
        al = text_layer(['11 · ΝΟΕΜΒΡΙΟΥ · 1918'], 'type', 64, None)
        return paste_alpha(img, al, W / 2, 190, (230, 236, 240), a, shadow=0.9)
    add(tA, tB, s_watch_open, tr=0)

    # 2 — carriage in the forest
    t0, t1 = P0('carriage') - 0.15, cue('carriage', 'υπογράφ', frac=0.4)
    add(t0, t1, plate_shot('P01_carriage', (768, 1400, 1.3), (720, 1560, 1.6), (-25, 4, 0), (25, -4, 0), t0, t1 + 0.6,
                           fog=0.35, warm_flicker=1.0, haze=0.15), tr=0.7, seed=12)
    # 3 — the signature
    t0, t1 = t1, P0('at11') + 0.05
    s_sign = plate_shot('P02_signing', (870, 1390, 1.75), (660, 1460, 2.0), (0, 0, -40), (0, 0, 40), t0, t1 + 0.4, warm_flicker=1.0)
    def s_sign2(t, fi, t0=t0, t1=t1):
        img = s_sign(t, fi)
        return time_chip(img, fade_win(t, t0 + 0.3, t1 + 0.5, 0.4, 0.4), '5:12', 'ΔΑΣΟΣ ΤΗΣ ΚΟΜΠΙΕΝΗΣ')
    add(t0, t1, s_sign2, tr=0.5, seed=13)
    # 4 — the hands sweep 5 -> 11 : six hours
    t0, t1 = t1, P0('generals') - 0.1
    ss0, ss1 = t0 + 0.25, P1('at11') + 0.1
    def s_sweep(t, fi, t0=t0, t1=t1):
        p = ease2(u_of(t, ss0, ss1))
        clock = lerp(5 * 3600 + 3, 11 * 3600, p)
        img = watch_scene(t, fi, clock, cam=(0.5, 0.47, lerp(1.0, 1.08, u_of(t, t0, t1)), 0.0), arc=(5 * 3600, 11 * 3600, p))
        a = fade_win(t, ss1 - 0.4, t1 + 1, 0.4, 0.3)
        al = text_layer(['6 ώρες'], 'serif', 110, 600)
        return paste_alpha(img, al, W * 0.5, H * 0.10, (60, 62, 190), a, shadow=0.7)
    add(t0, t1, s_sweep, tr=0.45, seed=14)
    # 5 — generals
    t0, t1 = t1, P0('attacks') - 0.05
    add(t0, t1, plate_shot('P03_hq', (830, 1280, 1.55), (820, 1340, 1.8), (-20, 0, 0), (20, 0, 0), t0, t1 + 0.4, haze=0.1), tr=0.4, seed=15)
    # 6 — attacks continue (trench)
    t0, t1 = t1, P0('lastday') - 0.1
    s_tr = plate_shot('P04_trench', (780, 1400, 1.4), (820, 1260, 1.65), (0, 0, -70), (0, 0, 70), t0, t1 + 0.4, fog=0.25, haze=0.15)
    def s_trench(t, fi, t0=t0, t1=t1):
        img = s_tr(t, fi)
        img = RAIN.apply(img, t, 0.9)
        for tf in (t0 + 0.5, t0 + 1.4, t0 + 2.1):
            d = t - tf
            if 0 <= d < 0.6:
                img = glow_at(img, 1100, 700, 520, 0.55 * math.exp(-d * 7), (170, 210, 245))
        return img
    add(t0, t1, s_trench, tr=0.35, seed=16)
    # 7 — 2.738
    t0, t1 = t1, P0('treb1') - 0.15
    c0 = cue('lastday', 'σκοτών', frac=0.35) - 0.2
    def s_count(t, fi, t0=t0, t1=t1):
        p = u_of(t, c0, P1('lastday') - 0.2)
        return counter_frame(t, fi, p, label=clamp01((t - P1('lastday') + 0.6) / 0.6))
    add(t0, t1, s_count, tr=0.6, seed=17)
    # 8 — map: Vrigne-Meuse 10:45
    t0, t1 = t1, cue('treb1', 'ογκιστ', 'ωγκυ', frac=0.3) - 0.1
    def s_map1(t, fi, t0=t0, t1=t1):
        u = ease2(u_of(t, t0 + 0.3, t1))
        cam = (lerp(4.2, 4.84, u), lerp(49.45, 49.66, u), lerp(1.0, 2.5, u))
        img = map_frame(t, fi, cam, front=ease2(u_of(t, t0, t0 + 1.6)), marks=[('vrigne', u_of(t, t0 + 0.9, t0 + 1.6), None)])
        return time_chip(img, fade_win(t, t0 + 0.35, t1 + 0.6, 0.3, 0.3), '10:45', 'ΓΑΛΛΙΑ · VRIGNE-MEUSE')
    add(t0, t1, s_map1, tr=0.5, seed=18)
    # 9 — Trébuchon running on the embankment
    t0, t1 = t1, P0('treb2') - 0.1
    s_run = plate_shot('P05_meuse', (850, 1480, 1.4), (960, 1690, 1.8), (-30, 0, 0), (30, 0, 0), t0, t1 + 0.4, fog=0.35, haze=0.2, fog_speed=(26, -2))
    def s_treb(t, fi, t0=t0, t1=t1):
        img = RAIN2.apply(s_run(t, fi), t, 0.8)
        return name_chip(img, fade_win(t, t0 + 0.5, t1 + 0.3, 0.4, 0.4), 'Ωγκυστέν Τρεμπυσόν', '40 χρονών · βοσκός · αγγελιοφόρος')
    add(t0, t1, s_treb, tr=0.5, seed=19)
    # 10 — the note being written
    t0, t1 = t1, P0('treb3') - 0.05
    def s_note(t, fi, t0=t0, t1=t1):
        p = u_of(t, t0 + 0.15, P1('treb2') - 0.1)
        bg = wood_bg(); out, _ = view(bg, bg.shape[1] / 2, bg.shape[0] / 2 + 80, 1.5)
        nimg, na, pen = note_image(p)
        sc = lerp(0.66, 0.72, u_of(t, t0, t1)); ang = -0.05
        nh, nw = na.shape; c, s = math.cos(ang) * sc, math.sin(ang) * sc
        M = np.float32([[c, -s, 0], [s, c, 0]]); M[:, 2] = np.float32([W / 2, H * 0.44]) - M[:, :2] @ np.float32([nw / 2, nh / 2])
        ni = cv2.warpAffine(nimg, M, (W, H)); nal = cv2.warpAffine(na, M, (W, H))
        sh = cv2.GaussianBlur(np.roll(nal, 22, 0), (0, 0), 18) * 0.7
        out = out * (1 - sh[..., None]); out = out * (1 - nal[..., None]) + ni * nal[..., None]
        if pen is not None and p < 1:
            px, py = M[0, 0] * pen[0] + M[0, 1] * pen[1] + M[0, 2], M[1, 0] * pen[0] + M[1, 1] * pen[1] + M[1, 2]
            lay = np.zeros((H, W), np.float32); hi = np.zeros((H, W), np.float32)
            d = np.array([0.62, -0.78]); nrm = np.array([0.78, 0.62]); tip = np.array([px, py])
            def q(l, w): return tip + d * l + nrm * w
            nib = [q(0, 0), q(46, -7), q(70, -9), q(70, 9), q(46, 7)]
            body = [q(66, -11), q(300, -15), q(310, 0), q(300, 15), q(66, 11)]
            for poly, v in ((nib, 0.75), (body, 1.0)):
                cv2.fillPoly(lay, [np.int32(np.array(poly) * 4)], v, cv2.LINE_AA, shift=2)
            cv2.line(hi, tuple(np.int32(q(80, -5))), tuple(np.int32(q(290, -8))), 1.0, 3, cv2.LINE_AA)
            sh2 = cv2.GaussianBlur(np.roll(np.roll(lay, 34, 1), 30, 0), (0, 0), 12) * 0.45
            out = out * (1 - sh2[..., None]); out = out * (1 - lay[..., None]) + np.array([34, 30, 28], np.float32) * lay[..., None]
            out = out + hi[..., None] * 70
        return out
    add(t0, t1, s_note, tr=0.45, seed=20)
    # 11 — the bullet (P05 tight) then the note in the mud
    tb = P0('treb3') + 0.25
    t0, t1 = P0('treb3') - 0.05, P0('treb3') + 1.0
    def s_hit(t, fi, t0=t0, t1=t1):
        img = plate('P05_meuse').render(t, fi, (975, 1700, lerp(2.2, 2.35, u_of(t, t0, t1))), (0, 0, 0), fog=0.3, haze=0.2)
        d = t - tb
        if d >= 0: img = flash(img, 0.85 * math.exp(-d * 9))
        return img
    add(t0, t1, s_hit, tr=0.0, kind='cut')
    t0, t1 = t1, P0('price1') - 0.1
    s_mud = plate_shot('P06_mud', (768, 1450, 1.85), (768, 1500, 2.1), (0, 0, -30), (0, 0, 30), t0, t1 + 0.4, haze=0.05)
    def s_mud2(t, fi, t0=t0, t1=t1):
        img = s_mud(t, fi)
        nimg, na, _ = note_image(1.0)
        u = ease2(u_of(t, t0, t1 + 0.4)); sc = lerp(0.36, 0.46, u); ang = 0.22
        nh, nw = na.shape; c, s = math.cos(ang) * sc, math.sin(ang) * sc
        M = np.float32([[c, -s, 0], [s, c, 0]]); M[:, 2] = np.float32([W * 0.5, H * 0.5]) - M[:, :2] @ np.float32([nw / 2, nh / 2])
        dirt = (0.62 + 0.38 * cv2.resize(fbm(nh // 4, nw // 4, 20, 81, 3), (nw, nh)))[..., None]
        ni = cv2.warpAffine(nimg * dirt, M, (W, H)); nal = cv2.warpAffine(na, M, (W, H)) * 0.96
        out = img * (1 - nal[..., None]) + ni * nal[..., None]
        return RAIN.apply(out, t, 1.0)
    add(t0, t1, s_mud2, tr=0.6, seed=21)
    # 12 — map: Ville-sur-Haine 10:58
    t0, t1 = t1, cue('price1', 'καναδ', frac=0.3) - 0.1
    def s_map2(t, fi, t0=t0, t1=t1):
        u = ease2(u_of(t, t0, t1))
        cam = (lerp(4.84, 4.065, u), lerp(49.70, 50.40, u), 2.5 - 0.9 * math.sin(math.pi * u))
        img = map_frame(t, fi, cam, 1.0, marks=[('vrigne', 0.6, None), ('ville', u_of(t, t0 + 0.7, t0 + 1.3), None)])
        return time_chip(img, fade_win(t, t0 + 0.35, t1 + 0.6, 0.3, 0.3), '10:58', 'ΒΕΛΓΙΟ · VILLE-SUR-HAINE')
    add(t0, t1, s_map2, tr=0.5, seed=22)
    # 13 — the village door
    t0, t1 = t1, P0('price2') - 0.05
    s_vil = plate_shot('P07_village', (760, 1450, 1.45), (600, 1500, 1.75), (-25, 0, 0), (25, 0, 0), t0, t1 + 0.4, fog=0.15, haze=0.15)
    def s_vil2(t, fi, t0=t0, t1=t1):
        return name_chip(s_vil(t, fi), fade_win(t, t0 + 0.5, t1 + 0.3, 0.4, 0.4), 'Τζορτζ Λόρενς Πράις', '25 χρονών · Καναδάς')
    add(t0, t1, s_vil2, tr=0.5, seed=23)
    # 14 — sniper POV and the shot
    t0, t1 = t1, P0('gun1') - 0.1
    tshot = cue('price2', 'πυροβολ', frac=0.7)
    s_sn = plate_shot('P08_sniper_b', (790, 1430, 1.3), (815, 1360, 1.5), (0, 0, -40), (0, 0, 40), t0, t1 + 0.4, haze=0.15)
    def s_sniper(t, fi, t0=t0, t1=t1):
        img = s_sn(t, fi); d = t - tshot
        if d >= 0:
            img = flash(img, 0.6 * math.exp(-d * 12))
            img = glow_at(img, 700, 820, 160, 1.4 * math.exp(-d * 14), (190, 235, 255))
        return img
    add(t0, t1, s_sniper, tr=0.45, seed=24)
    # 15 — map: Chaumont 10:59
    t0, t1 = t1, cue('gun1', 'αμερικ', frac=0.3) - 0.1
    def s_map3(t, fi, t0=t0, t1=t1):
        u = ease2(u_of(t, t0, t1))
        cam = (lerp(4.065, 5.42, u), lerp(50.40, 49.31, u), 2.5 - 1.0 * math.sin(math.pi * u))
        img = map_frame(t, fi, cam, 1.0, marks=[('vrigne', 0.6, None), ('ville', 0.6, None), ('chaumont', u_of(t, t0 + 0.7, t0 + 1.3), None)])
        return time_chip(img, fade_win(t, t0 + 0.35, t1 + 0.6, 0.3, 0.3), '10:59', 'ΓΑΛΛΙΑ · CHAUMONT-DEVANT-DAMVILLERS')
    add(t0, t1, s_map3, tr=0.5, seed=25)
    # 16 — watch 10:59 + name
    t0, t1 = t1, P0('gun2') - 0.05
    def s_w1059(t, fi, t0=t0, t1=t1):
        img = watch_scene(t, fi, 10 * 3600 + 59 * 60 + 2 + (t - t0), cam=(0.5, 0.45, lerp(1.25, 1.45, u_of(t, t0, t1)), 0.02))
        return name_chip(img, fade_win(t, t0 + 0.3, t1 + 0.3, 0.4, 0.4), 'Χένρι Γκάνθερ', '23 χρονών · Βαλτιμόρη, ΗΠΑ')
    add(t0, t1, s_w1059, tr=0.45, seed=26)
    # 17 — the letter / demotion
    t0, t1 = t1, P0('gun3') - 0.05
    tdem = cue('gun2', 'υποβιβ', frac=0.35)
    s_let = plate_shot('P14_letter', (900, 1380, 1.45), (960, 1420, 1.8), (0, 0, -40), (0, 0, 40), t0, t1 + 0.4)
    def s_letter(t, fi, t0=t0, t1=t1):
        img = s_let(t, fi)
        a = fade_win(t, tdem, t1 + 0.4, 0.3, 0.3)
        if a > 0:
            img = band(img, 230, 450, 0.5 * a, 80)
            al = text_layer(['ΛΟΧΙΑΣ'], 'type', 96, None)
            img = paste_alpha(img, al, W / 2, 290, (236, 241, 245), a, shadow=1)
            # strike
            sp = ease(u_of(t, tdem + 0.35, tdem + 0.75))
            if sp > 0:
                cv2.line(img, (int(W / 2 - 230), 300), (int(W / 2 - 230 + 460 * sp), 284), tuple(float(v) for v in RED * 1.5), 12, cv2.LINE_AA)
            a2 = clamp01((t - tdem - 0.8) / 0.4) * a
            al2 = text_layer(['ΣΤΡΑΤΙΩΤΗΣ'], 'type', 96, None)
            img = paste_alpha(img, al2, W / 2, 400, (120, 125, 235), a2, shadow=1)
        return img
    add(t0, t1, s_letter, tr=0.5, seed=27)
    # 18 — he gets up and runs
    t0, t1 = t1, P0('gun4') - 0.05
    s_rn = plate_shot('P10_runner', (740, 1450, 1.4), (705, 1330, 1.7), (0, 0, -60), (0, 0, 60), t0, t1 + 0.4, fog=0.3, haze=0.15)
    add(t0, t1, lambda t, fi: RAIN2.apply(s_rn(t, fi), t, 0.7), tr=0.4, seed=28)
    # 19 — Germans wave him back
    t0, t1 = t1, cue('gun4', 'πόλεμ', frac=0.55) - 0.1
    add(t0, t1, plate_shot('P09_mgnest', (800, 1400, 1.4), (835, 1260, 1.75), (-25, 0, -30), (25, 0, 30), t0, t1 + 0.4, fog=0.3, haze=0.2), tr=0.35, seed=29)
    # 20 — seconds hand 10:59:5x (macro on sub-dial)
    t0, t1 = t1, P0('gun5') - 0.05
    def subdial_cam(sc):
        z = W * 0.92 * sc / (2 * (WR + 60))
        return 0.5 - (WR * 0.47 * z) / H
    def s_secs(t, fi, t0=t0, t1=t1):
        sc = lerp(2.4, 2.7, u_of(t, t0, t1))
        return watch_scene(t, fi, 10 * 3600 + 59 * 60 + 48 + (t - t0), cam=(0.5, subdial_cam(sc), sc, 0.0))
    add(t0, t1, s_secs, tr=0.0, kind='cut')
    # 21 — he keeps going ... the shot
    t0, t1 = t1, P0('gun7') - 0.1
    tgs = cue('gun6', 'πυροβ', frac=0.4)
    def s_last(t, fi, t0=t0, t1=t1):
        img = plate('P10_runner').render(t, fi, (700, lerp(1180, 1100, u_of(t, t0, tgs)), lerp(2.0, 2.4, ease2(u_of(t, t0, tgs)))), (0, 0, 0), fog=0.25)
        img = RAIN2.apply(img, t, 0.7)
        d = t - tgs
        if d >= 0:
            img = flash(img, clamp01(1 - d * 4))
            img = img * clamp01(1 - (d - 0.12) * 5) if d > 0.12 else img
        return img
    add(t0, t1, s_last, tr=0.0, kind='cut')
    # 22 — black: 10:59
    t0, t1 = t1, P1('gun7') + 0.5
    def s_black(t, fi, t0=t0, t1=t1):
        img = np.full((H, W, 3), 14, np.float32)
        a = fade_win(t, t0 + 0.2, t1 + 0.4, 0.8, 0.5)
        al = text_layer(['10:59'], 'type', 230, None)
        img = paste_alpha(img, al, W / 2, H * 0.40, (60, 62, 170), a * 0.9)
        al2 = text_layer(['Χένρι Γκάνθερ'], 'serif', 60, 500)
        return paste_alpha(img, al2, W / 2, H * 0.40 + 170, (200, 206, 212), a * 0.8)
    add(t0, t1, s_black, tr=0.0, kind='cut')
    # 23 — 11:00:00 — the watch stops
    t11 = P0('eleven') + 0.02
    t0, t1 = t1, cue('eleven', 'όπλα', 'οπλα', frac=0.45) - 0.25
    def s_eleven(t, fi, t0=t0, t1=t1):
        u = ease2(u_of(t, t11 - 0.3, t1))
        sc = lerp(2.6, 1.05, u)
        v = lerp(subdial_cam(2.6), 0.47, u)
        clock = 11 * 3600 + min(0, t - t11)
        return watch_scene(t, fi, clock, cam=(0.5, v, sc, 0.0), stopped=t >= t11)
    add(t0, t1, s_eleven, tr=0.6, kind='fade')
    # 24 — the bugler, silence
    t0, t1 = t1, P0('grave1') - 0.1
    add(t0, t1, plate_shot('P11_bugler_b', (720, 1420, 1.32), (690, 1330, 1.45), (-15, 0, 0), (15, 0, 0), t0, t1 + 0.5, fog=0.45, haze=0.25, fog_speed=(10, -1)), tr=1.0, kind='fade')
    # 25 — the crosses, the plaque, the lie
    t0, t1 = t1, P0('mons1') - 0.1
    tpl = cue('grave1', 'πέθανε', 'πεθανε', frac=0.45) - 0.2
    tst0, tst1 = P0('grave2'), P1('grave2') + 0.15
    t11w = tst1 + 0.05
    tout = P0('grave3') + 1.2
    s_cr = plate_shot('P12_crosses', (800, 1330, 1.4), (590, 1380, 1.6), (-20, 0, 0), (20, 0, 0), t0, tpl + 0.6, haze=0.1, fog=0.2)
    s_cr2 = plate_shot('P12_crosses', (580, 1380, 1.6), (768, 1380, 1.32), (20, 0, 0), (-20, 0, 0), tout - 0.4, t1 + 0.5, haze=0.1, fog=0.4)
    def s_grave(t, fi, t0=t0, t1=t1):
        bgimg = s_cr(t, fi) if t < tout else s_cr2(t, fi)
        a_in = clamp01((t - tpl) / 0.5) * clamp01((tout + 0.5 - t) / 0.5)
        if a_in <= 0:
            return bgimg
        dark = bgimg * (1 - 0.45 * a_in)
        return plaque_frame(t, fi, dark, a_in, u_of(t, tst0, tst1), u_of(t, t11w, t11w + 0.6))
    add(t0, t1, s_grave, tr=0.6, seed=30)
    # 26 — St Symphorien: two graves facing each other
    t0, t1 = t1, P1('end') + 0.6
    tp0, tp1 = P0('mons2'), P0('mons4')
    te = P0('end')
    def s_cem(t, fi, t0=t0, t1=t1):
        keys = [(t0, (768, 1450, 1.35)), (tp0 - 0.2, (768, 1540, 1.5)), (tp0 + 1.2, (520, 1640, 1.9)), (tp1 - 0.3, (540, 1650, 1.95)),
                (tp1 + 1.1, (1030, 1600, 1.9)), (te - 0.3, (1020, 1610, 1.95)), (te + 3.0, (768, 1460, 1.32))]
        for (ka, ca), (kb, cb) in zip(keys, keys[1:]):
            if t <= kb:
                cam = lerp3(ca, cb, ease2(u_of(t, ka, kb))); break
        else:
            cam = keys[-1][1]
        px = 25 * math.sin((t - t0) * 0.25)
        img = plate('P13_cemetery').render(t, fi, cam, (px, 0, 0), haze=0.1, fog=0.22, fog_speed=(8, -1))
        img = MOTES.apply(img, t, 0.8)
        img = name_chip(img, fade_win(t, tp0 + 1.3, tp1 - 0.2, 0.4, 0.4), 'Τζον Παρ', 'ο πρώτος Βρετανός νεκρός · † 21 Αυγούστου 1914')
        img = name_chip(img, fade_win(t, tp1 + 1.2, te - 0.1, 0.4, 0.4), 'Τζορτζ Έλισον', 'ο τελευταίος Βρετανός νεκρός · † 11 Νοεμβρίου 1918, 9:30 π.μ.')
        return img
    add(t0, t1, s_cem, tr=0.6, seed=31)
    # 27 — end card
    t0, t1 = t1, T
    def s_end(t, fi, t0=t0, t1=t1):
        img = watch_scene(t, fi, 11 * 3600, cam=(0.5, 0.40, lerp(0.62, 0.66, u_of(t, t0, t1)), 0.0), stopped=True, light=0.8)
        a = fade_win(t, t0 + 0.4, t1 + 5, 0.8, 0.5)
        al = text_layer(['ΕΝΑ ΛΕΠΤΟ', 'ΠΡΙΝ ΤΗΝ ΕΙΡΗΝΗ'], 'serif', 86, 650, 1.1, track=8)
        img = paste_alpha(img, al, W / 2, H * 0.70, (236, 241, 245), a, shadow=1)
        al2 = text_layer(['11 · 11 · 1918'], 'type', 52, None)
        img = paste_alpha(img, al2, W / 2, H * 0.70 + 160, (200, 206, 212), a * 0.9, shadow=1)
        al3 = text_layer(['Mind Twist GR'], 'serif', 40, 500, track=4)
        img = paste_alpha(img, al3, W / 2, H * 0.93, (170, 176, 182), a * 0.8)
        return img * clamp01((T - t) / 0.9)
    add(t0, t1, s_end, tr=0.9, kind='fade')

build()

def frame(t, fi):
    idx = max(i for i, s in enumerate(SH) if s['t0'] <= t + 1e-6) if t >= SH[0]['t0'] else 0
    s = SH[idx]
    img = s['fn'](t, fi)
    if idx > 0 and s['tr'] > 0 and t < s['t0'] + s['tr']:
        prev = SH[idx - 1]['fn'](t, fi)
        p = (t - s['t0']) / s['tr']
        if s['kind'] == 'fade':
            img = crossfade(prev, img, ease2(p))
        else:
            m = ink_mask(ease2(p), seed=s['seed'], center=s['center'])
            img = prev * (1 - m) + img * m
    img = np.clip(img, 0, 255)
    img = subtitles(img, t)
    img = FILM.apply(img, fi, t, strength=1.0)
    return np.clip(img, 0, 255).astype(np.uint8)

if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'stills':
        os.makedirs(f'{ROOT}/test', exist_ok=True)
        for a in sys.argv[2:]:
            t = float(a); im = frame(t, int(t * FPS))
            cv2.imwrite(f'{ROOT}/test/still_{a}.jpg', im, [cv2.IMWRITE_JPEG_QUALITY, 88])
            print('still', a, flush=True)
    elif mode == 'shots':
        for s in SH: print(f"{s['t0']:7.2f} {s['t1']:7.2f} {s['kind']} {s['fn'].__name__}")
    elif mode == 'video':
        f0, f1, out = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
        tmp = out.replace('.mp4', '.part.mp4')
        p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                              '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '17', '-tune', 'grain', '-pix_fmt', 'yuv420p', tmp], stdin=subprocess.PIPE)
        for fi in range(f0, f1):
            p.stdin.write(frame(fi / FPS, fi).tobytes())
            if fi % 60 == 0: print(out, fi, flush=True)
        p.stdin.close(); p.wait()
        if p.returncode == 0: os.rename(tmp, out)
