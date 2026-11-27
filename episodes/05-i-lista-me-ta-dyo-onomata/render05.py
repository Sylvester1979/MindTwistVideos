"""Episode 5 renderer: «Η λίστα με τα δύο ονόματα». Data-driven: timeline.json (shots + subs) + MOVES below.
   python3 render05.py stills t1 t2 ...      -> test/still_<t>.jpg
   python3 render05.py video f0 f1 out.mp4    -> frames [f0,f1)
   python3 render05.py borderchk               -> margin check of every plate move
Plates: $MT_ROOT/plates/<k>.png (prep05.py), depth in $MT_ROOT/depth/<k>.npy (optional).
3D clips (ink style, «on twos»): $MT_ROOT/clips/<B|C|D>/stills_<X>_<NNN>.png, odd frames 1..L-1 (render in the Blender sandbox).
Graphics drawn here: G02 Ionian map, G09 the demanded list (blank), G11 clock, G18 the orders, G26 Corfu, G29 Greece,
G36 the stone plaque, G37 the two names typed (mirror of ep 04), the «;» on the paper (C3D_still)."""
import sys, os, json, subprocess, functools, math
import numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'engine'))
from eng import *
from eng import _ink_noise, FONT
FONT['mono'] = '/usr/share/fonts/truetype/freefont/FreeMono.ttf'      # a typewriter face that has Greek

TL = json.load(open(f'{ROOT}/timeline.json'))
T, SHOTS, SUBS = TL['T'], TL['shots'], TL['subs']
XF = 0.6                       # crossfade between shots (s)
FILM, MOTES = Film(seed=9), Motes(n=50, seed=4)
INKC = np.array([40, 44, 50], np.float32)
CREAMC = np.array([222, 236, 244], np.float32)
WHITE = (236, 242, 246)

# camera: (cx, cy, zoom) in plate fractions, start -> end; par = (px, py, pz) at the end
D = dict(par=(6, 0, 60), motes=0.0, flick=0.0)
MOVES = {
    'S03': dict(c=((0.42, 0.52, 1.20), (0.58, 0.52, 1.20)), par=(12, 0, 30)),
    'S04': dict(c=((0.50, 0.50, 1.05), (0.50, 0.56, 1.30))),
    'S05': dict(c=((0.50, 0.50, 1.05), (0.50, 0.45, 1.22)), par=(0, 0, 90)),
    'S06': dict(c=((0.50, 0.50, 1.32), (0.50, 0.50, 1.05)), par=(0, 0, -50)),
    'S07': dict(c=((0.50, 0.55, 1.05), (0.50, 0.42, 1.30))),
    'S08': dict(c=((0.50, 0.50, 1.05), (0.52, 0.48, 1.28)), motes=0.3),
    'S10': dict(c=((0.50, 0.50, 1.10), (0.50, 0.50, 1.18)), par=(0, 0, 20)),
    'S12': dict(c=((0.50, 0.55, 1.05), (0.50, 0.55, 1.30)), motes=0.4),
    'S13': dict(c=((0.50, 0.58, 1.25), (0.50, 0.45, 1.08)), par=(0, 0, -60)),
    'S14': dict(c=((0.50, 0.50, 1.05), (0.52, 0.45, 1.28)), motes=0.3, flick=0.4),
    'S16': dict(c=((0.50, 0.55, 1.10), (0.50, 0.50, 1.30)), par=(0, 0, 80)),
    'S17': dict(c=((0.50, 0.50, 1.05), (0.50, 0.48, 1.28))),
    'S20': dict(c=((0.50, 0.50, 1.05), (0.50, 0.46, 1.22)), par=(0, 0, 60)),
    'S21': dict(c=((0.50, 0.50, 1.10), (0.50, 0.50, 1.17))),
    'S22': dict(c=((0.42, 0.50, 1.20), (0.58, 0.50, 1.20)), par=(12, 0, 30)),
    'S23': dict(c=((0.50, 0.50, 1.05), (0.50, 0.52, 1.28)), flick=0.5, motes=0.4),
    'S24': dict(c=((0.58, 0.50, 1.20), (0.42, 0.50, 1.20)), par=(-12, 0, 30)),
    'S25': dict(c=((0.50, 0.50, 1.10), (0.50, 0.50, 1.17))),
    'S27': dict(c=((0.50, 0.50, 1.05), (0.50, 0.56, 1.28)), par=(0, 0, 80)),
    'S28': dict(c=((0.50, 0.50, 1.30), (0.50, 0.50, 1.05)), par=(0, 0, -50)),
    'S30': dict(c=((0.50, 0.50, 1.05), (0.50, 0.52, 1.20)), par=(0, 0, 60)),
    'S33': dict(c=((0.50, 0.50, 1.30), (0.50, 0.50, 1.05)), par=(0, 0, -50)),
    'S34': dict(c=((0.50, 0.50, 1.05), (0.50, 0.52, 1.28)), motes=0.3),
    'S35': dict(c=((0.50, 0.50, 1.35), (0.50, 0.50, 1.05)), par=(0, 0, -60)),
}
# text cards: shot -> (lines, size, y, t_in, t_out) times relative to shot start (t_out < 0: from shot end)
CARDS = {
    'G02_map': (['Ζάκυνθος · Σεπτέμβριος 1943'], 62, 0.14, 0.3, -0.3),
    'S05': (['275'], 96, 0.14, 0.4, -0.3),
    'S06': (['Θεσσαλονίκη · 1943'], 66, 0.14, 0.2, -0.3),
    'S08': (['Λουκάς Καρρέρ · δήμαρχος'], 62, 0.14, 0.6, -0.3),
    'S14': (['Μητροπολίτης Χρυσόστομος'], 62, 0.14, 0.4, -0.3),
    'G26_corfu': (['Κέρκυρα · Ιούνιος 1944'], 64, 0.14, 0.2, -0.3),
    'S27': (['Αύγουστος 1944'], 66, 0.14, 0.2, -0.3),
    'G29_greece': (['πάνω από 8 στους 10'], 66, 0.14, 0.6, -0.3),
    'S30': (['275', 'κανείς δεν παραδόθηκε'], 62, 0.14, 0.3, -0.3),
    'G36_plaque': (['Δίκαιοι των Εθνών · 1978'], 62, 0.14, 0.8, -0.4),
}


# ---------------------------------------------------------------- plates
@functools.lru_cache(None)
def plate(k):
    return Plate(k)


@functools.lru_cache(None)
def content_box(key):
    """the drawing inside its blank paper margin: rows/cols from each edge that are >90% blank paper, +14 px."""
    pm = cv2.imread(f'{ROOT}/plates/{key}_paper.png', 0) > 127
    Hh, Ww = pm.shape

    def run(v):
        n = 0
        for x in v:
            if x <= 0.9: break
            n += 1
        return n + 14 if n > 4 else n
    cf, rf = pm.mean(0), pm.mean(1)
    return run(cf), run(rf), Ww - run(cf[::-1]), Hh - run(rf[::-1])


def keep_inside(P, cx, cy, z, px, py, pz, focus=0.55):
    """never let the camera (incl. parallax/dolly) see past the drawing: zoom in just enough, then clamp the centre."""
    x0, y0, x1, y1 = content_box(P.key)
    dmax = max(focus, 1 - focus)
    mxp = dmax * (abs(px) + abs(pz) * 0.5 / z) + 4
    myp = dmax * (abs(py) + abs(pz) * (H / W) * 0.5 / z) + 4
    hw = lambda zz: P.pw / (2 * zz)
    hh = lambda zz: H / 2 * P.pw / W / zz
    z = max(z, P.pw / max(1.0, (x1 - x0) - 2 * mxp), (H * P.pw / W) / max(1.0, (y1 - y0) - 2 * myp))
    cx = min(max(cx, x0 + hw(z) + mxp), x1 - hw(z) - mxp)
    cy = min(max(cy, y0 + hh(z) + myp), y1 - hh(z) - myp)
    return cx, cy, z


def shot_u(s, t):
    return ease(clamp01((t - s['t0'] + XF / 2) / (s['t1'] - s['t0'] + XF)))


def shot_frame(s, t, fi):
    k = s['k']
    if k in SPECIAL:
        img = SPECIAL[k](s, t, fi)
    else:
        mv = {**D, **MOVES.get(k, {})}
        (a0, b0, z0), (a1, b1, z1) = mv.get('c', ((0.5, 0.5, 1.05), (0.5, 0.5, 1.25)))
        u = shot_u(s, t)
        P = plate(k)
        px, py, pz = mv['par']
        a0, b0, z0 = keep_inside(P, a0 * P.pw, b0 * P.ph, z0, 0, 0, 0)
        a1, b1, z1 = keep_inside(P, a1 * P.pw, b1 * P.ph, z1, px, py, pz)
        cx, cy, z = lerp(a0, a1, u), lerp(b0, b1, u), lerp(z0, z1, u)
        cx, cy, z = keep_inside(P, cx, cy, z, px * u, py * u, pz * u)       # safety net only
        img = P.render(t, fi, (cx, cy, z), par=(px * u, py * u, pz * u), focus=0.55, boil=0.0)
        if mv['flick']:
            img = img * (1 + 0.035 * mv['flick'] * (np.sin(t * 9.1) + 0.6 * np.sin(t * 15.7 + 1)))
        if mv['motes']:
            img = MOTES.apply(img, t, mv['motes'])
    if k in CARDS:
        img = card(img, s, t, *CARDS[k])
    return img


def card(img, s, t, lines, size, y, ti, to):
    t_out = s['t1'] + to if to < 0 else s['t0'] + to
    a = clamp01((t - s['t0'] - ti) / 0.6) * clamp01((t_out - t) / 0.6)
    if a <= 0: return img
    al = text_layer(lines, 'serif', size, 560, 1.25)
    cy = int(H * y) + (al.shape[0] // 2 - 40 if len(lines) > 1 else 0)
    img = band(img, cy - al.shape[0] // 2 - 6, cy + al.shape[0] // 2 + 6, 0.45 * a, 90)
    return paste_alpha(img, al, W / 2, cy, WHITE, a, shadow=1.0, rough=True)


# ---------------------------------------------------------------- the 3D clips (ink, on twos)
CLIP_LEN = {'B': 210, 'C': 418, 'D': 240}     # C: 13.93″ since 26/11 (was 12″, slowed down)
CLIP_GAIN = PAPER / np.array([232, 247, 255], np.float32)        # Blender ink paper -> the plates' paper


@functools.lru_cache(maxsize=6)
def clip_img(c, f):
    im = cv2.imread(f'{ROOT}/clips/{c}/stills_{c}_{f:03d}.png')
    if im is None: raise FileNotFoundError(f'{c} {f}')
    return im.astype(np.float32) * CLIP_GAIN


def clip_at(c, tau):
    """clip time tau (s) -> its frame. All 30 frames/s are rendered (the user, 26/11: no frames left out)."""
    n = CLIP_LEN[c]
    f = 1 + int(min(max(tau, 0.0), n / 30 - 1e-3) * 30)
    return clip_img(c, min(f, n))


def zoom_img(img, z, fx=0.5, fy=0.5):
    if z <= 1.0001: return img
    M = np.float32([[z, 0, W * fx * (1 - z)], [0, z, H * fy * (1 - z)]])
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def hook_frame(s, t, fi):          # the last part of C, at normal speed, ending with the hook
    dur = s['t1'] - s['t0'] + XF / 2
    return clip_at('C', CLIP_LEN['C'] / 30 - dur + (t - s['t0']))


def burn_frame(s, t, fi):          # B: the fire from «να κάψει» to the end of the clip
    dur = s['t1'] - s['t0'] + XF / 2
    return clip_at('B', 7.0 - dur + (t - s['t0']))


def paper_frame(s, t, fi):         # C at normal speed: made the length of the shot, so the names land on «δύο ονόματα»
    return clip_at('C', t - s['t0'])


def finale_frame(s, t, fi):
    return clip_at('D', t - s['t0'])


QMARK = (0.86, 0.34)               # where the «;» is drawn on the last frame of C (frame fractions)


def doubt_frame(s, t, fi):         # the paper, still; an ink «;» appears in the margin
    r = t - s['t0']; u = shot_u(s, t)
    img = zoom_img(clip_img('C', 417).copy(), lerp(1.0, 1.10, u), 0.5, 0.45)
    a = clamp01((r - 1.2) / 1.4)
    if a > 0:
        al = text_layer([';'], 'serif', 210, 500)
        img = paste_alpha(img, al, W * QMARK[0], H * QMARK[1], INKC, ease(a), shadow=0.0, rough=True)
    return img


def ash_frame(s, t, fi):           # the ash in the tray, still (a frame of D, before its camera rises)
    return zoom_img(clip_img('D', 81).copy(), lerp(1.0, 1.08, shot_u(s, t)), 0.55, 0.45)


# ---------------------------------------------------------------- maps (Natural Earth land, no modern borders)
class Map:
    def __init__(self, lon0, lon1, lat0, lat1, mw, labels, focus=None, seed=41):
        self.b = (lon0, lon1, lat0, lat1); self.mw = mw
        self.cosl = math.cos(math.radians((lat0 + lat1) / 2))
        self.mh = int(mw * (lat1 - lat0) / ((lon1 - lon0) * self.cosl))
        self.labels, self.focus, self.seed = labels, focus, seed

    def proj(self, lon, lat):
        lon0, lon1, lat0, lat1 = self.b
        return (np.asarray(lon) - lon0) / (lon1 - lon0) * self.mw, (lat1 - np.asarray(lat)) / (lat1 - lat0) * self.mh

    def polys(self):
        lon0, lon1, lat0, lat1 = self.b
        gj = json.load(open(f'{ROOT}/map/ne_10m_land.geojson')); out = []
        for ft in gj['features']:
            g = ft['geometry']
            if g is None: continue
            polys = {'Polygon': [g['coordinates']], 'MultiPolygon': g['coordinates']}.get(g['type'], [])
            for poly in polys:
                arr = np.array(poly[0], np.float64)
                if arr.ndim != 2 or len(arr) < 3: continue
                if arr[:, 0].max() < lon0 - 1 or arr[:, 0].min() > lon1 + 1 or arr[:, 1].max() < lat0 - 1 or arr[:, 1].min() > lat1 + 1:
                    continue
                out.append(arr)
        return out

    @functools.lru_cache(None)
    def static(self):
        MW, MH = self.mw, self.mh
        img = np.ones((MH, MW, 3), np.float32) * PAPER * 1.02 * (0.90 + 0.12 * fbm(MH, MW, 260, self.seed, 4))[..., None]
        land = np.zeros((MH, MW), np.uint8); focus = np.zeros((MH, MW), np.uint8); coast = []
        for arr in self.polys():
            x, y = self.proj(arr[:, 0], arr[:, 1]); pts = np.stack([x, y], 1)
            cv2.fillPoly(land, [np.round(pts * 4).astype(np.int32)], 255, cv2.LINE_AA, shift=2); coast.append(pts)
            if self.focus is not None:
                inside = cv2.pointPolygonTest(arr.astype(np.float32), self.focus, False) >= 0
                if inside: cv2.fillPoly(focus, [np.round(pts * 4).astype(np.int32)], 255, cv2.LINE_AA, shift=2)
        sea = 1 - land.astype(np.float32) / 255
        hatch = np.zeros((MH, MW), np.float32)
        for yy in range(0, MH + 60, 11): cv2.line(hatch, (0, yy), (MW, yy - 40), 1.0, 1, cv2.LINE_AA)
        dist = cv2.distanceTransform((sea > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
        rip = sum(np.clip(1 - np.abs(dist - k) / 1.6, 0, 1) for k in (10, 22, 38)) * sea
        tone_ = np.clip(sea * 0.10 + hatch * sea * 0.35 + rip * 0.45, 0, 1)
        img = img * (1 - tone_[..., None] * 0.7) + INK * tone_[..., None] * 0.7
        lay = np.zeros((MH, MW), np.float32); rng = np.random.default_rng(3)
        for pts in coast:
            cv2.polylines(lay, [np.round((pts + rng.normal(0, 0.6, pts.shape)) * 4).astype(np.int32)], True, 1.0, 4, cv2.LINE_AA, shift=2)
        lay = np.clip(lay, 0, 1) * (0.8 + 0.2 * fbm(MH, MW, 6, 44, 2))
        img = img * (1 - lay[..., None]) + INK * lay[..., None]
        self.focus_mask = focus.astype(np.float32) / 255
        self.land_mask = land.astype(np.float32) / 255
        from PIL import Image, ImageDraw
        pil = Image.new('L', (MW, MH), 0); d = ImageDraw.Draw(pil)
        for txt, lon, lat, size, track, dy in self.labels:
            x, y = self.proj(lon, lat); f = font('serif', size, 600)
            tw = sum(f.getlength(ch) + track for ch in txt) - track
            xx = x - tw / 2
            for ch in txt:
                d.text((xx, y - size / 2 + dy), ch, font=f, fill=255); xx += f.getlength(ch) + track
        la = np.asarray(pil).astype(np.float32) / 255
        img = img * (1 - la[..., None] * 0.9) + INK * la[..., None] * 0.9
        return img

    def view(self, img, cx, cy, zoom):
        s = W * zoom / self.mw
        M = np.float32([[s, 0, W / 2 - cx * s], [0, s, H / 2 - cy * s]])
        return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), M


ZAK = (20.85, 37.78)                     # Zakynthos (island centre)
MAP_ION = Map(18.6, 23.4, 36.3, 40.6, 3000, [
    ('ΙΟΝΙΟ ΠΕΛΑΓΟΣ', 19.6, 37.6, 70, 18, 0), ('ΚΕΡΚΥΡΑ', 19.85, 39.62, 52, 8, -70), ('ΚΕΦΑΛΟΝΙΑ', 20.55, 38.22, 52, 8, -90),
    ('ΠΕΛΟΠΟΝΝΗΣΟΣ', 22.2, 37.45, 66, 16, 0), ('ΖΑΚΥΝΘΟΣ', 20.85, 37.52, 64, 12, 40)], focus=ZAK)
COR = (19.85, 39.6)
MAP_COR = Map(18.9, 21.2, 38.9, 40.5, 2600, [('ΚΕΡΚΥΡΑ', 19.93, 39.42, 64, 12, 0), ('ΗΠΕΙΡΟΣ', 20.75, 39.85, 60, 14, 0),
                                             ('ΙΟΝΙΟ ΠΕΛΑΓΟΣ', 19.45, 39.15, 52, 12, 0)], focus=COR, seed=42)
MAP_GR = Map(19.0, 28.6, 34.6, 42.0, 2600, [('ΕΛΛΑΔΑ', 22.6, 39.4, 90, 26, 0)], seed=43)


def map_ion_frame(s, t, fi):
    r = t - s['t0']; u = shot_u(s, t)
    m = MAP_ION; base = m.static()
    x0, y0 = m.proj(20.6, 38.4); x1, y1 = m.proj(*ZAK)
    lit = ease(clamp01((r - 0.6) / 1.4))
    if lit > 0:                                                        # the island comes up bright, outlined in ink
        fm = m.focus_mask
        ring = cv2.dilate(fm, np.ones((9, 9), np.uint8)) - fm
        base = base * (1 - (fm * 0.35 * lit)[..., None]) + PAPER * 1.08 * (fm * 0.35 * lit)[..., None]
        base = base * (1 - (ring * lit)[..., None]) + INK * (ring * lit)[..., None]
    out, _ = m.view(base, lerp(x0, x1, u), lerp(y0, y1, u), lerp(1.6, 2.6, ease2(u)))
    return out * vignette()


def map_cor_frame(s, t, fi):
    r = t - s['t0']; dur = s['t1'] - s['t0']; u = shot_u(s, t)
    m = MAP_COR; base = m.static().copy()
    x, y = m.proj(*COR)
    p = clamp01((r - 1.5) / max(1.0, dur - 2.5))
    if p > 0:                                                         # a dashed ink line leaves the island northwards
        lay = np.zeros(base.shape[:2], np.float32)
        x2, y2 = m.proj(19.95, 40.7); n = 24
        for i in range(int(n * ease(p))):
            if i % 2: continue
            a, b = i / n, (i + 1) / n
            cv2.line(lay, (int(lerp(x, x2, a)), int(lerp(y - 60, y2, a))), (int(lerp(x, x2, b)), int(lerp(y - 60, y2, b))), 1.0, 14, cv2.LINE_AA)
        base = base * (1 - lay[..., None] * 0.85) + INK * lay[..., None] * 0.85
    out, _ = m.view(base, x + 120, lerp(y + 80, y - 60, u), lerp(2.0, 2.3, u))
    out = out * vignette()
    v = int(round(1800 * ease2(clamp01((r - 1.6) / max(0.5, dur * 0.55)))))
    return counter_overlay(out, v, clamp01((r - 1.4) / 0.4), y=0.52, dark=0.93)   # the dashed line passes under the band


def counter_overlay(out, value, a, y=0.47, dark=0.70):
    if a <= 0: return out
    s = f'{value:,}'.replace(',', '.')
    al = text_layer([s], 'serif', 230, 620)
    cy = int(H * y)
    out = band(out, cy - 150, cy + 150, dark * a, 120)
    return paste_alpha(out, al, W / 2, cy, WHITE, a, shadow=1.0, rough=True)


@functools.lru_cache(None)
def figure_alpha(h=150):
    """a small standing figure in ink (head, coat, legs): no face, no detail."""
    w = int(h * 0.42); a = np.zeros((h, w), np.float32)
    cv2.circle(a, (w // 2, int(h * 0.12)), int(h * 0.09), 1.0, -1, cv2.LINE_AA)
    body = np.array([[w * 0.18, h * 0.24], [w * 0.82, h * 0.24], [w * 0.92, h * 0.66], [w * 0.08, h * 0.66]], np.int32)
    cv2.fillPoly(a, [body], 1.0, cv2.LINE_AA)
    cv2.rectangle(a, (int(w * 0.24), int(h * 0.64)), (int(w * 0.44), int(h * 0.98)), 1.0, -1, cv2.LINE_AA)
    cv2.rectangle(a, (int(w * 0.56), int(h * 0.64)), (int(w * 0.76), int(h * 0.98)), 1.0, -1, cv2.LINE_AA)
    return cv2.GaussianBlur(a, (0, 0), 0.8)


def map_gr_frame(s, t, fi):
    r = t - s['t0']; dur = s['t1'] - s['t0']; u = shot_u(s, t)
    m = MAP_GR; base = m.static()
    x, y = m.proj(22.6, 38.8)
    out, _ = m.view(base, x, y, lerp(1.9, 2.0, u))
    out = out * vignette() * 0.92
    fa = figure_alpha(); gap = 92; x0 = W / 2 - 4.5 * gap; yy = H * 0.66
    out = band(out, int(yy - 110), int(yy + 110), 0.55, 80)
    fade = ease(clamp01((r - 1.2) / 1.6))
    for i in range(10):
        a = clamp01((r - 0.15 * i) / 0.4)
        if i < 8: a *= 1 - 0.85 * clamp01(fade * 1.6 - i * 0.08)     # eight of the ten go, one after another
        out = paste_alpha(out, fa, x0 + i * gap, yy, WHITE, a, shadow=0.8, rough=True)
    return out


# ---------------------------------------------------------------- typewriter pages
@functools.lru_cache(None)
def page_blank(w=1500, h=2300, seed=1):
    return np.ones((h, w, 3), np.float32) * PAPER * 1.04 * (0.9 + 0.1 * fbm(h, w, 70, 60 + seed, 4))[..., None]


@functools.lru_cache(None)
def paper_bg():
    return PAPER * 1.02 * paper_tex() * (0.92 + 0.08 * fbm(H, W, 220, 61, 3))[..., None]


def typed_text(lay, x, y, text, size, n_chars, seed):
    """typewriter text: first n_chars glyphs, each slightly offset, uneven ink. Returns the x after the last glyph."""
    from PIL import Image, ImageDraw
    f = font('mono', size); rng = np.random.default_rng(seed)
    im = Image.fromarray((lay * 255).astype(np.uint8)); d = ImageDraw.Draw(im)
    for i, ch in enumerate(text):
        a = int(205 + 50 * rng.random()); jx, jy = rng.normal(0, 0.7), rng.normal(0, 0.9)
        if i < n_chars: d.text((x + jx, y + jy), ch, font=f, fill=a)
        x += f.getlength(ch)
    lay[:] = np.asarray(im).astype(np.float32) / 255
    return x


def view_page(pg, fx, fy, zoom, bg_dark=0.55):
    h, w = pg.shape[:2]
    s = W * zoom / w
    bg = paper_bg().copy() * bg_dark
    M = np.float32([[s, 0, W / 2 - fx * w * s], [0, s, H / 2 - fy * h * s]])
    out = cv2.warpAffine(pg, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    m = cv2.warpAffine(np.ones((h, w), np.float32), M, (W, H), flags=cv2.INTER_LINEAR)[..., None]
    return bg * (1 - m) + out * m


HEAD = [('ΟΝΟΜΑ', 215), ('ΔΙΕΥΘΥΝΣΙΣ', 575), ('ΠΕΡΙΟΥΣΙΑ', 960)]


def demanded_frame(s, t, fi):
    """the list the commander wants: the three column heads are typed, the rows stay empty."""
    r = t - s['t0']; u = shot_u(s, t)
    w, h = 1500, 2300; pg = page_blank(w, h, 3).copy(); lay = np.zeros((h, w), np.float32)
    top = 300
    cv2.line(lay, (110, top + 40), (w - 110, top + 40), 0.8, 3, cv2.LINE_AA)
    for i in range(1, 22):                                             # ruled, empty rows
        cv2.line(lay, (110, top + 40 + i * 82), (w - 110, top + 40 + i * 82), 0.22, 2, cv2.LINE_AA)
    for xx in (550, 935):
        cv2.line(lay, (xx, top - 30), (xx, top + 40 + 21 * 82), 0.3, 2, cv2.LINE_AA)
    # one head per spoken word: «Ονόματα, διευθύνσεις, περιουσίες»
    words = [sb for sb in SUBS if sb['k'] == 'G09_page']
    for j, (txt, xx) in enumerate(HEAD):
        t_in = words[j]['t0'] if j < len(words) else s['t0'] + 0.4 + j * 0.8
        n = int(len(txt) * clamp01((t - t_in + 0.1) / 0.45))
        typed_text(lay, xx, top - 44, txt, 54, n, 20 + j)
    lay = cv2.GaussianBlur(lay, (0, 0), 0.9) * (0.8 + 0.2 * fbm(h, w, 5, 71, 2))
    pg = pg * (1 - lay[..., None]) + INKC * lay[..., None]
    return view_page(pg, 0.50, lerp(0.47, 0.49, u), lerp(1.16, 1.22, u)) * vignette()


NAMES = ['Ο Ζακύνθου Χρυσόστομος', 'Δήμαρχος Ζακυνθίων Λουκάς Καρρέρ']


def names_frame(s, t, fi):
    """mirror of ep 04: on a blank page the typewriter types the two names, as on the paper (Vitsos)."""
    r = t - s['t0']; dur = s['t1'] - s['t0']; u = shot_u(s, t)
    w, h = 1500, 2300; pg = page_blank(w, h, 5).copy(); lay = np.zeros((h, w), np.float32)
    total = sum(len(n) for n in NAMES); p = clamp01((r - 0.3) / max(1.0, dur * 0.72))
    k = int(total * p)
    for j, nm in enumerate(NAMES):
        typed_text(lay, 200, 760 + j * 115, nm, 56, max(0, min(len(nm), k)), 40 + j)
        k -= len(nm)
    lay = cv2.GaussianBlur(lay, (0, 0), 0.9) * (0.8 + 0.2 * fbm(h, w, 5, 72, 2))
    pg = pg * (1 - lay[..., None]) + INKC * lay[..., None]
    return view_page(pg, 0.50, lerp(0.47, 0.49, u), lerp(1.16, 1.22, u)) * vignette()


# ---------------------------------------------------------------- clock
@functools.lru_cache(None)
def clock_face(R=430):
    S = 2 * R + 80; img = np.ones((S, S, 3), np.float32) * INK * 1.5
    c = (S // 2, S // 2)
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32); d = np.sqrt((xx - c[0]) ** 2 + (yy - c[1]) ** 2)
    dial = (d < R * 0.90).astype(np.float32)
    img = img * (1 - dial[..., None]) + PAPER * (0.88 + 0.12 * fbm(S, S, 90, 81, 3))[..., None] * dial[..., None]
    ring = ((d > R * 0.90) & (d < R)).astype(np.float32)
    img = img * (1 - ring[..., None]) + INK * 2.2 * (0.8 + 0.4 * fbm(S, S, 20, 82, 2))[..., None] * ring[..., None]
    lay = np.zeros((S, S), np.float32)
    for i in range(60):
        a = math.radians(i * 6); r0 = R * (0.78 if i % 5 == 0 else 0.83)
        cv2.line(lay, (int(c[0] + r0 * math.sin(a)), int(c[1] - r0 * math.cos(a))),
                 (int(c[0] + R * 0.86 * math.sin(a)), int(c[1] - R * 0.86 * math.cos(a))), 1.0, 7 if i % 5 == 0 else 3, cv2.LINE_AA)
    from PIL import Image, ImageDraw
    pil = Image.fromarray((lay * 255).astype(np.uint8)); dd = ImageDraw.Draw(pil); f = font('serif', 84, 560)
    for hnum in range(1, 13):
        a = math.radians(hnum * 30); txt = str(hnum)
        tw = f.getlength(txt); x = c[0] + R * 0.64 * math.sin(a) - tw / 2; y = c[1] - R * 0.64 * math.cos(a) - 52
        dd.text((x, y), txt, font=f, fill=255)
    lay = np.asarray(pil).astype(np.float32) / 255
    img = img * (1 - lay[..., None] * 0.9) + INK * lay[..., None] * 0.9
    return img, c, R


def clock_frame(s, t, fi):
    r = t - s['t0']; dur = s['t1'] - s['t0']; u = shot_u(s, t)
    face, c, R = clock_face(); img = face.copy()
    p = ease2(clamp01(r / max(1.0, dur + XF)))
    hours = 18.0 + 15.0 * p                                            # evening -> the next day
    ha, ma = math.radians((hours % 12) * 30), math.radians((hours % 1) * 360)
    lay = np.zeros(img.shape[:2], np.float32)
    cv2.line(lay, c, (int(c[0] + R * 0.45 * math.sin(ha)), int(c[1] - R * 0.45 * math.cos(ha))), 1.0, 22, cv2.LINE_AA)
    cv2.line(lay, c, (int(c[0] + R * 0.70 * math.sin(ma)), int(c[1] - R * 0.70 * math.cos(ma))), 1.0, 12, cv2.LINE_AA)
    cv2.circle(lay, c, 26, 1.0, -1, cv2.LINE_AA)
    sh = np.roll(np.roll(cv2.GaussianBlur(lay, (0, 0), 6), 10, 0), 8, 1)
    img = img * (1 - 0.35 * sh[..., None]) * (1 - lay[..., None]) + INK * lay[..., None]
    out = np.ones((H, W, 3), np.float32) * INK * 1.4 * (0.85 + 0.3 * _ink_noise(9))[..., None]
    S = img.shape[0]; z = lerp(1.0, 1.12, u); sz = int(S * z * 0.98)
    im2 = cv2.resize(img, (sz, sz), interpolation=cv2.INTER_AREA)
    x0, y0 = W // 2 - sz // 2, int(H * 0.46) - sz // 2
    yy, xx = np.mgrid[0:sz, 0:sz].astype(np.float32); m = (np.sqrt((xx - sz / 2) ** 2 + (yy - sz / 2) ** 2) < sz / 2 * (2 * R + 0) / S).astype(np.float32)
    m = cv2.GaussianBlur(m, (0, 0), 1.5)[..., None]
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + sz), min(H, y0 + sz)
    out[ya:yb, xa:xb] = out[ya:yb, xa:xb] * (1 - m[ya - y0:yb - y0, xa - x0:xb - x0]) + im2[ya - y0:yb - y0, xa - x0:xb - x0] * m[ya - y0:yb - y0, xa - x0:xb - x0]
    return out * vignette()


# ---------------------------------------------------------------- the orders, again and again
@functools.lru_cache(None)
def order_sheet(seed, w=900, h=1270):
    """a typed order: illegible typed marks (no words), a header rule and an empty round stamp. No insignia."""
    rng = np.random.default_rng(seed)
    pg = np.ones((h, w, 3), np.float32) * PAPER * 1.03 * (0.9 + 0.1 * fbm(h, w, 60, 90 + seed, 3))[..., None]
    lay = np.zeros((h, w), np.float32)
    cv2.line(lay, (80, 190), (w - 80, 190), 0.7, 3, cv2.LINE_AA)
    for k in range(int(rng.integers(5, 9))):                            # heading marks
        x = 300 + k * 40; cv2.rectangle(lay, (x, 120), (x + 26, 150), float(0.6 + 0.3 * rng.random()), -1)
    y = 260
    while y < h - 260:
        x = 90; L = int(rng.integers(500, w - 200))
        while x < L:
            ww = int(rng.integers(40, 140))
            for j in range(ww // 17):
                hh = int(rng.integers(12, 20)); cv2.rectangle(lay, (x + j * 17, y - hh // 2), (x + j * 17 + 10, y + hh // 2), float(0.5 + 0.35 * rng.random()), -1)
            x += ww + 26
        y += 46
    cv2.circle(lay, (w - 230, h - 170), 85, 0.55, 5, cv2.LINE_AA); cv2.circle(lay, (w - 230, h - 170), 62, 0.4, 3, cv2.LINE_AA)
    lay = cv2.GaussianBlur(lay, (0, 0), 1.0) * (0.75 + 0.25 * fbm(h, w, 5, 95 + seed, 2))
    return pg * (1 - lay[..., None]) + INKC * lay[..., None]


def orders_frame(s, t, fi):
    """seen from above: order after order lands on the pile (on «επιμένει», «ζητά», «ξανά», «ξανά»)."""
    r = t - s['t0']; u = shot_u(s, t)
    out = np.ones((H, W, 3), np.float32) * INK * 2.0 * (0.8 + 0.4 * fbm(H, W, 40, 96, 3))[..., None]
    beats = [sb['t0'] for sb in SUBS if sb['k'] == 'G18_orders']
    drops = [s['t0'] + 0.1] + [b + 0.05 for b in beats[1:]] + [beats[-1] + 0.75] if beats else [s['t0'] + 0.3 * i for i in range(4)]
    rng = np.random.default_rng(5)
    for i, td in enumerate(drops):
        p = clamp01((t - td) / 0.35)
        if t < td - 0.01: break
        sh = order_sheet(i % 3 + 1)
        ang = rng.uniform(-9, 9); dx, dy = rng.uniform(-50, 50), rng.uniform(-60, 60)
        sc = lerp(1.35, 1.0, ease2(p)) * lerp(1.0, 1.06, u)
        h, w = sh.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc * 0.95)
        M[0, 2] += W / 2 - w / 2 + dx; M[1, 2] += H * 0.47 - h / 2 + dy
        lay = cv2.warpAffine(sh, M, (W, H), flags=cv2.INTER_LINEAR)
        m = cv2.warpAffine(np.ones((h, w), np.float32), M, (W, H), flags=cv2.INTER_LINEAR)
        shadow = cv2.GaussianBlur(np.roll(np.roll(m, int(14 + 40 * (1 - p)), 0), int(10 + 30 * (1 - p)), 1), (0, 0), 14 + 20 * (1 - p))
        out = out * (1 - 0.55 * shadow[..., None])
        a = (m * ease(clamp01(p * 3)))[..., None]
        out = out * (1 - a) + lay * a
    return out * vignette()


# ---------------------------------------------------------------- the stone plaque
PLAQUE = ['ΜΗΤΡΟΠΟΛΙΤΗΣ ΧΡΥΣΟΣΤΟΜΟΣ', 'ΔΗΜΑΡΧΟΣ ΛΟΥΚΑΣ ΚΑΡΡΕΡ']


@functools.lru_cache(None)
def plaque_img(w=1600, h=2400):
    n1, n2, n3 = fbm(h, w, 220, 101, 5), fbm(h, w, 30, 102, 3), fbm(h, w, 6, 103, 2)
    stone = PAPER * (0.55 + 0.25 * n1 + 0.10 * n2 + 0.06 * n3)[..., None]
    speck = (fbm(h, w, 3, 105, 1) > 0.78).astype(np.float32) * 0.10            # granite-like specks, no veins
    stone = stone * (1 - speck[..., None])
    for k in (60, 76):                                                            # a cut border around the inscription
        cv2.rectangle(stone, (k, k), (w - k, h - k), tuple(float(v) * 0.45 for v in PAPER), 4, cv2.LINE_AA)
    from PIL import Image, ImageDraw
    im = Image.new('L', (w, h), 0); d = ImageDraw.Draw(im)
    y = h * 0.40
    for i, ln in enumerate(PLAQUE):
        size = 84
        while font('serif', size, 600).getlength(ln) + 10 * len(ln) > w * 0.80: size -= 2
        f = font('serif', size, 600)
        tw = sum(f.getlength(c) + 10 for c in ln) - 10; x = (w - tw) / 2
        for c in ln:
            d.text((x, y), c, font=f, fill=255); x += f.getlength(c) + 10
        y += size * 1.9
    a = cv2.GaussianBlur(np.asarray(im).astype(np.float32) / 255, (0, 0), 0.9)
    hi = np.roll(np.roll(a, 3, 0), 3, 1)                                 # lit lower-right edge of the cut
    stone = stone * (1 - 0.70 * a[..., None]) + np.float32(28) * 0.70 * a[..., None]
    stone = stone * (1 - 0.30 * np.clip(hi - a, 0, 1)[..., None]) + PAPER * 0.30 * np.clip(hi - a, 0, 1)[..., None]
    return stone


def plaque_frame(s, t, fi):
    u = shot_u(s, t); pg = plaque_img(); h, w = pg.shape[:2]
    z = lerp(1.05, 1.32, u); sc = W * z / w
    M = np.float32([[sc, 0, W / 2 - 0.5 * w * sc], [0, sc, H / 2 - lerp(0.50, 0.46, u) * h * sc]])
    out = cv2.warpAffine(pg, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return out * vignette()


SPECIAL = {'C3D_hook': hook_frame, 'G02_map': map_ion_frame, 'G09_page': demanded_frame, 'G11_clock': clock_frame,
           'B3D': burn_frame, 'G18_orders': orders_frame, 'C3D': paper_frame, 'G26_corfu': map_cor_frame,
           'G29_greece': map_gr_frame, 'C3D_still': doubt_frame, 'B3D_ash': ash_frame, 'G36_plaque': plaque_frame,
           'G37_names': names_frame, 'D3D': finale_frame}

# ---------------------------------------------------------------- subtitles
SUBY = 1610


@functools.lru_cache(None)
def sub_alpha(text):
    size = 58
    while True:
        al = text_layer([text], 'serif', size, 560)
        if al.shape[1] < W - 80 or size <= 40: return al
        size -= 2


def subtitles(img, t):
    for s in SUBS:
        if s['t0'] - 0.05 <= t <= s['t1'] + 0.1:
            a = clamp01((t - s['t0'] + 0.05) / 0.1) * clamp01((s['t1'] + 0.1 - t) / 0.1)
            img = band(img, SUBY - 48, SUBY + 48, 0.62 * a, 70)
            img = paste_alpha(img, sub_alpha(s['text']), W / 2, SUBY, WHITE, a, shadow=1.0, rough=False)
    return img


# ---------------------------------------------------------------- frame
def frame(t, fi):
    cur = [s for s in SHOTS if s['t0'] - XF / 2 <= t < s['t1'] + XF / 2]
    if not cur:
        cur = [SHOTS[-1]]
    if len(cur) == 1:
        img = shot_frame(cur[0], t, fi)
    else:
        a, b = cur[0], cur[1]
        u = ease(clamp01((t - (b['t0'] - XF / 2)) / XF))
        img = crossfade(shot_frame(a, t, fi), shot_frame(b, t, fi), u)
    img = img * clamp01(t / 0.5) * clamp01((T - t) / 1.4)
    img = FILM.apply(img, fi, t, strength=0.9)
    if os.environ.get('NOSUBS') is None:
        img = subtitles(img, t)
    return np.clip(img, 0, 255).astype(np.uint8)


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'stills':
        os.makedirs(f'{ROOT}/test', exist_ok=True)
        for a in sys.argv[2:]:
            t = float(a)
            cv2.imwrite(f'{ROOT}/test/still_{a}.jpg', frame(t, int(t * FPS)), [cv2.IMWRITE_JPEG_QUALITY, 88])
            print('still', a, flush=True)
    elif mode == 'borderchk':
        os.environ['BORDERCHK'] = '1'
        _imread = cv2.imread
        for s_ in SHOTS:
            if s_['k'] not in SPECIAL: content_box(s_['k'])

        def _box_mask(path):
            m = np.full_like(_imread(path, 0), 255)
            x0, y0, x1, y1 = content_box(os.path.basename(str(path))[:-len('_paper.png')])
            m[y0:y1, x0:x1] = 0
            return m
        cv2.imread = lambda path, *a: (_box_mask(path) if str(path).endswith('_paper.png') else _imread(path, *a))
        for s in SHOTS:
            if s['k'] in SPECIAL: continue
            for q in (0.0, 0.25, 0.5, 0.75, 1.0):
                t = lerp(s['t0'] - XF / 2, s['t1'] + XF / 2 - 0.01, q)
                shot_frame(s, t, int(t * FPS))
    elif mode == 'video':
        f0, f1, out = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
        tmp = out.replace('.mp4', '.part.mp4')
        p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                              '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18', '-tune', 'grain', '-pix_fmt', 'yuv420p', tmp], stdin=subprocess.PIPE)
        for fi in range(f0, f1):
            p.stdin.write(frame(fi / FPS, fi).tobytes())
            if fi % 60 == 0: print(out, fi, flush=True)
        p.stdin.close(); p.wait()
        if p.returncode == 0: os.rename(tmp, out)
