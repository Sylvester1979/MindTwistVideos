"""Episode 4 renderer: «Η λίστα του Σίντλερ». Data-driven: timeline.json (shots + subs) + MOVES below.
   python3 render04.py stills t1 t2 ...      -> test/still_<t>.jpg
   python3 render04.py video f0 f1 out.mp4    -> frames [f0,f1)
Plates: $MT_ROOT/plates/<k>.png (prep04.py), depth in $MT_ROOT/depth/<k>.npy (optional).
Graphics drawn here: G04 map 1939, G17/G20/G36 the list (motif), Q30 quote card, S35 inscription."""
import sys, os, json, subprocess, functools, math
import numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'engine'))
from eng import *
from eng import _ink_noise

TL = json.load(open(f'{ROOT}/timeline.json'))
T, SHOTS, SUBS = TL['T'], TL['shots'], TL['subs']
XF = 0.6                       # crossfade between shots (s)
FILM, MOTES = Film(seed=9), Motes(n=50, seed=4)
INKC = np.array([40, 44, 50], np.float32)
CREAMC = np.array([222, 236, 244], np.float32)
ALIAS = {'S33_redcoat': 'S32_redcoat'}

# camera: (cx, cy, zoom) in plate fractions, start -> end; par = (px, py, pz) at the end
D = dict(par=(6, 0, 60), motes=0.0, flick=0.0)
MOVES = {
    'S01_hands':        dict(c=((0.50, 0.55, 1.05), (0.52, 0.60, 1.30)), motes=0.3),
    'S02_crew':         dict(c=((0.50, 0.50, 1.30), (0.50, 0.50, 1.05)), par=(0, 0, -50)),
    'S03_twohands':     dict(c=((0.50, 0.55, 1.05), (0.50, 0.58, 1.28))),
    'S05f_schindler':   dict(c=((0.50, 0.50, 1.05), (0.50, 0.32, 1.30)), motes=0.3),
    'S06_shop':         dict(c=((0.42, 0.52, 1.20), (0.58, 0.50, 1.20)), par=(12, 0, 30)),
    'S07_factory':      dict(c=((0.50, 0.40, 1.15), (0.50, 0.55, 1.15)), par=(0, 8, 30)),
    'S08_pots':         dict(c=((0.50, 0.55, 1.05), (0.46, 0.60, 1.25)), motes=0.5),
    'S09_column':       dict(c=((0.50, 0.62, 1.20), (0.50, 0.45, 1.20)), par=(0, 10, 30)),
    'S10_window':       dict(c=((0.50, 0.50, 1.05), (0.40, 0.45, 1.28))),
    'S11_bribe':        dict(c=((0.50, 0.50, 1.05), (0.48, 0.52, 1.30)), flick=0.6),
    'S12_safe':         dict(c=((0.50, 0.45, 1.05), (0.45, 0.60, 1.25)), flick=0.4),
    'S13_plaszow':      dict(c=((0.42, 0.50, 1.20), (0.58, 0.50, 1.20)), par=(10, 0, 30)),
    'S14_wagons':       dict(c=((0.50, 0.50, 1.05), (0.55, 0.45, 1.30)), par=(0, 0, 90)),
    'S15_desk':         dict(c=((0.50, 0.50, 1.05), (0.55, 0.32, 1.30)), motes=0.3),
    'S16_shells':       dict(c=((0.50, 0.55, 1.05), (0.50, 0.50, 1.30)), par=(0, 0, 110), motes=0.3),
    'S18_set_typewriter': dict(c=((0.50, 0.50, 1.05), (0.45, 0.38, 1.32))),
    'S19_pemper':       dict(c=((0.50, 0.50, 1.05), (0.55, 0.45, 1.30)), motes=0.3),
    'S21_train':        dict(c=((0.50, 0.60, 1.20), (0.50, 0.42, 1.08)), par=(0, 0, -40)),
    'S22_old_schindler': dict(c=((0.50, 0.50, 1.05), (0.42, 0.36, 1.30))),
    'S23_zion':         dict(c=((0.50, 0.40, 1.15), (0.50, 0.62, 1.15)), par=(0, 8, 30)),
    'S24_beverly':      dict(c=((0.50, 0.50, 1.05), (0.55, 0.55, 1.25)), par=(0, 0, 80)),
    'S25_shop':         dict(c=((0.50, 0.50, 1.05), (0.58, 0.42, 1.28))),
    'S26_poldek':       dict(c=((0.50, 0.52, 1.05), (0.50, 0.33, 1.30)), motes=0.3),
    'S27_book_reel':    dict(c=((0.50, 0.50, 1.05), (0.50, 0.55, 1.25)), motes=0.5),
    'S28_set_night':    dict(c=((0.50, 0.55, 1.05), (0.50, 0.45, 1.28)), par=(0, 0, 80)),
    'S29_mila':         dict(c=((0.55, 0.50, 1.05), (0.70, 0.40, 1.35))),
    'S31_hands_tremble': dict(c=((0.50, 0.45, 1.10), (0.55, 0.40, 1.25)), shake=1.0),
    'S32_redcoat':      dict(c=((0.50, 0.55, 1.05), (0.50, 0.72, 1.75))),
    'S33_redcoat':      dict(c=((0.50, 0.72, 1.75), (0.50, 0.55, 1.05))),
    'S34_zion1993':     dict(c=((0.50, 0.45, 1.25), (0.55, 0.62, 1.25)), par=(0, 10, 30)),
    'S35_slab':         dict(c=((0.50, 0.50, 1.05), (0.50, 0.46, 1.40))),
}
# text cards: shot -> (lines, size, y, t_in, t_out) times relative to shot start (t_out < 0: from shot end)
CARDS = {
    'S05f_schindler': (['Όσκαρ Σίντλερ'], 76, 0.14, 0.3, -0.4),
    'S07_factory': (['Κρακοβία'], 66, 0.14, 0.2, -0.3),
    'S13_plaszow': (['Πλάσοφ, 1944'], 66, 0.14, 0.4, -0.5),
    'S16_shells': (['Μπρίνλιτς'], 66, 0.14, 0.2, -0.3),
    'S18_set_typewriter': (['Στην ταινία'], 66, 0.14, 0.1, -0.2),
    'S19_pemper': (['Μίετεκ Πέμπερ'], 70, 0.14, 0.8, -0.4),
    'S23_zion': (['1974'], 70, 0.14, 0.2, -0.3),
    'S24_beverly': (['Μπέβερλι Χιλς, 1980'], 64, 0.14, 0.1, -0.2),
    'S26_poldek': (['Πόλντεκ Πφέφερμπεργκ'], 66, 0.14, 0.3, -0.5),
    'S29_mila': (['Μίλα Πφέφερμπεργκ'], 66, 0.14, 0.2, -0.4),
}
QUOTES = {   # shot -> (lines, attribution, y, t_in rel. start, t_out rel. end)
    'Q30_goeth': (['«Όταν έβλεπες τον Γκαιτ,', 'έβλεπες τον θάνατο.»'], 'Πόλντεκ Πφέφερμπεργκ', 0.44, 0.2, -0.3),
    'S33_redcoat': (['«Η Αμερική ήξερε.', 'Η Βρετανία ήξερε.', 'Η Ρωσία ήξερε.»'], 'Στίβεν Σπίλμπεργκ', 0.24, 0.3, -0.4),
}
QWIN = [(s['t0'] + QUOTES[s['k']][3] - 0.2, s['t1'] + QUOTES[s['k']][4]) for s in SHOTS if s['k'] in QUOTES]
INSCRIPTION = ['OSKAR SCHINDLER', '28.4.1908 – 9.10.1974', 'DER UNVERGESSLICHE', 'LEBENSRETTER', '1200 VERFOLGTER JUDEN']
SLAB_BOX = (0.28, 0.395, 0.72, 0.565)   # blank area of the S35 slab (plate fractions)


# ---------------------------------------------------------------- plates
@functools.lru_cache(None)
def plate(k):
    P = Plate(ALIAS.get(k, k))
    if k == 'S35_slab':
        carve(P)
    return P


def carve(P):
    """German inscription cut into the blank centre of the slab (drawn into the plate, so it moves with it)."""
    l, t, r, b = SLAB_BOX
    x0, y0, x1, y1 = int(l * P.pw), int(t * P.ph), int(r * P.pw), int(b * P.ph)
    from PIL import Image, ImageDraw
    w, h = x1 - x0, y1 - y0
    im = Image.new('L', (w, h), 0); d = ImageDraw.Draw(im)
    lh = h / (len(INSCRIPTION) + 1.4)          # block sits lower and tighter: clear of the stones at the top corners
    for i, ln in enumerate(INSCRIPTION):
        size = 64
        while font('serif', size, 600).getlength(ln) + 8 * len(ln) > w * 0.74 and size > 30: size -= 2
        fi = font('serif', size, 600)
        tw = sum(fi.getlength(c) + 8 for c in ln) - 8
        x = (w - tw) / 2; y = lh * (i + 0.95)
        for c in ln:
            d.text((x, y), c, font=fi, fill=255); x += fi.getlength(c) + 8
    a = np.asarray(im).astype(np.float32) / 255
    a = cv2.GaussianBlur(a, (0, 0), 0.8)
    hi = np.roll(np.roll(a, 2, 0), 2, 1)          # lit lower-right edge of the cut
    sub = P.img[y0:y1, x0:x1]
    sub = sub * (1 - 0.72 * a[..., None]) + np.float32(40) * 0.72 * a[..., None]
    sub = sub * (1 - 0.25 * np.clip(hi - a, 0, 1)[..., None]) + PAPER * 0.25 * np.clip(hi - a, 0, 1)[..., None]
    P.img[y0:y1, x0:x1] = sub


@functools.lru_cache(None)
def content_box(key):
    """the drawing inside its blank paper margin (and, if there is one, the drawn frame line):
    rows/cols from each edge that are >90% blank paper (_paper mask), +14 px for the frame line.
    S22 has a wide drawn frame (6-8%); the others a thin margin of ~1.5%."""
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
    """never let the camera (incl. the parallax/dolly displacement) see past the drawing: not the plate edge
    (the engine would mirror it) and not the paper margin/frame (it would read as a photo being zoomed).
    Zoom in just enough, then clamp the centre."""
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


def shot_frame(s, t, fi):
    k = s['k']
    if k in SPECIAL:
        return SPECIAL[k](s, t, fi)
    mv = {**D, **MOVES.get(k, {})}
    (a0, b0, z0), (a1, b1, z1) = mv.get('c', ((0.5, 0.5, 1.05), (0.5, 0.5, 1.25)))
    dur = s['t1'] - s['t0'] + XF
    u = ease(clamp01((t - s['t0'] + XF / 2) / dur))
    P = plate(k)
    px, py, pz = mv['par']
    # clamp the two END POINTS (not each frame): the move stays smooth and never sticks to an edge
    a0, b0, z0 = keep_inside(P, a0 * P.pw, b0 * P.ph, z0, 0, 0, 0)
    a1, b1, z1 = keep_inside(P, a1 * P.pw, b1 * P.ph, z1, px, py, pz)
    cx, cy = lerp(a0, a1, u), lerp(b0, b1, u)
    if mv.get('shake'):
        cx += mv['shake'] * 2.2 * math.sin(t * 31.0) * (1 + 0.5 * math.sin(t * 3.1))
        cy += mv['shake'] * 1.6 * math.sin(t * 27.3 + 1)
    z = lerp(z0, z1, u)
    cx, cy, z = keep_inside(P, cx, cy, z, px * u, py * u, pz * u)       # safety net only
    cam = (cx, cy, z)
    img = P.render(t, fi, cam, par=(px * u, py * u, pz * u), focus=0.55, boil=0.0)
    if mv['flick']:
        img = img * (1 + 0.035 * mv['flick'] * (np.sin(t * 9.1) + 0.6 * np.sin(t * 15.7 + 1)))
    if mv['motes']:
        img = MOTES.apply(img, t, mv['motes'])
    if k in CARDS:
        img = card(img, s, t, *CARDS[k])
    if k in QUOTES:
        img = quote(img, s, t, *QUOTES[k])
    return img


def card(img, s, t, lines, size, y, ti, to):
    t_out = s['t1'] + to if to < 0 else s['t0'] + to
    a = clamp01((t - s['t0'] - ti) / 0.6) * clamp01((t_out - t) / 0.6)
    if a <= 0: return img
    al = text_layer(lines, 'serif', size, 560, 1.25)
    cy = int(H * y)
    img = band(img, cy - al.shape[0] // 2 - 6, cy + al.shape[0] // 2 + 6, 0.45 * a, 90)
    return paste_alpha(img, al, W / 2, cy, (236, 242, 246), a, shadow=1.0, rough=True)


def paper_text(img, al, cy, a, pad=(44, 40), op=0.92):
    """ink text on a torn cream paper strip."""
    px, py = pad
    h, w = al.shape[0] + 2 * py, al.shape[1] + 2 * px
    x0, y0 = W // 2 - w // 2, cy - h // 2
    n = _ink_noise(5)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.minimum(np.minimum(xx, w - 1 - xx), np.minimum(yy, h - 1 - yy)) + (n[:h, :w] - 0.5) * 18
    edge = np.clip(d / 3.0, 0, 1)
    sub = img[y0:y0 + h, x0:x0 + w]
    sh = cv2.GaussianBlur(edge, (0, 0), 7)[..., None]
    sub = sub * (1 - 0.35 * a * np.roll(np.roll(sh, 6, 0), 4, 1))
    paper = CREAMC * (0.93 + 0.07 * n[:h, :w])[..., None]
    m = (edge * op * a)[..., None]
    img[y0:y0 + h, x0:x0 + w] = sub * (1 - m) + paper * m
    return paste_alpha(img, al, W / 2, cy, INKC, a, rough=True)


def quote(img, s, t, lines, who, y, ti, to):
    t_in = s['t0'] + ti
    t_out = s['t1'] + to if to < 0 else s['t0'] + to
    a = clamp01((t - t_in) / 0.8) * clamp01((t_out - t) / 0.8)
    if a <= 0: return img
    al = text_layer(lines, 'serif', 72, 520, 1.22)
    al2 = text_layer(['— ' + who], 'serif', 44, 460)
    h = al.shape[0] + al2.shape[0] + 10
    both = np.zeros((h, max(al.shape[1], al2.shape[1])), np.float32)
    both[:al.shape[0], (both.shape[1] - al.shape[1]) // 2:(both.shape[1] - al.shape[1]) // 2 + al.shape[1]] = al
    o = (both.shape[1] - al2.shape[1]) // 2
    both[al.shape[0] + 10:, o:o + al2.shape[1]] = al2 * 0.85
    return paper_text(img, both, int(H * y), a)


# ---------------------------------------------------------------- dark backgrounds
@functools.lru_cache(None)
def dark_bg():
    n = _ink_noise(9)
    return INK * 1.6 * (0.8 + 0.4 * n)[..., None] * vignette()


@functools.lru_cache(None)
def paper_bg():
    return PAPER * 1.02 * paper_tex() * (0.92 + 0.08 * fbm(H, W, 220, 61, 3))[..., None]


def q30_frame(s, t, fi):
    out = dark_bg().copy()
    return quote(out, s, t, *QUOTES['Q30_goeth'])


# ---------------------------------------------------------------- map 1939
LON0, LON1, LAT0, LAT1 = 9.0, 27.0, 42.5, 61.5     # tall enough for 9:16 at every zoom (no edge reflection)
MW = 2400
COSL = math.cos(math.radians(51.5))
MH = int(MW * (LAT1 - LAT0) / ((LON1 - LON0) * COSL))
CITIES = {'Βερολίνο': (13.40, 52.52), 'Βαρσοβία': (21.01, 52.23), 'Κρακοβία': (19.94, 50.06)}
# German advance, 1 Sept 1939 (no modern borders are drawn: they are not the 1939 ones)
ARROWS = [[(16.2, 54.0), (17.3, 53.8), (18.5, 53.45)],          # from Pomerania, across the Corridor
          [(20.8, 54.05), (21.0, 53.4), (21.0, 52.65)],         # from East Prussia
          [(14.9, 52.35), (16.2, 52.3), (17.8, 52.2)],          # from the west
          [(17.2, 51.0), (18.3, 50.6), (19.3, 50.35)],          # from Silesia
          [(19.6, 49.05), (19.85, 49.6), (19.95, 49.95)]]       # from Slovakia


def mproj(lon, lat):
    return (np.asarray(lon) - LON0) / (LON1 - LON0) * MW, (LAT1 - np.asarray(lat)) / (LAT1 - LAT0) * MH


def _geo(path):
    gj = json.load(open(path)); out = []
    for ft in gj['features']:
        g = ft['geometry']
        if g is None: continue
        polys = {'Polygon': [g['coordinates']], 'MultiPolygon': g['coordinates'],
                 'LineString': [[g['coordinates']]], 'MultiLineString': [[l] for l in g['coordinates']]}.get(g['type'], [])
        for poly in polys:
            for ring in poly:
                arr = np.array(ring, np.float64)
                if arr.ndim != 2 or len(arr) < 2: continue
                if arr[:, 0].max() < LON0 - 2 or arr[:, 0].min() > LON1 + 2 or arr[:, 1].max() < LAT0 - 2 or arr[:, 1].min() > LAT1 + 2:
                    continue
                if np.abs(np.diff(arr[:, 0])).max() > 30: continue
                out.append(arr)
    return out


def major_rivers(maxrank=8):
    """only the big rivers (Vistula, Oder, Elbe, Warta, Bug, Danube...): a readable map, not a tangle."""
    gj = json.load(open(f'{ROOT}/map/ne_10m_rivers_lake_centerlines.geojson')); out = []
    for ft in gj['features']:
        pr, g = ft['properties'], ft['geometry']
        if g is None or pr.get('featurecla') != 'River' or (pr.get('scalerank') or 99) > maxrank: continue
        lines = [g['coordinates']] if g['type'] == 'LineString' else g['coordinates']
        for l in lines:
            arr = np.array(l, np.float64)
            if arr.ndim != 2 or len(arr) < 2: continue
            if arr[:, 0].max() < LON0 or arr[:, 0].min() > LON1 or arr[:, 1].max() < LAT0 or arr[:, 1].min() > LAT1: continue
            out.append(arr)
    return out


@functools.lru_cache(None)
def map_static():
    img = np.ones((MH, MW, 3), np.float32) * PAPER * 1.02 * (0.90 + 0.12 * fbm(MH, MW, 260, 41, 4))[..., None]
    land = np.zeros((MH, MW), np.uint8); coast = []
    for arr in _geo(f'{ROOT}/map/ne_10m_land.geojson'):
        x, y = mproj(np.clip(arr[:, 0], LON0 - 3, LON1 + 3), np.clip(arr[:, 1], LAT0 - 3, LAT1 + 3))
        pts = np.stack([x, y], 1)
        cv2.fillPoly(land, [np.round(pts * 4).astype(np.int32)], 255, cv2.LINE_AA, shift=2); coast.append(pts)
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
    for arr in major_rivers():
        x, y = mproj(arr[:, 0], arr[:, 1])
        cv2.polylines(lay, [np.round(np.stack([x, y], 1) * 4).astype(np.int32)], False, 0.6, 3, cv2.LINE_AA, shift=2)
    lay = np.clip(lay, 0, 1) * (0.8 + 0.2 * fbm(MH, MW, 6, 44, 2))
    img = img * (1 - lay[..., None]) + INK * lay[..., None]
    from PIL import Image, ImageDraw
    pil = Image.new('L', (MW, MH), 0); d = ImageDraw.Draw(pil)

    def lab(txt, lon, lat, size, wght=600, track=6, dy=0):
        x, y = mproj(lon, lat); f = font('serif', size, wght)
        tw = sum(f.getlength(ch) + track for ch in txt) - track
        xx = x - tw / 2
        for ch in txt:
            d.text((xx, y - size / 2 + dy), ch, font=f, fill=255); xx += f.getlength(ch) + track
    lab('ΓΕΡΜΑΝΙΑ', 13.6, 51.2, 80, 600, 22)
    lab('ΠΟΛΩΝΙΑ', 20.6, 51.4, 92, 600, 26)
    lab('ΒΑΛΤΙΚΗ', 16.5, 55.4, 50, 500, 14)
    for name, (lon, lat) in CITIES.items():
        lab(name, lon, lat, 50, 560, 2, dy=-48)
    la = np.asarray(pil).astype(np.float32) / 255
    img = img * (1 - la[..., None] * 0.9) + INK * la[..., None] * 0.9
    for name, (lon, lat) in CITIES.items():
        x, y = mproj(lon, lat); cv2.circle(img, (int(x), int(y)), 11, tuple(float(v) for v in INK), -1, cv2.LINE_AA)
    for fx in (MW / 3, 2 * MW / 3):
        img *= (1 - 0.10 * np.exp(-((np.arange(MW) - fx) / 6) ** 2)[None, :, None])
    img *= (1 - 0.10 * np.exp(-((np.arange(MH) - MH / 2) / 6) ** 2)[:, None, None])
    return img


def arrow_layer(p):
    lay = np.zeros((MH, MW), np.float32)
    for k, path in enumerate(ARROWS):
        q = clamp01(p * 1.25 - k * 0.06)
        if q <= 0: continue
        pts = np.array([mproj(*c) for c in path], np.float32)
        # smooth with a quadratic bezier
        tt = np.linspace(0, 1, 60)[:, None]
        bz = (1 - tt) ** 2 * pts[0] + 2 * (1 - tt) * tt * pts[1] + tt ** 2 * pts[2]
        n = max(2, int(len(bz) * ease(q)))
        seg = bz[:n]
        cv2.polylines(lay, [np.round(seg * 4).astype(np.int32)], False, 1.0, 22, cv2.LINE_AA, shift=2)
        if n >= 3:
            tip, prev = seg[-1], seg[-3]
            dv = (tip - prev) / (np.linalg.norm(tip - prev) + 1e-6); nv = np.array([-dv[1], dv[0]])
            tri = np.array([tip + dv * 42, tip - dv * 18 + nv * 38, tip - dv * 18 - nv * 38])
            cv2.fillPoly(lay, [np.round(tri * 4).astype(np.int32)], 1.0, cv2.LINE_AA, shift=2)
    return lay * (0.85 + 0.15 * fbm(MH, MW, 8, 47, 2))


def view(img, cx, cy, zoom):
    s = W * zoom / MW
    M = np.float32([[s, 0, W / 2 - cx * s], [0, s, H / 2 - cy * s]])
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), M


def map_frame(s, t, fi):
    r = t - s['t0']; dur = s['t1'] - s['t0']
    u = ease(clamp01((r + XF / 2) / (dur + XF)))
    cx0, cy0 = mproj(17.0, 52.3); cx1, cy1 = mproj(18.3, 51.6)
    base = map_static().copy()
    p = clamp01((r - 1.6) / max(1.0, dur - 2.2))
    al = arrow_layer(p)[..., None] * 0.9
    base = base * (1 - al) + INK * al
    out, _ = view(base, lerp(cx0, cx1, u), lerp(cy0, cy1, u), lerp(1.18, 1.42, u))
    out = out * vignette()
    return card(out, s, t, ['Σεπτέμβριος 1939'], 66, 0.14, 0.2, -0.3)


# ---------------------------------------------------------------- the list (motif)
@functools.lru_cache(None)
def page_img(w=1500, h=2300, seed=1, lines=26, start=1):
    """a typed list page: numbers in typewriter type, names as illegible inked strokes (no letters)."""
    rng = np.random.default_rng(seed)
    img = page_blank(w, h, seed).copy()
    lay = np.zeros((h, w), np.float32)
    top, step = 260, (h - 400) / lines
    # header rule and column rules
    cv2.line(lay, (120, top - 60), (w - 120, top - 60), 0.8, 3, cv2.LINE_AA)
    cv2.line(lay, (120, top - 48), (w - 120, top - 48), 0.6, 2, cv2.LINE_AA)
    for i in range(lines):
        y = int(top + i * step)
        al = text_layer([f'{start + i}.'], 'type', 46, None)
        hh, ww_ = al.shape
        y0, x0 = y - hh // 2, 150
        lay[y0:y0 + hh, x0:x0 + ww_] = np.maximum(lay[y0:y0 + hh, x0:x0 + ww_], al * 0.85)
        x = 330
        for _ in range(int(rng.integers(2, 4))):                # surname, first name, (job)
            L = int(rng.integers(150, 330))
            for k in range(L // 18):                             # typed-looking marks, no letters
                xx = x + k * 18
                hgt = int(rng.integers(14, 26)); top_ = y - hgt // 2 + int(rng.integers(-2, 3))
                cv2.rectangle(lay, (xx, top_), (xx + 11, top_ + hgt), float(0.55 + 0.35 * rng.random()), -1)
            x += L + 50
            if x > w - 300: break
    lay = cv2.GaussianBlur(lay, (0, 0), 1.0) * (0.75 + 0.25 * fbm(h, w, 5, 70 + seed, 2))
    return img * (1 - lay[..., None]) + INKC * lay[..., None]


@functools.lru_cache(None)
def page_blank(w=1500, h=2300, seed=1):
    return np.ones((h, w, 3), np.float32) * PAPER * 1.04 * (0.9 + 0.1 * fbm(h, w, 70, 60 + seed, 4))[..., None]


def typed(page, p, lines=26, h=2300):
    """reveal the page line by line (p 0..1), as if typed."""
    top, step = 260, (h - 400) / lines
    n = p * lines
    i = int(n)
    if i >= lines: return page
    out = page.copy()
    blank = page_blank()
    yc = int(top + i * step - step / 2)
    x = int(330 + (n - i) * (page.shape[1] - 430)) if n - i > 0.02 else 0
    out[yc:yc + int(step), x:] = blank[yc:yc + int(step), x:]
    out[yc + int(step):] = blank[yc + int(step):]
    return out


def list_frame(s, t, fi):
    r = t - s['t0']; dur = s['t1'] - s['t0']
    u = ease(clamp01((r + XF / 2) / (dur + XF)))
    pg = typed(page_img(), clamp01(r / max(1.0, dur * 0.95)))
    out, _ = view_page(pg, 0.47, lerp(0.32, 0.42, u), lerp(1.12, 1.25, u))
    return out * vignette()


def view_page(pg, fx, fy, zoom):
    h, w = pg.shape[:2]
    s = W * zoom / w
    bg = paper_bg().copy() * 0.55
    M = np.float32([[s, 0, W / 2 - fx * w * s], [0, s, H / 2 - fy * h * s]])
    out = cv2.warpAffine(pg, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    m = cv2.warpAffine(np.ones((h, w), np.float32), M, (W, H), flags=cv2.INTER_LINEAR)[..., None]
    return bg * (1 - m) + out * m, M


@functools.lru_cache(None)
def pages_wall(n=24):
    """many list pages laid side by side (for the counts)."""
    cols, rows = 6, 4
    pw, ph = 750, 1150
    wall = np.ones((rows * ph + 60, cols * pw + 60, 3), np.float32) * INK * 1.4
    for i in range(cols * rows):
        pg = cv2.resize(page_img(seed=1 + i % 7, start=1 + 26 * i), (pw - 40, ph - 40), interpolation=cv2.INTER_AREA)
        y, x = 50 + (i // cols) * ph, 50 + (i % cols) * pw
        wall[y:y + ph - 40, x:x + pw - 40] = pg
    return wall


def counter_overlay(out, value, a, y=0.47, label=None):
    s = f'{value:,}'.replace(',', '.')
    al = text_layer([s], 'serif', 230, 620)
    cy = int(H * y)
    out = band(out, cy - 150, cy + 150, 0.70 * a, 120)
    out = paste_alpha(out, al, W / 2, cy, (236, 242, 246), a, shadow=1.0, rough=True)
    if label:
        out = paste_alpha(out, text_layer([label], 'serif', 60, 520), W / 2, cy + 150, (236, 242, 246), a, shadow=1.0)
    return out


def count_frame(s, t, fi):
    r = t - s['t0']; dur = s['t1'] - s['t0']
    u = ease(clamp01((r + XF / 2) / (dur + XF)))
    wall = pages_wall()
    h, w = wall.shape[:2]
    zoom = lerp(4.2, 1.0, ease2(clamp01(r / max(0.8, dur))))
    sx = W * zoom / w
    M = np.float32([[sx, 0, 0], [0, sx, 0]])
    M[0, 2] = W / 2 - lerp(0.5, 0.083, clamp01((zoom - 1) / 3.2)) * w * sx
    M[1, 2] = H / 2 - lerp(0.5, 0.135, clamp01((zoom - 1) / 3.2)) * h * sx
    out = cv2.warpAffine(wall, M, (W, H), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT, borderValue=tuple(float(v) for v in INK))
    out = out * vignette()
    v = int(round(1200 * ease2(clamp01((r - 0.2) / max(0.5, dur * 0.75)))))
    return counter_overlay(out, v, clamp01((r - 0.1) / 0.4))


def final_frame(s, t, fi):
    """descendants: the wall of pages multiplies, counter 1.200 -> 8.500+, then the single page remains."""
    r = t - s['t0']; dur = s['t1'] - s['t0']
    wall = pages_wall()
    h, w = wall.shape[:2]
    p = ease2(clamp01((r - 0.3) / max(1.0, dur * 0.55)))
    zoom = lerp(1.0, 0.55, p)
    sx = W * zoom / w
    M = np.float32([[sx, 0, W / 2 - 0.5 * w * sx], [0, sx, H / 2 - 0.5 * h * sx]])
    out = np.ones((H, W, 3), np.float32) * INK * 1.4
    # tile the wall to fill as it shrinks
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            M2 = M.copy(); M2[0, 2] += dx * w * sx; M2[1, 2] += dy * h * sx
            lay = cv2.warpAffine(wall, M2, (W, H), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            m = cv2.warpAffine(np.ones((h, w), np.float32), M2, (W, H), flags=cv2.INTER_LINEAR)[..., None]
            out = out * (1 - m) + lay * m
    out = out * vignette()
    v = int(round(lerp(1200, 8500, ease2(clamp01((r - 0.3) / max(1.0, dur * 0.40))))))   # lands on 8.500 as the word is spoken
    a = clamp01((r - 0.1) / 0.4)
    out = counter_overlay(out, v, a)
    if v >= 8500:
        plus = text_layer(['+'], 'serif', 150, 620)
        out = paste_alpha(out, plus, W / 2 + 330, int(H * 0.47) - 20, (236, 242, 246), a, shadow=1.0)
    return out


SPECIAL = {'G04_map': map_frame, 'G17_list': list_frame, 'G20_count': count_frame,
           'Q30_goeth': q30_frame, 'G36_final': final_frame}

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
        if any(a <= s['t0'] <= b for a, b in QWIN):
            continue   # the quote card already shows these words
        if s['t0'] - 0.05 <= t <= s['t1'] + 0.1:
            a = clamp01((t - s['t0'] + 0.05) / 0.1) * clamp01((s['t1'] + 0.1 - t) / 0.1)
            img = band(img, SUBY - 48, SUBY + 48, 0.45 * a, 70)
            img = paste_alpha(img, sub_alpha(s['text']), W / 2, SUBY, (236, 242, 246), a, shadow=1.0, rough=False)
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
        # BORDERCHK=1 makes Plate.render print how much paper margin the camera sees (0 = clean frame)
        # the _paper masks also mark white areas INSIDE the drawing, so we swap in a mask that is 1 only
        # outside content_box -> the printed value is the share of the frame that shows margin or plate edge.
        os.environ['BORDERCHK'] = '1'
        _imread = cv2.imread
        for s_ in SHOTS:                      # fill the cache with the real masks first
            if s_['k'] not in SPECIAL: content_box(ALIAS.get(s_['k'], s_['k']))

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
