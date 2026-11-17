"""Procedural elements drawn by us: pocket watch, war map, handwritten note, crosses counter, grave plaque."""
import cv2, numpy as np, json, functools, math
from PIL import Image, ImageDraw
from eng import *
from eng import _ink_noise

# ================================================================= generic view warp
def view(img, cx, cy, zoom, rot=0.0, bg=None, interp=cv2.INTER_LINEAR):
    """camera into an arbitrary image: zoom 1 = image width fills output width."""
    ih, iw = img.shape[:2]
    s = W * zoom / iw
    c, si = math.cos(rot), math.sin(rot)
    M = np.float32([[s * c, -s * si, 0], [s * si, s * c, 0]])
    M[:, 2] = np.float32([W / 2, H / 2]) - M[:, :2] @ np.float32([cx, cy])
    bv = tuple(float(v) for v in (bg if bg is not None else PAPER))
    if img.ndim == 3 and img.shape[2] == 4:
        bv = bv + (0.0,)
    return cv2.warpAffine(img, M, (W, H), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=bv), M

def to_screen(M, x, y):
    return M[0, 0] * x + M[0, 1] * y + M[0, 2], M[1, 0] * x + M[1, 1] * y + M[1, 2]

# ================================================================= pocket watch
WS = 2000  # watch canvas size
WR = 0.40 * WS

@functools.lru_cache(None)
def wood_bg():
    """dark charcoal table surface, portrait 1080x1920 *1.5 for camera moves"""
    h, w = int(H * 1.5), int(W * 1.5)
    streak = cv2.resize(fbm(h // 8, w // 1, 30, 9, 4), (w, h))
    n = fbm(h, w, 90, 10, 4)
    v = 0.10 + 0.07 * streak + 0.05 * n
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    spot = np.exp(-(((x - w * 0.45) / (w * 0.55)) ** 2 + ((y - h * 0.42) / (h * 0.45)) ** 2))
    v = v * (0.55 + 0.75 * spot)
    return np.clip(v[..., None] * PAPER * 1.0, 0, 255)

def _poly(img, pts, color, aa=True):
    cv2.fillPoly(img, [np.round(np.array(pts) * 16).astype(np.int32)], color, cv2.LINE_AA if aa else 8, shift=4)

def _rot(pts, ang, cx, cy):
    c, s = math.cos(ang), math.sin(ang)
    return [(cx + x * c - y * s, cy + x * s + y * c) for x, y in pts]

@functools.lru_cache(None)
def watch_static():
    S = WS; c = S / 2
    img = np.zeros((S, S, 3), np.float32); a = np.zeros((S, S), np.float32)
    y, x = np.mgrid[0:S, 0:S].astype(np.float32)
    r = np.sqrt((x - c) ** 2 + (y - c) ** 2); ang = np.arctan2(y - c, x - c)
    # bow + crown
    cv2.ellipse(img, (int(c), int(c - WR - 150)), (110, 120), 0, 0, 360, (95, 100, 104), 26, cv2.LINE_AA)
    cv2.ellipse(a, (int(c), int(c - WR - 150)), (110, 120), 0, 0, 360, 1.0, 26, cv2.LINE_AA)
    cv2.rectangle(img, (int(c - 55), int(c - WR - 85)), (int(c + 55), int(c - WR + 5)), (70, 74, 78), -1, cv2.LINE_AA)
    cv2.rectangle(a, (int(c - 55), int(c - WR - 85)), (int(c + 55), int(c - WR + 5)), 1.0, -1)
    for k in range(-50, 51, 12):
        cv2.line(img, (int(c + k), int(c - WR - 85)), (int(c + k), int(c - WR)), (40, 42, 45), 3, cv2.LINE_AA)
    # case
    case = (r < WR + 60).astype(np.float32)
    shade = 0.55 + 0.45 * np.cos(ang + 2.4) * 0.6 + 0.25 * np.clip(1 - np.abs(r - WR - 30) / 30, 0, 1)
    metal = np.clip(shade, 0.15, 1.2)[..., None] * np.array([118, 124, 128], np.float32)
    knurl = (np.sin(ang * 180) > 0.2) & (r > WR + 44)
    metal[knurl] *= 0.6
    img = img * (1 - case[..., None]) + metal * case[..., None]
    a = np.maximum(a, cv2.GaussianBlur(case, (0, 0), 1.2))
    # bezel inner shadow
    rim = np.clip(1 - np.abs(r - WR - 4) / 10, 0, 1)
    img *= (1 - 0.6 * rim[..., None])
    # dial
    dial = cv2.GaussianBlur((r < WR).astype(np.float32), (0, 0), 1.0)
    stain = fbm(S, S, 120, 21, 4)
    dv = (0.97 - 0.10 * (r / WR) ** 2 - 0.10 * np.clip(stain - 0.5, 0, 1) * 2 + 0.03 * fbm(S, S, 8, 22, 2))
    dcol = PAPER * 1.04 * np.clip(dv, 0, 1.1)[..., None]
    img = img * (1 - dial[..., None]) + dcol * dial[..., None]
    # cracks in enamel
    rng = np.random.default_rng(4)
    for _ in range(5):
        p = np.array([c, c]) + rng.normal(0, WR * 0.4, 2); th = rng.random() * 6.28
        pts = [p.copy()]
        for _ in range(12):
            th += rng.normal(0, 0.5); p = p + [math.cos(th) * 18, math.sin(th) * 18]; pts.append(p.copy())
        pts = np.array(pts)
        if np.all(np.hypot(pts[:, 0] - c, pts[:, 1] - c) < WR - 10):
            cv2.polylines(img, [np.int32(pts)], False, (120, 125, 130), 1, cv2.LINE_AA)
    # minute track
    for i in range(60):
        th = i / 60 * 2 * math.pi - math.pi / 2
        r0, r1, wd = (WR - 70, WR - 22, 7) if i % 5 == 0 else (WR - 48, WR - 22, 3)
        cv2.line(img, (int(c + r0 * math.cos(th)), int(c + r0 * math.sin(th))),
                 (int(c + r1 * math.cos(th)), int(c + r1 * math.sin(th))), tuple(float(v) for v in INK), wd, cv2.LINE_AA)
    cv2.circle(img, (int(c), int(c)), int(WR - 20), tuple(float(v) for v in INK), 4, cv2.LINE_AA)
    cv2.circle(img, (int(c), int(c)), int(WR - 74), tuple(float(v) for v in INK), 2, cv2.LINE_AA)
    # roman numerals (upright)
    pil = Image.fromarray(np.zeros((S, S), np.uint8)); d = ImageDraw.Draw(pil)
    f = font('serif', 120, 600)
    for i, rn in enumerate(['XII', 'I', 'II', 'III', 'IIII', 'V', '', 'VII', 'VIII', 'IX', 'X', 'XI']):
        if not rn: continue
        th = i / 12 * 2 * math.pi - math.pi / 2
        rr = WR - 170
        px, py = c + rr * math.cos(th), c + rr * math.sin(th)
        bb = d.textbbox((0, 0), rn, font=f)
        d.text((px - (bb[2] - bb[0]) / 2 - bb[0], py - (bb[3] - bb[1]) / 2 - bb[1]), rn, font=f, fill=255)
    na = np.asarray(pil).astype(np.float32) / 255 * (0.85 + 0.15 * fbm(S, S, 5, 23, 2))
    img = img * (1 - na[..., None]) + INK * na[..., None]
    # sub-seconds dial
    sc = (c, c + WR * 0.47); sr = WR * 0.20
    cv2.circle(img, (int(sc[0]), int(sc[1])), int(sr), tuple(float(v) for v in INK), 3, cv2.LINE_AA)
    for i in range(60):
        th = i / 60 * 2 * math.pi - math.pi / 2
        r0 = sr - (26 if i % 5 == 0 else 14)
        cv2.line(img, (int(sc[0] + r0 * math.cos(th)), int(sc[1] + r0 * math.sin(th))),
                 (int(sc[0] + (sr - 4) * math.cos(th)), int(sc[1] + (sr - 4) * math.sin(th))), tuple(float(v) for v in INK), 3 if i % 5 == 0 else 1, cv2.LINE_AA)
    return img, a, sc, sr

def _hand_hour(L):
    return [(-14, 40), (-10, -L * 0.55), (-38, -L * 0.72), (0, -L), (38, -L * 0.72), (10, -L * 0.55), (14, 40)]

def _hand_min(L):
    return [(-10, 50), (-7, -L * 0.8), (-22, -L * 0.86), (0, -L), (22, -L * 0.86), (7, -L * 0.8), (10, 50)]

def watch(clock, arc=None, glint=0.0, stopped=False):
    """clock = seconds since midnight (float). arc=(from_sec,to_sec,progress) drawn in red.
    returns (bgr, alpha) at WS."""
    img, a, sc, sr = watch_static()
    img = img.copy(); c = WS / 2
    h = (clock / 3600) % 12; m = (clock / 60) % 60
    s_int = math.floor(clock); frac = clock - s_int
    if stopped:
        sec = s_int % 60
    else:
        k = min(1, frac / 0.09)
        over = math.sin(min(1, frac / 0.22) * math.pi) * 0.12 * (1 - k * 0.3)
        sec = (s_int - 1 + k) % 60 + (over if frac < 0.22 else 0)
    if arc is not None:
        f0, f1, p = arc
        a0 = ((f0 / 3600) % 12) / 12 * 360 - 90
        a1 = a0 + (((f1 - f0) / 3600) / 12 * 360) * p
        lay = np.zeros((WS, WS), np.float32)
        cv2.ellipse(lay, (int(c), int(c)), (int(WR - 46), int(WR - 46)), 0, a0, a1, 1.0, 44, cv2.LINE_AA)
        lay = lay * (0.7 + 0.3 * fbm(WS, WS, 14, 31, 3)) * 0.8
        img = img * (1 - lay[..., None]) + RED * lay[..., None]
    sh = np.zeros((WS, WS), np.float32)  # hand shadows
    hand_ang = h / 12 * 2 * math.pi; min_ang = m / 60 * 2 * math.pi
    for pts in (_rot(_hand_hour(WR * 0.55), hand_ang, c + 10, c + 14), _rot(_hand_min(WR * 0.82), min_ang, c + 10, c + 14)):
        _poly(sh, pts, 1.0)
    sh = cv2.GaussianBlur(sh, (0, 0), 7) * 0.35
    img *= (1 - sh[..., None])
    ink = tuple(float(v) for v in INK)
    _poly(img, _rot(_hand_hour(WR * 0.55), hand_ang, c, c), ink)
    _poly(img, _rot(_hand_min(WR * 0.82), min_ang, c, c), ink)
    cv2.circle(img, (int(c), int(c)), 26, ink, -1, cv2.LINE_AA)
    cv2.circle(img, (int(c), int(c)), 9, (90, 95, 100), -1, cv2.LINE_AA)
    sa = sec / 60 * 2 * math.pi
    p1 = (sc[0] + (sr - 10) * math.sin(sa), sc[1] - (sr - 10) * math.cos(sa))
    p0 = (sc[0] - 30 * math.sin(sa), sc[1] + 30 * math.cos(sa))
    cv2.line(img, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), ink, 5, cv2.LINE_AA)
    cv2.circle(img, (int(sc[0]), int(sc[1])), 12, ink, -1, cv2.LINE_AA)
    # glass highlight
    y, x = np.mgrid[0:WS:4, 0:WS:4].astype(np.float32)
    r = np.sqrt((x - c) ** 2 + (y - c) ** 2); ang = np.arctan2(y - c, x - c)
    hl = np.clip(1 - np.abs(r - WR * 0.86) / (WR * 0.05), 0, 1) * np.clip(np.cos(ang + 2.3) * 1.6 - 0.6, 0, 1)
    hl = cv2.resize(hl, (WS, WS)) * (0.20 + glint)
    img = img + hl[..., None] * 255
    return np.clip(img, 0, 260), a

def watch_scene(t, fi, clock, cam=(0.5, 0.5, 1.0, 0.0), arc=None, stopped=False, glint=0.0, light=1.0):
    """composite watch over dark table. cam = (u,v) watch-centre on screen as fraction, scale (1 = watch fills 92% width), rot"""
    u, v, sc_, rot = cam
    bg = wood_bg()
    bgv, _ = view(bg, bg.shape[1] / 2 + (u - 0.5) * 60, bg.shape[0] / 2 + (v - 0.5) * 60, 1.5 * (0.9 + 0.1 * sc_), 0, bg=(10, 10, 10))
    img, a = watch(clock, arc, glint, stopped)
    z = W * 0.92 * sc_ / (2 * (WR + 60))
    cth, sth = math.cos(rot), math.sin(rot)
    M = np.float32([[z * cth, -z * sth, 0], [z * sth, z * cth, 0]])
    M[:, 2] = np.float32([W * u, H * v]) - M[:, :2] @ np.float32([WS / 2, WS / 2])
    wi = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_AREA if z < 1 else cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    wa = cv2.warpAffine(a, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
    # drop shadow
    shd = cv2.GaussianBlur(np.roll(np.roll(wa, int(30 * sc_), 1), int(40 * sc_), 0), (0, 0), 30 * sc_ + 1) * 0.75
    out = bgv * (1 - shd[..., None])
    out = out * (1 - wa[..., None]) + wi * wa[..., None]
    return out * light

# ================================================================= map
LON0, LON1, LAT0, LAT1 = 1.3, 7.7, 47.35, 51.75
MW = 2600
COSL = math.cos(math.radians(49.6))
MH = int(MW * (LAT1 - LAT0) / ((LON1 - LON0) * COSL))

def proj(lon, lat):
    return (lon - LON0) / (LON1 - LON0) * MW, (LAT1 - lat) / (LAT1 - LAT0) * MH

FRONT = [(3.72, 51.10), (3.80, 50.85), (3.88, 50.70), (3.95, 50.52), (4.06, 50.47), (4.12, 50.33), (4.05, 50.12),
         (4.20, 49.95), (4.55, 49.80), (4.84, 49.70), (5.10, 49.60), (5.25, 49.45), (5.42, 49.31), (5.60, 49.18),
         (5.78, 49.02), (6.05, 48.90), (6.40, 48.72), (6.80, 48.55), (7.00, 48.35), (7.05, 48.05), (7.12, 47.80), (7.17, 47.50)]
PLACES = {
    'paris': (2.35, 48.86), 'compiegne': (2.906, 49.427), 'mons': (3.95, 50.454), 'verdun': (5.38, 49.16),
    'sedan': (4.94, 49.70),
    'vrigne': (4.84, 49.704), 'ville': (4.065, 50.475), 'chaumont': (5.42, 49.31),
}

def _geo_lines(path, filt=None):
    gj = json.load(open(path)); out = []
    for ft in gj['features']:
        if filt and not filt(ft.get('properties', {})): continue
        g = ft['geometry']
        if g is None: continue
        polys = {'Polygon': [g['coordinates']], 'MultiPolygon': g['coordinates'],
                 'LineString': [[g['coordinates']]], 'MultiLineString': [[l] for l in g['coordinates']]}.get(g['type'], [])
        for poly in polys:
            for ring in poly:
                arr = np.array(ring, np.float64)
                if arr.ndim != 2 or len(arr) < 2: continue
                if arr[:, 0].max() < LON0 - 1 or arr[:, 0].min() > LON1 + 1 or arr[:, 1].max() < LAT0 - 1 or arr[:, 1].min() > LAT1 + 1:
                    continue
                if np.abs(np.diff(arr[:, 0])).max() > 30: continue
                out.append((g['type'], arr, ft.get('properties', {})))
    return out

@functools.lru_cache(None)
def map_static():
    S = 2  # supersample for land mask edges
    img = np.ones((MH, MW, 3), np.float32) * PAPER * 1.02
    stain = fbm(MH, MW, 260, 41, 4)
    img *= (0.90 + 0.12 * stain)[..., None]
    land = np.zeros((MH, MW), np.uint8)
    coast = []
    for typ, arr, _ in _geo_lines(f'{ROOT}/map/ne_10m_land.geojson'):
        lon = np.clip(arr[:, 0], LON0 - 2, LON1 + 2); lat = np.clip(arr[:, 1], LAT0 - 2, LAT1 + 2)
        x, y = proj(lon, lat)
        pts = np.stack([x, y], 1)
        cv2.fillPoly(land, [np.round(pts * 4).astype(np.int32)], 255, cv2.LINE_AA, shift=2)
        coast.append(pts)
    sea = 1 - land.astype(np.float32) / 255
    hatch = np.zeros((MH, MW), np.float32)
    for yy in range(0, MH, 11):
        cv2.line(hatch, (0, yy), (MW, yy + 40), 1.0, 1, cv2.LINE_AA)
    sea_tone = sea * 0.10 + hatch * sea * 0.35
    # coastal ripple lines
    dist = cv2.distanceTransform((sea > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
    rip = sum(np.clip(1 - np.abs(dist - k) / 1.6, 0, 1) for k in (10, 22, 38)) * sea
    sea_tone = np.clip(sea_tone + rip * 0.45, 0, 1)
    img = img * (1 - sea_tone[..., None] * 0.7) + INK * sea_tone[..., None] * 0.7
    ink = tuple(float(v) for v in INK)
    lay = np.zeros((MH, MW), np.float32)
    rng = np.random.default_rng(3)
    for pts in coast:
        j = pts + rng.normal(0, 0.6, pts.shape)
        cv2.polylines(lay, [np.round(j * 4).astype(np.int32)], True, 1.0, 4, cv2.LINE_AA, shift=2)
    # rivers
    rlay = np.zeros((MH, MW), np.float32)
    for path in (f'{ROOT}/map/ne_10m_rivers_europe.geojson', f'{ROOT}/map/ne_10m_rivers_lake_centerlines.geojson'):
        for typ, arr, pr in _geo_lines(path):
            x, y = proj(arr[:, 0], arr[:, 1])
            cv2.polylines(rlay, [np.round(np.stack([x, y], 1) * 4).astype(np.int32)], False, 1.0, 2, cv2.LINE_AA, shift=2)
    rlay = np.clip(rlay, 0, 1) * 0.55
    lay = np.clip(lay + rlay, 0, 1) * (0.8 + 0.2 * fbm(MH, MW, 6, 44, 2))
    img = img * (1 - lay[..., None]) + INK * lay[..., None]
    # labels
    pil = Image.new('L', (MW, MH), 0); d = ImageDraw.Draw(pil)
    def lab(txt, lon, lat, size, wght=600, track=6):
        x, y = proj(lon, lat); f = font('serif', size, wght)
        tw = sum(f.getlength(ch) + track for ch in txt) - track
        xx = x - tw / 2
        for ch in txt:
            d.text((xx, y - size / 2), ch, font=f, fill=255); xx += f.getlength(ch) + track
    lab('ΓΑΛΛΙΑ', 3.3, 48.35, 84, 600, 26)
    lab('ΒΕΛΓΙΟ', 4.75, 50.75, 72, 600, 22)
    lab('ΓΕΡΜΑΝΙΑ', 6.9, 49.95, 72, 600, 22)
    lab('ΒΟΡΕΙΑ ΘΑΛΑΣΣΑ', 2.6, 51.55, 46, 500, 12)
    lab('Παρίσι', 2.35, 48.76, 50, 500, 2)
    lab('Μονς', 3.85, 50.56, 46, 500, 2)
    lab('Βερντέν', 5.38, 49.06, 46, 500, 2)
    lab('Σεντάν', 5.12, 49.80, 40, 500, 2)
    lab('ποταμός Μεύση', 4.55, 49.42, 34, 400, 2)
    la = np.asarray(pil).astype(np.float32) / 255
    img = img * (1 - la[..., None] * 0.9) + INK * la[..., None] * 0.9
    for k in ('paris', 'mons', 'verdun', 'sedan'):
        x, y = proj(*PLACES[k]); cv2.circle(img, (int(x), int(y)), 9, ink, -1, cv2.LINE_AA)
    # folds
    for fx in (MW / 3, 2 * MW / 3):
        g = np.exp(-((np.arange(MW) - fx) / 6) ** 2)[None, :, None]
        img *= (1 - 0.10 * g)
    for fy in (MH / 2,):
        g = np.exp(-((np.arange(MH) - fy) / 6) ** 2)[:, None, None]
        img *= (1 - 0.10 * g)
    return img

def front_layer(p):
    """animated front line (0..1 drawn)."""
    pts = np.array([proj(*q) for q in FRONT], np.float32)
    seg = np.hypot(*np.diff(pts, axis=0).T); L = seg.sum(); target = L * p
    lay = np.zeros((MH, MW), np.float32)
    acc = 0.0; drawn = [pts[0]]
    for i, sl in enumerate(seg):
        if acc + sl <= target:
            drawn.append(pts[i + 1]); acc += sl
        else:
            u = (target - acc) / sl; drawn.append(pts[i] + (pts[i + 1] - pts[i]) * u); break
    drawn = np.array(drawn)
    if len(drawn) > 1:
        cv2.polylines(lay, [np.round(drawn * 4).astype(np.int32)], False, 1.0, 13, cv2.LINE_AA, shift=2)
        # teeth toward the east (German side)
        dd = np.hypot(*np.diff(drawn, axis=0).T); tot = 0
        for i in range(len(drawn) - 1):
            a_, b_ = drawn[i], drawn[i + 1]; l = dd[i]
            if l < 1: continue
            dv = (b_ - a_) / l; nv = np.array([-dv[1], dv[0]])
            if nv[0] < 0: nv = -nv
            s = (-tot) % 34
            while s < l:
                q = a_ + dv * s
                cv2.line(lay, tuple(np.int32(q)), tuple(np.int32(q + nv * 22)), 1.0, 5, cv2.LINE_AA)
                s += 34
            tot += l
    return lay

def map_frame(t, fi, cam, front=1.0, marks=(), compi=0.0, label_alpha=1.0):
    """cam = (lon, lat, zoom); marks = list of (key, strength 0..1, label lines)."""
    lon, lat, zoom = cam
    base = map_static().copy()
    if front > 0:
        fl = front_layer(front)
        base = base * (1 - fl[..., None] * 0.85) + RED * fl[..., None] * 0.85
    cx, cy = proj(lon, lat)
    out, M = view(base, cx, cy, zoom)
    k = W * zoom / MW
    if compi > 0:
        x, y = to_screen(M, *proj(*PLACES['compiegne']))
        r = 16 + 10 * math.sin(t * 4)
        cv2.circle(out, (int(x), int(y)), int(r * clamp01(compi * 2)), tuple(float(v) for v in INK), 5, cv2.LINE_AA)
        cv2.circle(out, (int(x), int(y)), 9, tuple(float(v) for v in INK), -1, cv2.LINE_AA)
        al = text_layer(['Κομπιένη'], 'serif', 52, 600)
        out = paste_alpha(out, al, x, y - 60, INK, compi)
    for key, s, lines in marks:
        if s <= 0: continue
        x, y = to_screen(M, *proj(*PLACES[key]))
        pulse = (t * 0.9) % 1.0
        ring = np.zeros((H, W), np.float32)
        for ph in (pulse, (pulse + 0.5) % 1):
            cv2.circle(ring, (int(x), int(y)), int(18 + 90 * ph), float(1 - ph), 4, cv2.LINE_AA)
        ring = ring[..., None] * clamp01(s * 2 - 0.4)
        out = out * (1 - ring) + RED * ring
        cv2.circle(out, (int(x), int(y)), int(16 * ease(s * 1.5)), tuple(float(v) for v in RED), -1, cv2.LINE_AA)
        cv2.line(out, (int(x), int(y) - 10), (int(x), int(y) + 10), (220, 225, 230), 3, cv2.LINE_AA)
        cv2.line(out, (int(x) - 7, int(y) - 3), (int(x) + 7, int(y) - 3), (220, 225, 230), 3, cv2.LINE_AA)
        if lines:
            al = text_layer(lines, 'serif', 54, 600, 1.15)
            out = paste_alpha(out, al, x, y + 46 + al.shape[0] / 2, INK, ease(s * 1.4 - 0.3) * label_alpha)
    return out

# ================================================================= handwritten note
NOTE_LINES = ['Rassemblement', 'à 11 h 30', 'pour le ravitaillement']

@functools.lru_cache(None)
def note_paper(w=1500, h=980):
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    base = PAPER * 1.03 * (0.88 + 0.12 * fbm(h, w, 80, 61, 4))[..., None]
    # foxing
    fox = np.clip((fbm(h, w, 40, 62, 3) - 0.62) * 4, 0, 1)
    base = base * (1 - 0.18 * fox[..., None]) + np.array([90, 120, 150], np.float32) * 0.18 * fox[..., None]
    # folds
    for fx in (w * 0.5,):
        g = np.exp(-((x - fx) / 5) ** 2); base *= (1 - 0.12 * g[..., None])
    g = np.exp(-((y - h * 0.5) / 5) ** 2); base *= (1 - 0.08 * g[..., None])
    # ruled lines
    for yy in np.arange(h * 0.16, h * 0.95, h * 0.13):
        cv2.line(base, (40, int(yy)), (w - 40, int(yy)), (190, 170, 150), 1, cv2.LINE_AA)
    # torn edge alpha
    e = fbm(h, w, 14, 63, 3)
    d = np.minimum.reduce([x, y, w - 1 - x, h - 1 - y])
    a = np.clip((d - 10 - e * 28) / 4, 0, 1)
    return base, a

@functools.lru_cache(None)
def note_text():
    w, h = 1500, 980
    pil = Image.new('L', (w, h), 0); d = ImageDraw.Draw(pil)
    f = font('hand', 132, 500)
    boxes = []
    y = 70
    for i, ln in enumerate(NOTE_LINES):
        x = 110 + (i * 40 if i else 0)
        bb = d.textbbox((x, y), ln, font=f)
        d.text((x, y), ln, font=f, fill=255)
        boxes.append(bb); y += 250
    return np.asarray(pil).astype(np.float32) / 255, boxes

def note_image(p):
    """note paper with handwriting written up to progress p (0..1). returns bgr, alpha, pen(x,y)|None"""
    base, a = note_paper()
    txt, boxes = note_text()
    lens = np.array([b[2] - b[0] for b in boxes], np.float32); tot = lens.sum()
    target = p * tot; mask = np.zeros_like(txt); pen = None; acc = 0
    for (x0, y0, x1, y1), L in zip(boxes, lens):
        if target >= acc + L:
            mask[max(0, y0 - 40):y1 + 40, :] = 1
        elif target > acc:
            xh = x0 + (target - acc)
            mask[max(0, y0 - 40):y1 + 40, :int(xh)] = 1
            cols = txt[y0:y1, max(0, int(xh) - 6):int(xh) + 1].sum(1)
            yy = y0 + (int(np.argmax(cols)) if cols.max() > 0 else (y1 - y0) // 2)
            pen = (xh, yy)
        acc += L
    m = cv2.GaussianBlur(mask, (0, 0), 3)
    ink = txt * m * (0.82 + 0.18 * fbm(*txt.shape, 4, 64, 2))
    col = np.array([60, 40, 30], np.float32)  # iron-gall brown-black
    out = base * (1 - ink[..., None]) + col * ink[..., None]
    return out, a, pen

# ================================================================= crosses counter
@functools.lru_cache(None)
def cross_field(n=2738, seed=8):
    rng = np.random.default_rng(seed)
    pts = []
    rows = 60
    for r in range(rows):
        z = 1.0 + r * 0.22                     # depth
        yy = H * 0.50 + 900 / z                  # horizon at ~0.50H
        cnt = int(10 + r * 3.4)
        span = 1700 / z * (1 + r * 0.05)
        for i in range(cnt):
            xx = W / 2 + (i / (cnt - 1) - 0.5) * span * 2.2 + rng.normal(0, 4 / z)
            pts.append((xx, yy + rng.normal(0, 2 / z), 1 / z))
    pts = np.array(pts)
    inside = (pts[:, 0] > -20) & (pts[:, 0] < W + 20)
    pts = pts[inside]
    order = np.argsort(-pts[:, 2] + rng.normal(0, 0.05, len(pts)))  # near first
    pts = pts[order][:n]
    return pts

def counter_frame(t, fi, p, label=1.0):
    """p: 0..1 progress of the count"""
    bg = wood_bg()
    out, _ = view(bg, bg.shape[1] / 2, bg.shape[0] / 2, 1.5)
    out = out * 0.65
    pts = cross_field()
    n = int(len(pts) * ease2(p))
    lay = np.zeros((H, W), np.float32)
    for x, y, s in pts[:n][::-1]:
        hgt = 46 * s; arm = 15 * s; th = max(1, int(round(4 * s)))
        cv2.line(lay, (int(x), int(y - hgt)), (int(x), int(y)), 1.0, th, cv2.LINE_AA)
        cv2.line(lay, (int(x - arm), int(y - hgt * 0.68)), (int(x + arm), int(y - hgt * 0.68)), 1.0, th, cv2.LINE_AA)
    lay = np.clip(lay, 0, 1) * 0.85
    out = out * (1 - lay[..., None]) + PAPER * 0.92 * lay[..., None]
    num = int(round(2738 * ease2(p)))
    s = f'{num:,}'.replace(',', '.')
    al = text_layer([s], 'serif', 260, 600)
    out = paste_alpha(out, al, W / 2, H * 0.24, PAPER * 1.05, 1.0, shadow=0.8)
    if label > 0:
        al2 = text_layer(['νεκροί', 'την τελευταία μέρα του πολέμου'], 'serif', 62, 500, 1.2)
        out = paste_alpha(out, al2, W / 2, H * 0.24 + 230, PAPER, label, shadow=0.6)
    return out

# ================================================================= grave plaque
PLAQUE = ['Augustin TRÉBUCHON', 'MORT POUR LA FRANCE', 'LE 10 NOVEMBRE 1918']

@functools.lru_cache(None)
def plaque_img(w=1500, h=860):
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    base = PAPER * 1.04 * (0.9 + 0.1 * fbm(h, w, 60, 71, 4))[..., None]
    rust = np.clip((fbm(h, w, 30, 72, 4) - 0.6) * 3, 0, 1)
    base = base * (1 - 0.3 * rust[..., None]) + np.array([40, 70, 110], np.float32) * 0.3 * rust[..., None]
    a = np.zeros((h, w), np.float32)
    cv2.rectangle(a, (20, 20), (w - 20, h - 20), 1.0, -1)
    a = cv2.GaussianBlur(a, (0, 0), 3)
    cv2.rectangle(base, (48, 48), (w - 48, h - 48), tuple(float(v) for v in INK), 6, cv2.LINE_AA)
    for (px, py) in ((80, 80), (w - 80, 80), (80, h - 80), (w - 80, h - 80)):
        cv2.circle(base, (px, py), 14, (70, 74, 78), -1, cv2.LINE_AA)
    pil = Image.new('L', (w, h), 0); d = ImageDraw.Draw(pil)
    sizes = [96, 84, 84]; ys = [140, 320, 610]
    boxes = []
    for ln, sz, yy in zip(PLAQUE, sizes, ys):
        f = font('serif', sz, 600)
        bb = d.textbbox((0, 0), ln, font=f); tw = bb[2] - bb[0]
        d.text(((w - tw) / 2 - bb[0], yy), ln, font=f, fill=255)
        boxes.append(d.textbbox(((w - tw) / 2 - bb[0], yy), ln, font=f))
    # locate '10' in last line
    f = font('serif', 84, 600); ln = PLAQUE[2]
    x_start = boxes[2][0]
    pre = d.textlength('LE ', font=f); ten = d.textlength('10', font=f)
    ten_box = (x_start + pre, boxes[2][1], x_start + pre + ten, boxes[2][3])
    t = np.asarray(pil).astype(np.float32) / 255 * (0.8 + 0.2 * fbm(h, w, 4, 73, 2))
    base = base * (1 - t[..., None]) + INK * t[..., None]
    return base, a, ten_box

def plaque_frame(t, fi, bgimg, p_in, p_strike, p_eleven, scale=1.0):
    base, a, tb = plaque_img()
    img = base.copy(); h, w = a.shape
    if p_strike > 0:
        lay = np.zeros((h, w), np.float32)
        x0, y0, x1, y1 = tb; ym = (y0 + y1) / 2 + 6
        xe = x0 - 20 + (x1 - x0 + 40) * ease(p_strike)
        cv2.line(lay, (int(x0 - 20), int(ym + 8)), (int(xe), int(ym - 8)), 1.0, 14, cv2.LINE_AA)
        if p_strike > 0.5:
            xe2 = x0 - 20 + (x1 - x0 + 40) * ease((p_strike - 0.5) * 2)
            cv2.line(lay, (int(x0 - 20), int(ym - 10)), (int(xe2), int(ym + 12)), 1.0, 12, cv2.LINE_AA)
        lay *= 0.85 + 0.15 * fbm(h, w, 5, 74, 2)
        img = img * (1 - lay[..., None]) + RED * lay[..., None]
    if p_eleven > 0:
        al = text_layer(['11'], 'hand', 170, 700)
        m = np.zeros_like(al); cut = int(al.shape[1] * ease(p_eleven))
        m[:, :cut] = 1; al = al * cv2.GaussianBlur(m, (0, 0), 4)
        cx = (tb[0] + tb[2]) / 2; cy = tb[1] - 70
        img = paste_alpha_local(img, al, cx, cy, RED)
    sc = 0.62 * scale * (0.92 + 0.08 * ease(p_in))
    ang = -0.035 + 0.01 * math.sin(t * 0.4)
    c, s = math.cos(ang) * sc, math.sin(ang) * sc
    M = np.float32([[c, -s, 0], [s, c, 0]]); M[:, 2] = np.float32([W / 2, H * 0.45]) - M[:, :2] @ np.float32([w / 2, h / 2])
    pi = cv2.warpAffine(img, M, (W, H)); pa = cv2.warpAffine(a, M, (W, H)) * ease(p_in)
    shd = cv2.GaussianBlur(np.roll(pa, 30, 0), (0, 0), 25) * 0.7
    out = bgimg * (1 - shd[..., None])
    return out * (1 - pa[..., None]) + pi * pa[..., None]

def paste_alpha_local(img, alpha, cx, cy, color):
    h, w = alpha.shape; ih, iw = img.shape[:2]
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    xs0, ys0 = max(0, x0), max(0, y0); xs1, ys1 = min(iw, x0 + w), min(ih, y0 + h)
    a = alpha[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0][..., None]
    img[ys0:ys1, xs0:xs1] = img[ys0:ys1, xs0:xs1] * (1 - a) + np.array(color, np.float32) * a
    return img
