"""Ep05 shots. frame(t) -> BGR uint8. Times come from $MT_ROOT/timeline.json (beats)."""
import os, sys, json, math, functools, datetime
from lib05 import *
import gfx

TL = json.load(open(f'{ROOT}/timeline.json'))
def b0(k): return TL[k][0]
def b1(k): return TL[k][1]

FILM = Film(seed=5)

# ======================================================================= desk + cards (hook)
@functools.lru_cache(None)
def desk_bg():
    h, w = H, W
    v = 0.10 + 0.07 * cv2.resize(fbm(h // 8, w, 30, 9, 4), (w, h)) + 0.05 * fbm(h, w, 90, 10, 4)
    Y, X = np.mgrid[0:h, 0:w].astype(np.float32)
    spot = np.exp(-(((X - w * 0.48) / (w * 0.75)) ** 2 + ((Y - h * 0.44) / (h * 0.55)) ** 2))
    v = v * (0.45 + 0.85 * spot)
    return np.clip(v[..., None] * PAPER * 1.05, 0, 255).astype(np.float32)

CW, CH = 900, 690

def local_strokes(shape, strokes, p=1.0):
    """render canvas-unit strokes into a local raster (S px/unit, origin 0,0)."""
    buf = np.zeros(shape, np.float32)
    class _C: z = S
    c = _C()
    c.pts = lambda P: np.asarray(P, np.float32) * S
    draw_strokes(buf, strokes, p, c)
    return np.clip(cv2.GaussianBlur(buf, (0, 0), 0.5) * (0.85 + 0.15 * fbm(*shape, 10, 3, 2)), 0, 1)

@functools.lru_cache(None)
def hidden_card(letter, seed):
    """card with its claim redacted (hook)."""
    rgb, a = card_paper(CW, CH, seed)
    ring = local_strokes(a.shape, [Stroke([(70 + 52 * math.cos(q), 92 + 52 * math.sin(q)) for q in np.linspace(0.25, 2 * math.pi + 0.5, 70)], 4.2, 0.6, seed)])
    put_alpha(rgb, ring, 0, 0, INK)
    L = Text(letter, F_SERIF, 74, wght=700, typed=False)
    put_text_local(rgb, L, 70, 92 + 26, INK, align='center')
    r = np.random.default_rng(seed)
    y = 52
    for row, frac in enumerate((0.92, 0.80, 0.55)):
        x0, x1 = 150, 150 + 700 * frac
        bar = np.zeros(a.shape, np.float32)
        cv2.rectangle(bar, (int(x0 * S), int(y * S)), (int(x1 * S), int((y + 34) * S)), 1.0, -1)
        hh, ww = bar.shape
        bar = bar * (0.82 + 0.18 * fbm(hh, ww, 7 * S, seed + row, 2))
        put_alpha(rgb, bar, 0, 0, INK, 0.92)
        y += 54
    rule = local_strokes(a.shape, [Stroke([(40, 236), (CW - 40, 236)], 1.8, 0.5, seed + 9)])
    put_alpha(rgb, rule, 0, 0, INK, 0.7)
    q = Text('?', F_SERIF, 300, wght=400, typed=False)
    put_text_local(rgb, q, CW / 2, 590, INK, 0.10, align='center')
    return rgb, a

def card_pose_final(which):
    return (534, 640, math.radians(1.3)) if which == 'A' else (546, 1330, math.radians(-1.0))

def hook(t):
    cam = Cam(W / 2, H / 2, 1.0)
    img = desk_bg().copy()
    tA0, tB0 = b0('hook1') - 0.25, b0('hook1') + 0.05
    poses = {}
    for which, t0, frm in (('A', tA0, (-700, 420, -0.30)), ('B', tB0, (1800, 1560, 0.26))):
        x, y, r = card_pose_final(which)
        u = spring(max(0, t - t0) / 0.85, 1.1, 5.5)
        px, py, pr = lerp(frm[0], x, u), lerp(frm[1], y, u), lerp(frm[2], r, u)
        lift, sc = 0.6, 1.0
        # A: 'η μία συνέβη πραγματικά' lift; B: 'η άλλη... όχι' lift + head-shake
        la = math.sin(math.pi * prog(t, b0('hook3'), b1('hook3') + 0.2)) if which == 'A' else math.sin(math.pi * prog(t, b0('hook4'), b1('hook4') + 0.25))
        lift += 1.4 * la; sc += 0.025 * la
        if which == 'B':
            k = prog(t, b1('hook4') - 0.45, b1('hook4') + 0.15)
            pr += 0.035 * math.sin(k * 6 * math.pi) * (1 - k)
        poses[which] = (px, py, pr, sc, lift)
    for which in ('B', 'A') if t > b0('hook4') - 0.1 and t < b1('hook4') + 0.3 else ('A', 'B'):
        px, py, pr, sc, lift = poses[which]
        rgb, a = hidden_card('Α' if which == 'A' else 'Β', 3 if which == 'A' else 7)
        blit_rgba(img, rgb, a, cam, px, py, 1.0, sc, pr, (CW / 2, CH / 2), shadow=0.8, lift=lift)
    # titles
    p = prog(t, b0('hook2'), b0('hook2') + 1.3)
    T1 = Text('ΜΙΑ ΑΛΗΘΕΙΑ  ·  ΕΝΑ ΨΕΜΑ', F_SERIF, 40, wght=500, track=5, typed=False)
    n = int(len(T1.txt) * p)
    text_at(img, cam, T1, W / 2, 112, C(236, 226, 210), 0.85, n=n, align='center')
    p2 = eout(prog(t, b0('hook5') - 0.1, b0('hook5') + 0.7))
    if p2 > 0:
        T2 = _T('Θα βρεις ποια είναι ποια;', F_SERIF, 66, 500)
        text_at(img, cam, T2, W / 2, 190 + 18 * (1 - p2), C(240, 232, 218), p2, align='center')
    return img

@functools.lru_cache(None)
def _T(txt, path, size, wght=None, track=0.0, typed=False, seed=0, index=0):
    return Text(txt, path, size, wght=wght, track=track, typed=typed, seed=seed, index=index)

# ======================================================================= story A scroll
YA = {'map': 1000, 'draft': 3000, 'home': 5000, 'road': 7000, 'watch': 9000, 'end': 11000}
PAPER_A = None
def paper_a():
    global PAPER_A
    if PAPER_A is None: PAPER_A = PaperStrip(12000, seed=11)
    return PAPER_A

# ---------------- map of western Japan
LON0, LON1, LAT0, LAT1 = 129.2, 133.6, 31.0, 35.4
MX0, MY0 = 60, YA['map'] - 560
KX = 960 / (LON1 - LON0); KY = KX / math.cos(math.radians(33.3))
def proj(lon, lat): return MX0 + (lon - LON0) * KX, MY0 + (LAT1 - lat) * KY
MAP_W, MAP_H = 960, (LAT1 - LAT0) * KY
HIRO = proj(132.455, 34.385); NAGA = proj(129.877, 32.750)

@functools.lru_cache(None)
def japan():
    cache = f'{ROOT}/map/japan_w.npz'
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True); return list(z['runs']), list(z['rings'])
    gj = json.load(open(f'{ROOT}/map/ne_10m_land.geojson'))
    runs, rings = [], []
    m = 0.4
    for f in gj['features']:
        g = f['geometry']
        polys = g['coordinates'] if g['type'] == 'MultiPolygon' else [g['coordinates']]
        for poly in polys:
            for ring in poly[:1]:
                R = np.array(ring)
                inb = (R[:, 0] > LON0 - m) & (R[:, 0] < LON1 + m) & (R[:, 1] > LAT0 - m) & (R[:, 1] < LAT1 + m)
                if not inb.any(): continue
                P = np.stack(proj(R[:, 0], R[:, 1]), 1)
                rings.append(P[inb] if len(R) > 200000 else P)
                # split into inside runs
                idx = np.where(inb)[0]
                br = np.where(np.diff(idx) > 1)[0]
                for seg in np.split(idx, br + 1):
                    if len(seg) < 4: continue
                    Q = P[seg]
                    L = np.hypot(*np.diff(Q, axis=0).T).sum()
                    if L < 14: continue
                    runs.append(Q)
    np.savez(cache, runs=np.array(runs, dtype=object), rings=np.array(rings, dtype=object))
    return runs, rings

@functools.lru_cache(None)
def coast_strokes():
    runs, _ = japan()
    out = []
    for i, Q in enumerate(runs):
        L = np.hypot(*np.diff(Q, axis=0).T).sum()
        out.append(Stroke(Q, 2.4 if L > 80 else 1.6, 0.6, 100 + i, taper=False))
    # draw the big islands first: sort by distance of centroid to Hiroshima
    out.sort(key=lambda s: -s.len)
    return out

@functools.lru_cache(None)
def map_layers():
    """land wash + waterlining, rasters in S px/unit over the map box."""
    _, rings = japan()
    R = Raster(MX0 - 40, MY0 - 40, MAP_W + 80, MAP_H + 80)
    for P in rings:
        if len(P) > 2: R.poly(P)
    land = R.alpha()
    wsh = wash(land, seed=4, edge=0.25, gran=0.3)
    sea = 1 - (land > 0.5).astype(np.uint8)
    d = cv2.distanceTransform(sea, cv2.DIST_L2, 5) / S
    wl = np.zeros_like(land)
    for dist, op in ((7, 0.75), (14, 0.5), (22, 0.32), (31, 0.2)):
        wl = np.maximum(wl, np.clip(1.2 - np.abs(d - dist) * S * 0.6, 0, 1) * op)
    # fade waterlines out at the map border
    hh, ww = land.shape
    Y, X = np.mgrid[0:hh, 0:ww].astype(np.float32) / S
    border = np.clip(np.minimum.reduce([X - 40, Y - 40, MAP_W + 40 - X, MAP_H + 40 - Y]) / 40, 0, 1)
    return wsh * border, wl * border, (R.x0, R.y0)

def draw_map(img, cam, t):
    t0 = b0('a_japan') - 0.15; t1 = b1('a_date') + 0.2
    p = eout(prog(t, t0, t1))
    wsh, wl, (x0, y0) = map_layers()
    pw = prog(t, t0 + 0.45 * (t1 - t0), t1 + 0.4)
    if pw > 0:
        blit(img, wsh, cam, x0, y0, PAPER_D, 0.55 * ease(pw), mode='multiply')
        blit(img, wl, cam, x0, y0, SEA, 0.85 * ease(pw), mode='multiply')
        blit(img, wsh, cam, x0, y0, SEA, 0.0)
    buf = np.zeros((H, W), np.float32)
    draw_strokes(buf, coast_strokes(), p, cam, tip=True)
    paint(img, inkify(buf), INK, 0.92)
    # frame + title
    fr = [Stroke([(40, MY0 - 20), (1040, MY0 - 20)], 1.6, 0.4, 7), Stroke([(40, MY0 + MAP_H + 20), (1040, MY0 + MAP_H + 20)], 1.6, 0.4, 8)]
    buf = np.zeros((H, W), np.float32); draw_strokes(buf, fr, eout(prog(t, t0, t0 + 1.0)), cam)
    paint(img, inkify(buf), INK, 0.6)
    T = _T('ΙΑΠΩΝΙΑ', F_SERIF, 92, 600, 22)
    k = prog(t, b0('a_japan') - 0.1, b0('a_japan') + 0.5)
    text_at(img, cam, T, 540, MY0 - 170 + 14 * (1 - eout(k)), INK, eout(k), align='center')
    D = _T('6 · ΑΥΓΟΥΣΤΟΥ · 1945', F_MONO, 44, None, 2, True, 21)
    text_at(img, cam, D, 540, MY0 - 82, INK, 0.92, n=type_n(D, t, b0('a_date') + 0.05, 16), align='center')
    # cities
    for (x, y), t_on, kanji, gr, side in ((HIRO, b0('a_date') + 0.7, '広島', 'Χιροσίμα', 1), (NAGA, b1('a_date') - 0.2, '長崎', 'Ναγκασάκι', -1)):
        k = prog(t, t_on, t_on + 0.5)
        if k <= 0: continue
        dot = np.zeros((H, W), np.float32)
        sx, sy = cam.pt(x, y)
        cv2.circle(dot, (int(sx * 8), int(sy * 8)), int(7 * cam.z * back(k) * 8), 1.0, -1, cv2.LINE_AA, shift=3)
        cv2.circle(dot, (int(sx * 8), int(sy * 8)), int(13 * cam.z * back(k) * 8), 1.0, max(1, int(1.6 * cam.z)), cv2.LINE_AA, shift=3)
        # pulse ring
        pk = (t - t_on) % 1.6 / 1.6
        if t - t_on > 0.4:
            cv2.circle(dot, (int(sx * 8), int(sy * 8)), int((14 + 40 * pk) * cam.z * 8), 1.0 - pk, 1, cv2.LINE_AA, shift=3)
        paint(img, dot, INK, 0.95)
        lx = x + side * 34; ly = y + 12
        KJ = _T(kanji, F_CJK, 42, None, 0, False, 0, 0)
        GR = _T(gr, F_SERIF_I, 36, 500)
        al = 'left'
        if side < 0:
            text_at(img, cam, KJ, lx - KJ.ink_w, ly - 6, INK, eout(k))
            text_at(img, cam, GR, lx - GR.ink_w, ly + 40, INK, eout(k) * 0.9)
        else:
            text_at(img, cam, KJ, lx, ly - 6, INK, eout(k))
            text_at(img, cam, GR, lx, ly + 40, INK, eout(k) * 0.9)
    # north arrow
    k = eout(prog(t, b1('a_date') - 0.4, b1('a_date') + 0.4))
    if k > 0:
        nx, ny = 960, MY0 + 60
        st = [Stroke([(nx, ny + 60), (nx, ny - 30)], 2.0, 0.3, 3), Stroke([(nx - 12, ny - 8), (nx, ny - 32), (nx + 12, ny - 8)], 2.0, 0.3, 4)]
        buf = np.zeros((H, W), np.float32); draw_strokes(buf, st, k, cam); paint(img, inkify(buf), INK, 0.8)
        N = _T('Β', F_SERIF, 30, 600)
        text_at(img, cam, N, nx, ny + 98, INK, k * 0.8, align='center')

# ---------------- drafting table: tanker blueprint, ID tag, calendar
SHEET_W, SHEET_H = 1000, 640
def tanker_strokes(x0, y0, L=700, wk=1.7):
    """1940s tanker profile (bridge amidships, engine + funnel aft), bow to the right. local units."""
    H_ = L * 0.085
    def P(pts): return [(x0 + u * L, y0 + v * H_) for u, v in pts]
    s = []
    s.append(Stroke(P([(0.0, 0.0), (0.02, 0.9), (0.06, 1.15), (0.94, 1.15), (0.985, 0.7), (1.0, -0.15)]), 2.2, 0.35, 1))
    s.append(Stroke(P([(-0.005, 0.0), (0.25, 0.08), (0.5, 0.1), (0.75, 0.06), (1.0, -0.15)]), 2.0, 0.35, 2))   # sheer / deck
    s.append(Stroke(P([(0.03, 0.62), (0.97, 0.62)]), 1.2, 0.2, 3))                                                # waterline
    s.append(Stroke(P([(0.44, 0.08), (0.44, -0.9), (0.56, -0.9), (0.56, 0.1)]), 1.8, 0.3, 4))                     # bridge
    s.append(Stroke(P([(0.45, -0.9), (0.46, -1.35), (0.54, -1.35), (0.55, -0.9)]), 1.6, 0.3, 5))
    s.append(Stroke(P([(0.03, 0.02), (0.03, -0.75), (0.2, -0.75), (0.2, 0.08)]), 1.8, 0.3, 6))                    # aft house
    s.append(Stroke(P([(0.09, -0.75), (0.085, -1.9), (0.15, -1.9), (0.155, -0.75)]), 1.8, 0.3, 7))                # funnel
    s.append(Stroke(P([(0.3, 0.09), (0.3, -2.6)]), 1.3, 0.2, 8))                                                  # masts
    s.append(Stroke(P([(0.78, 0.06), (0.78, -2.6)]), 1.3, 0.2, 9))
    s.append(Stroke(P([(0.2, -0.35), (0.44, -0.35)]), 0.9, 0.1, 10))                                              # catwalks
    s.append(Stroke(P([(0.56, -0.35), (0.95, -0.35)]), 0.9, 0.1, 11))
    for k in range(1, 20):                                                                                         # tank bulkheads
        u = 0.21 + k * 0.037
        if 0.44 < u < 0.56: continue
        s.append(Stroke(P([(u, 0.2), (u, 1.05)]), 0.8, 0.1, 20 + k))
    # dimension line
    s.append(Stroke(P([(0.0, 1.9), (1.0, 1.9)]), 0.9, 0.1, 50))
    s.append(Stroke(P([(0.0, 1.6), (0.0, 2.2)]), 0.9, 0.1, 51)); s.append(Stroke(P([(1.0, 1.6), (1.0, 2.2)]), 0.9, 0.1, 52))
    s.append(Stroke(P([(0.02, 1.8), (0.0, 1.9), (0.02, 2.0)]), 0.9, 0.1, 53)); s.append(Stroke(P([(0.98, 1.8), (1.0, 1.9), (0.98, 2.0)]), 0.9, 0.1, 54))
    for st in s: st.w *= wk
    return s

@functools.lru_cache(None)
def sheet_base():
    rgb, a = card_paper(SHEET_W, SHEET_H, 31, base=C(230, 228, 218), edge=0.10)
    grid = np.zeros(a.shape, np.float32)
    for x in range(20, SHEET_W, 20):
        grid[:, int(x * S)] = 1.0
    for y in range(20, SHEET_H, 20):
        grid[int(y * S), :] = 1.0
    put_alpha(rgb, grid, 0, 0, C(150, 170, 190), 0.25)
    # title block
    tb = local_strokes(a.shape, [Stroke([(640, 530), (960, 530), (960, 610), (640, 610), (640, 530)], 1.8, 0.2, 2),
                                 Stroke([(640, 568), (960, 568)], 1.2, 0.1, 3)])
    put_alpha(rgb, tb, 0, 0, INK_BLUE, 0.9)
    put_text_local(rgb, _T('三菱重工業', F_CJK, 28, None, 3, False, 0, 0), 656, 560, INK_BLUE)
    put_text_local(rgb, _T('1945', F_MONO, 28, None, 1, True, 5), 656, 600, INK_BLUE)
    return rgb, a

def draft_sheet_rgba(t):
    rgb, a = sheet_base(); rgb = rgb.copy()
    p = eout(prog(t, b0('a_who') + 0.2, b1('a_trip') - 0.6))
    lay = local_strokes(a.shape, tanker_strokes(70, 300, 860), p)
    put_alpha(rgb, lay, 0, 0, INK_BLUE, 0.95)
    return rgb, a

@functools.lru_cache(None)
def id_tag():
    w, h = 520, 270
    rgb, a = card_paper(w, h, 41, base=C(218, 196, 150), edge=0.2)
    hole = Raster(0, 0, w, h).ell((40, h / 2), (13, 13)).alpha()
    a = a * (1 - hole)
    ring = local_strokes(a.shape, [Stroke([(40 + 19 * math.cos(q), h / 2 + 19 * math.sin(q)) for q in np.linspace(0, 2 * math.pi, 40)], 1.4, 0.1, 1)])
    put_alpha(rgb, ring, 0, 0, C(120, 100, 70), 0.7)
    return rgb, a

def id_tag_rgba(t):
    rgb, a = id_tag(); rgb = rgb.copy()
    t0 = b0('a_who')
    k = prog(t, t0 + 0.3, t0 + 0.8)
    put_text_local(rgb, _T('山口 彊', F_CJK, 64, None, 6, False, 0, 0), 90, 100, INK, eout(k))
    N1 = _T('Τσουτόμου Γιαμαγκούτσι', F_MONO, 28, None, 0, True, 3)
    put_text_local(rgb, N1, 92, 160, INK, n=type_n(N1, t, t0 + 0.5, 20))
    N2 = _T('μηχανικός · 29 ετών', F_MONO, 28, None, 0, True, 4)
    put_text_local(rgb, N2, 92, 208, INK, n=type_n(N2, t, t0 + 1.7, 20))
    return rgb, a

MONTHS = [(5, 'ΜΑΪΟΣ'), (6, 'ΙΟΥΝΙΟΣ'), (7, 'ΙΟΥΛΙΟΣ'), (8, 'ΑΥΓΟΥΣΤΟΣ')]
CAL_W, CAL_H = 330, 360

@functools.lru_cache(None)
def cal_page(m, name):
    rgb, a = card_paper(CAL_W, CAL_H, 60 + m, base=C(234, 226, 210), edge=0.12)
    band = np.zeros(a.shape, np.float32); band[:int(78 * S)] = 1
    put_alpha(rgb, band * a, 0, 0, C(64, 58, 54), 0.92)
    put_text_local(rgb, _T(name, F_SERIF, 40, 600, 4), CAL_W / 2, 56, C(236, 228, 214), align='center')
    put_text_local(rgb, _T('1945', F_MONO, 24, None, 3, True, m), CAL_W / 2, 112, INK, align='center')
    days = 'ΔΤΤΠΠΣΚ'
    for i, d in enumerate(days):
        put_text_local(rgb, _T(d, F_MONO, 18, None, 0, True, 90 + i), 30 + i * 45 + 8, 148, INK, 0.7)
    first = datetime.date(1945, m, 1).weekday()
    ndays = (datetime.date(1945, m % 12 + 1, 1) - datetime.timedelta(days=1)).day
    for d in range(1, ndays + 1):
        k = first + d - 1; col, row = k % 7, k // 7
        put_text_local(rgb, _T(str(d), F_MONO, 20, None, 0, True, 200 + d), 30 + col * 45 + 12, 186 + row * 34, INK, 0.85, align='center')
    return rgb, a

def cal_circle(rgb, a, k):
    """pencil circle around 6 (Aug 1945, Monday -> col 0, row 1)."""
    if k <= 0: return rgb
    cx, cy = 30 + 0 * 45 + 12, 186 + 1 * 34 - 7
    st = Stroke([(cx + 21 * math.cos(q), cy + 16 * math.sin(q)) for q in np.linspace(-0.6, 2 * math.pi + 0.2, 50)], 2.4, 0.8, 5)
    put_alpha(rgb, local_strokes(a.shape, [st], k), 0, 0, C(70, 70, 80), 0.85)
    return rgb

def draw_calendar(img, cam, t, x, y, rot):
    t0, t1 = b0('a_trip') + 0.2, b1('a_trip') - 0.1
    flips = np.linspace(t0, t1, 4)[:3]
    # pages stack: last (Aug) at bottom
    for i in range(3, -1, -1):
        m, name = MONTHS[i]
        rgb, a = cal_page(m, name)
        if i == 3:
            rgb = cal_circle(rgb.copy(), a, eout(prog(t, t1, t1 + 0.6)))
        if i < 3:
            tf = flips[i]
            if t > tf + 0.32: continue
            f = eout(prog(t, tf, tf + 0.32))
            sy = math.cos(f * math.pi / 2 * 1.0)
            dark = 0.35 * f
            rr = rgb * (1 - dark)
            blit_rgba(img, rr, a, cam, x, y - CAL_H / 2, 1.0, (1.0, max(0.02, sy)), rot, (CAL_W / 2, 0), shadow=0.55 if i == 0 or f > 0 else 0.0, lift=0.4 + f)
        else:
            blit_rgba(img, rgb, a, cam, x, y - CAL_H / 2, 1.0, 1.0, rot, (CAL_W / 2, 0), shadow=0.55, lift=0.4)
    # binding rings
    for k in range(5):
        px = x - CAL_W / 2 + 50 + k * 58
        st = [Stroke([(px - 6, y - CAL_H / 2 + 6), (px - 6, y - CAL_H / 2 - 14), (px + 6, y - CAL_H / 2 - 14), (px + 6, y - CAL_H / 2 + 6)], 3.0, 0.1, k)]
        buf = np.zeros((H, W), np.float32); draw_strokes(buf, st, 1.0, cam); paint(img, buf, C(60, 60, 64), 0.9)

def draw_draft(img, cam, t):
    y = YA['draft']
    t0 = b0('a_who') - 0.35
    u = spring(max(0, t - t0) / 0.9, 1.0, 5.0)
    rgb, a = draft_sheet_rgba(t)
    blit_rgba(img, rgb, a, cam, 540, lerp(y + 900, y - 300, u), 1.0, 1.0, math.radians(-1.6), (SHEET_W / 2, SHEET_H / 2), shadow=0.6, lift=0.5)
    # calendar (left) and ID tag (right)
    uc = spring(max(0, t - (b0('a_trip') - 0.3)) / 0.8, 1.0, 5.0)
    if uc > 0:
        draw_calendar(img, cam, t, lerp(-300, 250, uc), y + 330, math.radians(lerp(-14, -3, uc)))
    ut = spring(max(0, t - (b0('a_who') + 0.15)) / 0.8, 1.1, 5.0)
    if ut > 0:
        rgb, a = id_tag_rgba(t)
        blit_rgba(img, rgb, a, cam, lerp(1500, 740, ut), y + 470, 1.0, 1.0, math.radians(lerp(16, 4, ut)), (260, 135), shadow=0.7, lift=0.8)

# ---------------- ticket + home
@functools.lru_cache(None)
def ticket():
    w, h = 640, 290
    rgb, a = card_paper(w, h, 71, base=C(208, 214, 192), edge=0.18)
    hole = Raster(0, 0, w, h).ell((560, 60), (14, 14)).alpha()
    a = a * (1 - hole)
    dash = [Stroke([(470, y), (470, y + 9)], 1.4, 0.0, y) for y in range(14, h - 14, 18)]
    put_alpha(rgb, local_strokes(a.shape, dash), 0, 0, INK, 0.6)
    put_text_local(rgb, _T('広島', F_CJK, 76, None, 4, False, 0, 0), 40, 128, INK)
    put_text_local(rgb, _T('→', F_SERIF, 70, 500), 222, 120, INK)
    put_text_local(rgb, _T('長崎', F_CJK, 76, None, 4, False, 0, 0), 300, 128, INK)
    put_text_local(rgb, _T('ΧΙΡΟΣΙΜΑ  →  ΝΑΓΚΑΣΑΚΙ', F_MONO, 24, None, 1, True, 8), 44, 196, INK)
    put_text_local(rgb, _T('6.8.1945', F_MONO, 22, None, 1, True, 9), 494, 150, INK, 0.8)
    put_text_local(rgb, _T('№ 0806', F_MONO, 22, None, 1, True, 10), 494, 190, INK, 0.8)
    return rgb, a

def house_strokes(cx, by, k=1.35):
    """small Japanese house, front view: hipped roof with lifted eaves, sliding doors, window. scaled by k about (cx,by)."""
    s = []
    s.append(Stroke([(cx - 250, by - 150), (cx - 175, by - 250), (cx + 175, by - 250), (cx + 250, by - 150)], 2.6, 0.8, 1))
    s.append(Stroke([(cx - 270, by - 140), (cx - 250, by - 150), (cx + 250, by - 150), (cx + 270, by - 140)], 2.4, 0.6, 2))
    s.append(Stroke([(cx - 215, by - 150), (cx - 215, by)], 2.2, 0.6, 3)); s.append(Stroke([(cx + 215, by - 150), (cx + 215, by)], 2.2, 0.6, 4))
    s.append(Stroke([(cx - 290, by), (cx + 290, by)], 2.2, 0.8, 5))
    for k, x in enumerate((-180, -110, -40)):
        s.append(Stroke([(cx + x, by - 2), (cx + x, by - 120), (cx + x + 64, by - 120), (cx + x + 64, by - 2)], 1.6, 0.4, 10 + k))
        for j in range(1, 4):
            s.append(Stroke([(cx + x, by - 120 + j * 30), (cx + x + 64, by - 120 + j * 30)], 0.9, 0.2, 20 + k * 4 + j))
    s.append(Stroke([(cx + 60, by - 120), (cx + 180, by - 120), (cx + 180, by - 50), (cx + 60, by - 50), (cx + 60, by - 120)], 1.8, 0.4, 30))
    s.append(Stroke([(cx + 120, by - 120), (cx + 120, by - 50)], 1.2, 0.2, 31))
    for k in range(14):   # roof tiles
        x = cx - 160 + k * 24
        s.append(Stroke([(x, by - 245), (x - 8 - 4 * (k - 6.5) / 6.5, by - 155)], 0.9, 0.3, 40 + k))
    for st in s:
        st.P = np.column_stack([cx + (st.P[:, 0] - cx) * 1.35, by + (st.P[:, 1] - by) * 1.35]); st.L *= 1.35; st.len *= 1.35; st.w *= 1.45
    return s

def draw_home(img, cam, t):
    y = YA['home']
    tk = b0('a_home') - 0.4
    u = spring(max(0, t - tk) / 0.85, 1.0, 5.0)
    rgb, a = ticket()
    blit_rgba(img, rgb, a, cam, lerp(-500, 520, u), y - 470, 1.0, 1.0, math.radians(lerp(-20, -4, u)), (320, 145), shadow=0.7, lift=0.7)
    # house drawn while 'στη γυναίκα του και στο μωρό'
    t0 = b0('a_home') + 0.35 * (b1('a_home') - b0('a_home'))
    p = eout(prog(t, t0, b1('a_home') + 0.2))
    cx, by = 540, y + 470
    lamp = ease(prog(t, b1('a_home') - 0.5, b1('a_home') + 0.5))
    if lamp > 0:
        K = 1.35
        win = Raster(cx + 50 * K, by - 130 * K, 140 * K, 90 * K).poly([(cx + 62 * K, by - 118 * K), (cx + 178 * K, by - 118 * K), (cx + 178 * K, by - 52 * K), (cx + 62 * K, by - 52 * K)]).alpha()
        blit(img, wash(win, 7, 0.3, 0.2), cam, cx + 50 * K, by - 130 * K, AMBER, 0.8 * lamp, mode='multiply')
        sx, sy = cam.pt(cx + 120 * K, by - 85 * K)
        glow = radial(int(sx), int(sy), int(170 * cam.z))
        paint(img, glow, AMBER, 0.22 * lamp, mode='screen')
    buf = np.zeros((H, W), np.float32)
    draw_strokes(buf, _house(), p, cam, tip=True)
    paint(img, inkify(buf), INK, 0.92)
    gr = Raster(cx - 400, by - 6, 800, 40)
    k = ease(prog(t, b1('a_home') - 0.3, b1('a_home') + 0.4))
    if k > 0:
        h = hatch_uv(cam, math.radians(-30), 9, 1.8, 3)
        m = warp(gr.poly([(cx - 390, by + 2), (cx + 390, by + 2), (cx + 340, by + 30), (cx - 340, by + 30)]).alpha(1.0, 3), layer_M(cam, cx - 400, by - 6))
        paint(img, m * h, INK, 0.5 * k)

@functools.lru_cache(None)
def _house():
    return house_strokes(540, YA['home'] + 470)

# ---------------- the road, the forgotten hanko
RX = 470
def road_strokes(y):
    s = []
    top, bot = y - 720, y + 760
    s.append(Stroke([(RX - 60, top), (RX - 60, bot)], 2.0, 0.8, 1)); s.append(Stroke([(RX + 60, top), (RX + 60, bot)], 2.0, 0.8, 2))
    for k, yy in enumerate((y - 420, y + 20, y + 420)):
        s.append(Stroke([(40, yy - 50), (RX - 60, yy - 50)], 1.6, 0.6, 10 + k)); s.append(Stroke([(40, yy + 50), (RX - 60, yy + 50)], 1.6, 0.6, 20 + k))
        s.append(Stroke([(RX + 60, yy - 50), (1040, yy - 50)], 1.6, 0.6, 30 + k)); s.append(Stroke([(RX + 60, yy + 50), (1040, yy + 50)], 1.6, 0.6, 40 + k))
    return s

def steps_path(y):
    """walk up from the shipyard, stop, turn, walk back. returns list of (x, y, ang, side) in time order and the stop index."""
    pts = []
    yy = y + 700; i = 0
    while yy > y - 120:
        pts.append((RX + (-14 if i % 2 else 14), yy, -math.pi / 2, i % 2)); yy -= 46; i += 1
    stop = len(pts)
    # U-turn
    cx, cy = RX + 50, y - 120
    for a in np.linspace(math.pi, 2 * math.pi, 5)[1:]:
        pts.append((cx + 50 * math.cos(a) + (-12 if i % 2 else 12) * math.cos(a), cy + 50 * math.sin(a), a + math.pi / 2, i % 2)); i += 1
    yy = y - 80
    while yy < y + 640:
        pts.append((RX + 100 + (-14 if i % 2 else 14), yy, math.pi / 2, i % 2)); yy += 46; i += 1
    return pts, stop

@functools.lru_cache(None)
def footprint():
    R = Raster(-14, -22, 28, 44)
    R.ell((0, -6), (8, 12)).ell((0, 13), (6, 6))
    return R.alpha(0.6, 5)

@functools.lru_cache(None)
def hanko_obj():
    """the seal itself: a short wooden cylinder, side view, inked."""
    w, h = 120, 230
    R = Raster(0, 0, w, h).poly([(14, 12), (106, 12), (106, 214), (14, 214)]).ell((60, 12), (46, 10)).ell((60, 214), (46, 10))
    a = R.alpha(0.5, 2)
    rgb = np.ones(a.shape + (3,), np.float32) * C(196, 160, 112)
    hh, ww = a.shape
    X = np.mgrid[0:hh, 0:ww][1].astype(np.float32) / S
    shade = 0.72 + 0.38 * np.clip(np.sin(np.clip((X - 14) / 92, 0, 1) * math.pi), 0, 1) ** 0.8
    rgb *= np.clip(shade, 0, 1.2)[..., None] * (0.9 + 0.1 * fbm(hh, ww, 8 * S, 3, 2))[..., None]
    grain = hatch_layer(hh, ww, math.radians(90), 7, 0.8, 4)
    put_alpha(rgb, grain * a, 0, 0, C(120, 90, 60), 0.35)
    st = local_strokes(a.shape, [Stroke([(14, 12), (14, 214)], 1.3, 0.3, 1), Stroke([(106, 12), (106, 214)], 1.3, 0.3, 2),
                                 Stroke([(14 + 46 * (1 + math.cos(q)), 214 + 10 * math.sin(q)) for q in np.linspace(0, math.pi, 20)], 2.0, 0.2, 3),
                                 Stroke([(60 + 46 * math.cos(q), 12 + 10 * math.sin(q)) for q in np.linspace(0, 2 * math.pi, 36)], 2.0, 0.2, 4)])
    put_alpha(rgb, st, 0, 0, INK, 0.9)
    band = np.zeros(a.shape, np.float32); band[int(190 * S):int(214 * S)] = 1
    put_alpha(rgb, band * a, 0, 0, VERM, 0.75)          # red ink on the face
    return rgb, a

@functools.lru_cache(None)
def hanko_print():
    """red seal imprint: ring + 山口 stacked, with stamp texture."""
    d = 150
    a = np.zeros((int(d * S), int(d * S)), np.float32)
    cv2.circle(a, (int(d * S / 2), int(d * S / 2)), int(d * S / 2 - 8 * S), 1.0, int(7 * S), cv2.LINE_AA)
    tmp = np.zeros(a.shape + (3,), np.float32)
    for ch, yy in (('山', 70), ('口', 128)):
        put_text_local(tmp, _T(ch, F_CJK, 58, None, 0, False, 0, 0), d / 2, yy, (255, 255, 255), align='center')
    a = np.maximum(a, tmp[..., 0] / 255)
    hh, ww = a.shape
    ink = 0.6 + 0.4 * fbm(hh, ww, 14 * S, 9, 3)
    holes = (fbm(hh, ww, 2.5 * S, 10, 2) > 0.3).astype(np.float32)
    return np.clip(a * ink * (0.5 + 0.5 * holes) * 1.2, 0, 1)

def draw_road(img, cam, t):
    y = YA['road']
    t0 = b0('a_road') - 0.6
    buf = np.zeros((H, W), np.float32)
    draw_strokes(buf, _road(), eout(prog(t, t0, t0 + 1.2)), cam)
    paint(img, inkify(buf), INK, 0.8)
    # blocks: light hatching
    k = ease(prog(t, t0 + 0.5, t0 + 1.4))
    if k > 0:
        hm = hatch_uv(cam, math.radians(45), 11, 1.4, 5)
        U, V = cam.uv()
        road = (np.abs(U - RX) < 66) | (np.abs(V - (y - 420)) < 56) | (np.abs(V - (y + 20)) < 56) | (np.abs(V - (y + 420)) < 56)
        inside = (U > 40) & (U < 1040) & (V > y - 720) & (V < y + 760)
        paint(img, hm * (~road & inside), INK, 0.22 * k)
    # labels
    for txt, yy, kk in (('ΣΤΑΘΜΟΣ', y - 640, 0), ('ΝΑΥΠΗΓΕΙΟ', y + 700, 1)):
        T = _T(txt, F_MONO, 34, None, 4, True, 30 + kk)
        tx = 760
        ka = prog(t, t0 + 0.3 + 0.2 * kk, t0 + 1.0 + 0.2 * kk)
        box = Raster(tx - 140, yy - 40, 280, 64).poly([(tx - 130, yy - 34), (tx + 130, yy - 34), (tx + 130, yy + 20), (tx - 130, yy + 20)]).alpha()
        blit(img, box, cam, tx - 140, yy - 40, PAPER, ka)
        text_at(img, cam, T, tx, yy, INK, 0.9, n=int(len(txt) * ka), align='center')
    # footsteps
    pts, stop = _steps()
    tw0, tw1 = b0('a_road') - 0.1, b1('a_road') + 0.15            # walk up
    tb0, tb1 = b0('a_back') + 0.1, b1('a_back') + 0.6              # walk back
    n_up = int(stop * eout(prog(t, tw0, tw1)) + 0.5)
    n_bk = int((len(pts) - stop) * prog(t, tb0, tb1) + 0.5)
    nshow = n_up + (n_bk if n_up >= stop else 0)
    fp = footprint()
    for i in range(min(nshow, len(pts))):
        x, yy, ang, side = pts[i]
        age = (nshow - i)
        op = 0.85 * (0.55 + 0.45 * math.exp(-age / 10))
        blit(img, fp, cam, x, yy, INK, op, rot=ang + math.pi / 2, anchor=(14, 22))
    # "!"
    ke = prog(t, b0('a_forgot') + 0.1, b0('a_forgot') + 0.4)
    if ke > 0 and t < b1('a_back') + 1.0:
        E = _T('!', F_SERIF, 120, 700)
        text_at(img, cam, E, RX - 120, y - 140, INK, min(1, ke * 2), align='center', scale=back(ke, 2.2))
    # the hanko: appears on 'τη σφραγίδα του', stamps on 'το χάνκο'
    hx, hy = 800, y - 260
    tf = b0('a_forgot'); dur = b1('a_forgot') - tf
    t_app = tf + 0.45 * dur; t_hit = tf + 0.86 * dur
    if t > t_app - 0.1:
        ua = spring(max(0, t - t_app) / 0.6, 1.0, 5)
        if t < t_hit:
            lift_y = lerp(-500, -60, ua)
        else:
            q = t - t_hit
            lift_y = 0 if q < 0.12 else -min(140, (q - 0.12) * 900)
        # imprint
        if t >= t_hit:
            k = eout(prog(t, t_hit, t_hit + 0.08))
            blit(img, hanko_print(), cam, hx, hy + 200, VERM, 0.92 * k, anchor=(75, 75), rot=math.radians(-6), mode='multiply')
        rgb, a = hanko_obj()
        sq = 1.0
        if t_hit <= t < t_hit + 0.12:
            sq = 1 - 0.06 * math.sin((t - t_hit) / 0.12 * math.pi)
        blit_rgba(img, rgb, a, cam, hx, hy + 200 + lift_y, 1.0, (1.0, sq), math.radians(-6), (60, 214), shadow=0.6, lift=0.2 + abs(lift_y) / 120)

@functools.lru_cache(None)
def _road(): return road_strokes(YA['road'])
@functools.lru_cache(None)
def _steps(): return steps_path(YA['road'])

# ---------------- the watch
@functools.lru_cache(None)
def fbm_scr():
    return fbm(H, W, 120, 61, 4)

def flash_time():
    return b0('a_flash') + 0.38

def draw_watch(img, cam, t):
    """pocket watch ticking to 8:15. At the flash it stops; then it fades and leaves its shadow burnt into the paper."""
    y = YA['watch']
    t_815 = b1('a_815') - 0.15
    tf = flash_time()
    stopped = t >= tf
    clock = 8 * 3600 + 15 * 60 + (min(t, tf) - t_815)
    rgb, a = gfx.watch(clock, glint=0.05, stopped=stopped)
    s = 2 * (gfx.WR + 60) / 800.0                       # watch px per canvas unit (case = 800 units)
    anc = (gfx.WS / 2 / s, gfx.WS / 2 / s)
    q = t - tf
    op = 1.0 if q < 0.35 else 1 - 0.9 * ease(prog(q, 0.35, 1.9))
    if q > 0.2:
        # like the shadows of Hiroshima: the paper around it scorches, where the watch lay stays pale
        k = ease(prog(q, 0.25, 1.6))
        sh = warp(cv2.GaussianBlur(a, (0, 0), 9), layer_M(cam, 540, y, 1.0, math.radians(-4), anc, s))
        X, Y = cam.uv()
        scorch = (0.55 + 0.45 * fbm_scr()) * (1 - sh) * vband(cam, y - 1000, y + 1000, 220)
        paint(img, scorch, C(150, 110, 78), 0.55 * k, mode='multiply')
    if op > 0:
        blit_rgba(img, rgb.astype(np.float32), a, cam, 540, y, op, 1.0, math.radians(-4), anc, shadow=0.8 * (1 if q < 0 else op), lift=1.0, s=s)

def flash_amount(t):
    q = t - flash_time()
    if q < 0: return 0.0, 0.0, q
    f = eout(q / 0.06) * (1.0 if q < 0.22 else math.exp(-(q - 0.22) / 0.55))
    burn = clamp01(q / 0.15) * (0.18 + 0.30 * math.exp(-max(0, q - 0.2) / 1.1))
    return f, burn, q

exec(compile(open(os.path.join(HERE, 'story_a2.py')).read(), os.path.join(HERE, 'story_a2.py'), 'exec'))
exec(compile(open(os.path.join(HERE, 'story_b.py')).read(), os.path.join(HERE, 'story_b.py'), 'exec'))
exec(compile(open(os.path.join(HERE, 'qr.py')).read(), os.path.join(HERE, 'qr.py'), 'exec'))

def story_a_cam(t):
    Y = YA
    keys = [(b0('a_japan') - 0.6, 540, Y['map'] - 80, 1.06, 0.0),
            (b1('a_date') + 0.2, 540, Y['map'], 1.0, 0.0),
            (b0('a_who') - 0.45, 540, Y['map'] + 40, 0.98, 0.0),
            (b0('a_who') + 0.35, 540, Y['draft'], 1.0, 0.0),
            (b1('a_trip') - 0.1, 540, Y['draft'] + 30, 1.03, 0.0),
            (b0('a_home') - 0.4, 540, Y['draft'] + 40, 1.03, 0.0),
            (b0('a_home') + 0.4, 540, Y['home'], 1.0, 0.0),
            (b1('a_home') + 0.3, 540, Y['home'] + 40, 1.04, 0.0),
            (b0('a_road') - 0.65, 540, Y['home'] + 60, 1.04, 0.0),
            (b0('a_road') + 0.15, 540, Y['road'], 1.0, 0.0),
            (b0('a_forgot') + 0.2, 560, Y['road'] - 80, 1.08, 0.0),
            (b1('a_forgot') + 0.2, 640, Y['road'] - 60, 1.12, 0.0),
            (b0('a_back') + 0.1, 600, Y['road'] - 40, 1.08, 0.0),
            (b1('a_back') + 0.3, 560, Y['road'] + 120, 1.0, 0.0),
            (b0('a_815') - 0.75, 560, Y['road'] + 160, 1.0, 0.0),
            (b0('a_815') - 0.05, 540, Y['watch'], 1.0, 0.0),
            (flash_time(), 540, Y['watch'] + 10, 1.18, math.radians(0.5)),
            (b0('a_3km') - 0.9, 540, Y['watch'] + 10, 1.22, math.radians(0.5)),
            (b0('a_3km') - 0.15, 560, Y['blast'] + 60, 1.0, 0.0),
            (b1('a_3km') + 0.2, 540, Y['blast'] + 140, 1.06, 0.0),
            (b0('a_burn') - 0.55, 540, Y['blast'] + 160, 1.06, 0.0),
            (b0('a_burn') + 0.15, 540, Y['cloud'], 1.0, 0.0),
            (b1('a_alive') + 0.1, 540, Y['cloud'] + 90, 1.05, 0.0),
            (b0('a_train') - 0.55, 540, Y['cloud'] + 110, 1.05, 0.0),
            (b0('a_train') + 0.2, 560, Y['route'] - 20, 1.0, 0.0),
            (b1('a_train') + 0.2, 470, Y['route'] - 30, 1.08, 0.0),
            (b0('a_ghost') - 0.5, 470, Y['route'] - 30, 1.08, 0.0),
            (b0('a_ghost') + 0.25, 480, Y['ghost'] + 340, 1.15, 0.0),
            (b1('a_ghost') + 0.1, 460, Y['ghost'] + 380, 1.32, 0.0),
            (b0('a_aug9') - 0.55, 460, Y['ghost'] + 390, 1.32, 0.0),
            (b0('a_aug9') + 0.15, 540, Y['report'] - 300, 1.0, 0.0),
            (b0('a_boss') + 0.4, 560, Y['report'] + 40, 1.0, 0.0),
            (b1('a_nobelieve') + 0.3, 590, Y['report'] + 180, 1.1, 0.0),
            (b1('a_light') + 0.2, 590, Y['report'] + 160, 1.2, 0.0),
            (b0('a_quote') - 0.1, 540, Y['quote'], 1.0, 0.0),
            (b1('a_quote') + 0.2, 540, Y['quote'] + 20, 1.05, 0.0),
            (b0('a_again') - 0.5, 540, Y['quote'] + 30, 1.05, 0.0),
            (b0('a_again') + 0.2, 540, Y['life'] - 60, 1.0, 0.0),
            (b1('a_93') + 0.6, 540, Y['life'] - 80, 1.06, 0.0)]
    return cam_path(t, keys)

def story_a(t):
    cam = story_a_cam(t)
    img = paper_a().render(cam)
    vis = lambda yc: abs(cam.cy - yc) < 2100 / cam.z
    for key, fn in (('map', draw_map), ('draft', draw_draft), ('home', draw_home), ('road', draw_road), ('watch', draw_watch),
                    ('blast', draw_blast), ('cloud', draw_cloud), ('route', draw_route), ('ghost', draw_ghost),
                    ('report', draw_report), ('quote', draw_quote), ('life', draw_life)):
        if vis(YA[key]): fn(img, cam, t)
    # flash: white-hot bloom from the sky, then a warm over-exposure that lingers until we leave the watch
    f, burn, q = flash_amount(t)
    if q >= 0 and t < b0('a_3km') + 0.5:
        leave = 1 - ease(prog(t, b0('a_3km') - 0.9, b0('a_3km') - 0.1))
        bloom = radial(860, 60, 1700)[..., None]
        hot = np.float32([246, 250, 255]) * 0.7 + AMBER * 0.3
        img = img * (1 - burn * leave) + AMBER * 1.08 * burn * leave
        img = img * (1 - f) + hot * (0.75 + 0.25 * bloom) * f
    img = ear_blur(img, t)
    g, o = light_amount(t)
    if o > 0:
        img = img * (1 - o) + (np.float32([246, 250, 255]) * 0.8 + AMBER * 0.2) * o
    return img, cam

# ======================================================================= card A forms (bridge), desk
def t_bridge_a(): return b1('a_93') + 0.5
def t_dive_b(): return b0('b_1912') - 0.35

_CA = None
def cardanim():
    global _CA
    if _CA is None:
        import cardanim05
        _CA = cardanim05
    return _CA

def desk_cards(t, a_progress, b_hidden=True):
    """desk with card A building itself (a_progress 0..1 maps to the anim schedule) and card B hidden below."""
    ca = cardanim()
    cam = Cam(W / 2, H / 2, 1.0)
    img = desk_bg().copy()
    # B, hidden, already waiting
    rgb, a = hidden_card('Β', 7)
    x, y, r = card_pose_final('B')
    blit_rgba(img, rgb, a, cam, x, y, 1.0, 1.0, r, (CW / 2, CH / 2), shadow=0.8, lift=0.6)
    # A arrives and fills
    tt = 0.6 + a_progress * 6.8
    A = ca.card_A(tt)
    ua = spring(a_progress * 2.2, 1.0, 5.0)
    x, y, r = card_pose_final('A')
    ax = np.ones(A.shape[:2], np.float32)
    blit_rgba(img, A, ax, cam, lerp(-700, x, ua), lerp(300, y, ua), 1.0, 1.0, lerp(-0.3, r, ua), (CW / 2, CH / 2), shadow=0.8, lift=0.6, s=1)
    T1 = _T('ΜΙΑ ΑΛΗΘΕΙΑ  ·  ΕΝΑ ΨΕΜΑ', F_SERIF, 40, 500, 5)
    text_at(img, cam, T1, W / 2, 112, C(236, 226, 210), 0.85, align='center')
    return img

# ======================================================================= transitions + frame
def motion_blur(img, v):
    """vertical blur proportional to camera speed (px/frame)."""
    k = int(min(41, abs(v) * 0.6))
    if k < 3: return img
    ker = np.zeros((k, 1), np.float32); ker[:, 0] = 1.0 / k
    return cv2.filter2D(img, -1, ker)

# ---------------- subtitles (channel style: white serif on a soft dark band)
SUBY = 1610
_SUBS = None
def subs():
    global _SUBS
    if _SUBS is None:
        import subs05
        _SUBS = subs05.build(TL)
    return _SUBS

@functools.lru_cache(None)
def sub_alpha(text):
    size = 58
    while True:
        T_ = Text(text, F_SERIF, size, wght=560, typed=False)
        if T_.ink_w < W - 90 or size <= 40: return T_
        size -= 2

# beats whose words are already written in the frame (the quote, the closing titles): no subtitle there
NOSUB = ('a_quote', 'r_cta')

def subtitles(img, t):
    cam = Cam(W / 2, H / 2, 1.0)
    for s_ in subs():
        if any(b0(k) - 0.05 <= s_['t0'] <= b1(k) + 0.05 for k in NOSUB):
            continue
        if s_['t0'] - 0.05 <= t <= s_['t1'] + 0.1:
            a = clamp01((t - s_['t0'] + 0.05) / 0.1) * clamp01((s_['t1'] + 0.1 - t) / 0.1)
            img = eng.band(img, SUBY - 48, SUBY + 48, 0.45 * a, 70)
            T_ = sub_alpha(s_['text'])
            sh = cv2.GaussianBlur(warp(T_.a, layer_M(cam, W / 2 + 3, SUBY + 4, 1.0, 0, (T_.ink_w / 2 + 0.2 * (T_.h / 1.6), T_.base - 22))), (0, 0), 3)
            paint(img, sh, np.float32([0, 0, 0]), 0.6 * a)
            text_at(img, cam, T_, W / 2, SUBY + 22, C(246, 242, 236), a, align='center')
    return img

_prev = {}
def frame(t, fi):
    t_dive = b1('hook5') + 0.15
    tb = t_bridge_a()
    if t < t_dive:
        img = hook(t)
    elif t < t_dive + 0.9:
        u = (t - t_dive) / 0.9
        h = hook(t_dive)
        z = 1 + 5 * ein(u)
        cx, cy = 534, 600
        M = np.float32([[z, 0, W / 2 - z * cx], [0, z, H / 2 - z * cy]])
        hz = cv2.warpAffine(h, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        sa, cam = story_a(b0('a_japan') - 0.6)
        mix = ease(prog(u, 0.55, 1.0))
        img = hz * (1 - mix) + sa * mix
    elif t < tb + 0.6:
        img, cam = story_a(min(t, tb + 0.6))
        v = 0
        if 'cy' in _prev and abs(_prev.get('t', -9) - t) < 0.05:
            v = (cam.cy - _prev['cy']) * cam.z
        _prev.update(cy=cam.cy, t=t + 1 / 30)
        img = motion_blur(img, v)
        if t > tb:
            m = eng.ink_mask(prog(t, tb, tb + 0.6), seed=13, center=(0.5, 0.45))
            img = img * (1 - m) + desk_cards(t, prog(t, tb, t_dive_b())) * m
    elif t < t_dive_b():
        img = desk_cards(t, prog(t, tb, t_dive_b()))
    elif t < t_dive_b() + 0.9:
        u = (t - t_dive_b()) / 0.9
        h = desk_cards(t_dive_b(), 1.0)
        z = 1 + 5 * ein(u)
        cx, cy = 546, 1300
        M = np.float32([[z, 0, W / 2 - z * cx], [0, z, H / 2 - z * cy]])
        hz = cv2.warpAffine(h, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        sb, cam = story_b(b0('b_1912') - 0.3)
        mix = ease(prog(u, 0.55, 1.0))
        img = hz * (1 - mix) + sb * mix
    elif t < t_bridge_b() + 0.6:
        img, cam = story_b(min(t, t_bridge_b() + 0.6))
        v = 0
        if 'cyb' in _prev and abs(_prev.get('tb', -9) - t) < 0.05:
            v = (cam.cy - _prev['cyb']) * cam.z
        _prev.update(cyb=cam.cy, tb=t + 1 / 30)
        img = motion_blur(img, v)
        if t > t_bridge_b():
            m = eng.ink_mask(prog(t, t_bridge_b(), t_bridge_b() + 0.6), seed=17, center=(0.5, 0.6))
            img = img * (1 - m) + desk_b_forming(t) * m
    elif t < b0('q_which') - 0.05:
        img = desk_b_forming(t)
    elif t < b0('r_true'):
        img = desk_question(t)
    elif t < b0('r_only') - 0.2:
        img = reveal_cards(t)
    elif t < b0('r_wife') - 0.6:
        img = desk_recognised(t)
    elif t < b0('r_wife'):
        m = eng.ink_mask(prog(t, b0('r_wife') - 0.6, b0('r_wife')), seed=19, center=(0.55, 0.5))
        img = desk_recognised(t) * (1 - m) + story_r(t)[0] * m
    elif t < b1('r_sea') + 0.3:
        img, cam = story_r(t)
        v = 0
        if 'cyr' in _prev and abs(_prev.get('tr', -9) - t) < 0.05:
            v = (cam.cy - _prev['cyr']) * cam.z
        _prev.update(cyr=cam.cy, tr=t + 1 / 30)
        img = motion_blur(img, v)
    elif t < b1('r_sea') + 0.9:
        m = eng.ink_mask(prog(t, b1('r_sea') + 0.3, b1('r_sea') + 0.9), seed=23, center=(0.5, 0.5))
        img = story_r(t)[0] * (1 - m) + desk_end(t) * m
    else:
        img = desk_end(t)
    if os.environ.get('NOSUBS') is None:
        img = subtitles(img, t)
    return grade(img, t, fi, FILM, warmth=0.6)
