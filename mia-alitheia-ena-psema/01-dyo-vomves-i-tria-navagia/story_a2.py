
# ======================================================================= story A, part 2 (after the flash)
YA.update({'blast': 11000, 'cloud': 13000, 'route': 15000, 'ghost': 17000, 'report': 19000, 'quote': 21000, 'life': 23000})

def cam_off(cam, dy):
    """same camera, but content drawn at canvas y appears dy lower (re-use of earlier drawings)."""
    return Cam(cam.cx, cam.cy - dy, cam.z, cam.rot)

def ink_strokes(img, cam, strokes, p, color=INK, op=0.92, tip=False, ws=1.0):
    if p <= 0: return
    buf = np.zeros((H, W), np.float32)
    draw_strokes(buf, strokes, p, cam, tip=tip, width_scale=ws)
    paint(img, inkify(buf), color, op)

def circle_pts(cx, cy, r, a0=-math.pi / 2, a1=None, n=None):
    a1 = a0 + 2 * math.pi if a1 is None else a1
    n = n or max(24, int(abs(a1 - a0) * r / 6))
    return [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in np.linspace(a0, a1, n)]

# ---------------- A7: three kilometres
BC = (600, YA['blast'] - 40)
KM = 128.0
YAMA = (BC[0] - 3 * KM * 0.70, BC[1] + 3 * KM * 0.72)

@functools.lru_cache(None)
def delta_rivers():
    """stylised Ota river delta: channels fanning south through the city."""
    y0, y1 = YA['blast'] - 820, YA['blast'] + 820
    chans = []
    r = np.random.default_rng(31)
    for k in range(5):
        xs = BC[0] - 40 + k * 22
        xm = BC[0] + (-380 + k * 190) * 0.5
        xe = BC[0] - 430 + k * 215
        P = np.array(bez((xs, y0), (xm, BC[1] - 60), (xe, y1), 80))
        ph = r.uniform(0, 6.28)
        P[:, 0] += 26 * np.sin(np.linspace(0, 3.2, len(P)) + ph) * np.linspace(0.2, 1, len(P))
        chans.append(P)
    banks = []
    for i, P in enumerate(chans):
        T = np.gradient(P, axis=0); T /= np.hypot(*T.T)[:, None]; N = np.stack([-T[:, 1], T[:, 0]], 1)
        w = np.linspace(4, 10, len(P))[:, None]
        banks += [Stroke(P + N * w, 1.3, 0.8, 300 + i * 2, taper=False), Stroke(P - N * w, 1.3, 0.8, 301 + i * 2, taper=False)]
    R = Raster(0, y0, W, y1 - y0)
    for P in chans:
        R.line(P, 15)
    return banks, wash(R.alpha(0.8, 8), 9, 0.25, 0.3), (0, y0)

def draw_blast(img, cam, t):
    t0, t1 = b0('a_3km'), b1('a_3km')
    k = ease(prog(t, t0 - 0.8, t0 - 0.1))
    # the city underneath: a faint street grid, revealed outward from the centre
    kg = ease(prog(t, t0 - 0.9, t0 + 0.4))
    if kg > 0:
        U, V = cam.uv()
        d = np.sqrt((U - BC[0]) ** 2 + (V - BC[1]) ** 2)
        grid = np.maximum(hatch_uv(cam, 0.0, 46, 1.2, 21, 0.15), hatch_uv(cam, math.pi / 2, 52, 1.2, 22, 0.15))
        disc = np.clip((3.6 * KM - d) / 60, 0, 1) * np.clip((kg * 4.2 * KM - d) / 80, 0, 1)
        paint(img, grid * disc, INK, 0.16)
    H_ = _T('ΧΙΡΟΣΙΜΑ', F_SERIF, 80, 600, 18)
    kh = eout(prog(t, t0 - 0.6, t0 + 0.2))
    text_at(img, cam, H_, 540, BC[1] - 3 * KM - 120 + 12 * (1 - kh), INK, kh, align='center')
    Hy = _T('υπόκεντρο', F_SERIF_I, 32, 500)
    text_at(img, cam, Hy, BC[0] + 26, BC[1] - 18, INK, 0.85 * eout(prog(t, t0, t0 + 0.5)))
    rings = [Stroke(circle_pts(BC[0], BC[1], r * KM, -2.2, -2.2 + 2 * math.pi), 2.4 if r == 3 else 1.4, 0.4, 400 + r, taper=False) for r in (1, 2, 3)]
    ink_strokes(img, cam, rings, eout(prog(t, t0 - 0.3, t0 + 0.9)), op=0.85, tip=True)
    for r in (1, 2, 3):
        T_ = _T(f'{r} χλμ', F_MONO, 26, None, 0, True, 40 + r)
        a = -0.75
        text_at(img, cam, T_, BC[0] + r * KM * math.cos(a) + 10, BC[1] + r * KM * math.sin(a) - 6, INK, 0.85,
                n=type_n(T_, t, t0 + 0.2 + 0.25 * r, 18))
    # hypocentre
    kx = eout(prog(t, t0 - 0.2, t0 + 0.3))
    ink_strokes(img, cam, [Stroke([(BC[0] - 16, BC[1] - 16), (BC[0] + 16, BC[1] + 16)], 3.0, 0.2, 1),
                           Stroke([(BC[0] + 16, BC[1] - 16), (BC[0] - 16, BC[1] + 16)], 3.0, 0.2, 2)], kx)
    T_ = _T('08:15', F_MONO, 28, None, 1, True, 50)
    text_at(img, cam, T_, BC[0] + 26, BC[1] + 26, INK, 0.9, n=type_n(T_, t, t0, 14))
    # shock wave
    t_hit = t0 + 0.62 * (t1 - t0)
    q = t - t_hit
    if q > 0:
        R = 5.2 * KM * eout(q / 0.75)
        op = 1 - prog(q, 0.4, 1.0)
        if op > 0:
            sx, sy = cam.pt(*BC)
            ring = np.zeros((H, W), np.float32)
            cv2.circle(ring, (int(sx * 8), int(sy * 8)), int(R * cam.z * 8), 1.0, max(1, int((10 - 6 * prog(q, 0, 0.75)) * cam.z)), cv2.LINE_AA, shift=3)
            paint(img, inkify(ring), INK, 0.8 * op)
            fill = np.zeros((H, W), np.float32)
            cv2.circle(fill, (int(sx * 8), int(sy * 8)), int(R * cam.z * 8), 1.0, -1, cv2.LINE_AA, shift=3)
            paint(img, cv2.GaussianBlur(fill, (0, 0), 30), AMBER, 0.45 * op, mode='multiply')
    # Yamaguchi: a dot, knocked over when the wave passes
    dist = math.hypot(YAMA[0] - BC[0], YAMA[1] - BC[1])
    t_reach = t_hit + 0.75 * (1 - (1 - dist / (5.2 * KM)) ** (1 / 3)) if dist < 5.2 * KM else t_hit
    kd = eout(prog(t, t0 + 0.3, t0 + 0.7))
    push = eout(prog(t, t_reach, t_reach + 0.35)) * 24
    ux, uy = (YAMA[0] - BC[0]) / dist, (YAMA[1] - BC[1]) / dist
    px, py = YAMA[0] + ux * push, YAMA[1] + uy * push
    if kd > 0:
        dot = np.zeros((H, W), np.float32)
        sx, sy = cam.pt(px, py)
        cv2.circle(dot, (int(sx * 8), int(sy * 8)), int(9 * cam.z * back(kd) * 8), 1.0, -1, cv2.LINE_AA, shift=3)
        paint(img, dot, INK, 0.95)
        T_ = _T('Γιαμαγκούτσι', F_SERIF_I, 34, 500)
        text_at(img, cam, T_, YAMA[0] - 200, YAMA[1] + 60, INK, 0.9 * kd)
        if push > 1:
            lines = [Stroke([(px - ux * (30 + 14 * j) + uy * o, py - uy * (30 + 14 * j) - ux * o),
                             (px - ux * (52 + 14 * j) + uy * o, py - uy * (52 + 14 * j) - ux * o)], 1.6, 0.1, 60 + j)
                     for j, o in enumerate((-10, 0, 10))]
            ink_strokes(img, cam, lines, 1.0, op=0.7 * (1 - prog(t, t_reach + 0.5, t_reach + 1.4)))

# ---------------- A8: the cloud over the city, A9: but he lives

@functools.lru_cache(None)
def big_cloud():
    import cards05
    w, h = 900, 1300
    a = cards05.mushroom(int(w * S), int(h * S), w * S / 2, (h - 20) * S, (h - 120) * S, 23)
    return a, w, h

@functools.lru_cache(None)
def skyline():
    y = YA['cloud'] + 540
    r = np.random.default_rng(12)
    R = Raster(0, y - 120, W, 140)
    x = -10
    while x < W + 10:
        bw = r.uniform(18, 60); bh = r.uniform(12, 70) * (1.3 if abs(x - 540) > 200 else 0.6)
        R.poly([(x, y), (x + bw, y), (x + bw, y - bh), (x, y - bh)]); x += bw + r.uniform(0, 6)
    return R.alpha(0.5, 3), (0, y - 120)

def draw_cloud(img, cam, t):
    yc = YA['cloud']; yh = yc + 540
    t0, t1 = b0('a_burn') - 0.45, b1('a_burn') + 0.6
    a, w, h = big_cloud()
    p = prog(t, t0, t1)
    # warm sky glow behind
    sx, sy = cam.pt(540, yh - 200)
    glow = radial(int(sx), int(sy), int(900 * cam.z))
    paint(img, glow, AMBER, 0.35 * ease(prog(t, t0 - 0.3, t0 + 0.6)), mode='multiply')
    if p > 0:
        e = eout(p)
        sxs, sys_ = lerp(0.45, 1.0, e), lerp(0.10, 1.0, e ** 0.85)
        blit(img, a, cam, 540, yh, INK, 0.9 * min(1, p * 4), scale=(sxs, sys_), anchor=(w / 2, h - 20))
        # fire under the cap and at the root
        lay = np.zeros((H, W), np.float32)
        cx, cyb = cam.pt(540, yh)
        cv2.ellipse(lay, (int(cx), int(cyb)), (int(300 * cam.z * sxs), int(120 * cam.z)), 0, 180, 360, 1.0, -1)
        paint(img, cv2.GaussianBlur(lay, (0, 0), 40), AMBER, 0.6 * min(1, p * 3), mode='multiply')
    sk, (x0, y0) = skyline()
    blit(img, sk, cam, x0, y0, INK, 0.95)
    ink_strokes(img, cam, [Stroke([(0, yh), (W, yh)], 2.4, 0.6, 5, taper=False)], 1.0)
    # A9: heartbeat
    ta0, ta1 = b0('a_alive') - 0.5, b1('a_alive') + 0.2
    ye = yc + 760
    P = [(60, ye), (430, ye), (455, ye - 14), (480, ye), (505, ye), (522, ye + 22), (548, ye - 140), (574, ye + 60), (596, ye), (640, ye), (665, ye - 20), (690, ye), (1020, ye)]
    ink_strokes(img, cam, [Stroke(P, 2.6, 0.3, 70, taper=False)], eout(prog(t, ta0, ta1)), tip=True)

def ear_blur(img, t):
    """the eardrums: a short muffled, smeared moment."""
    tb0, tb1 = b0('a_burn'), b1('a_burn')
    te = tb0 + 0.55 * (tb1 - tb0)
    k = math.sin(math.pi * prog(t, te - 0.1, te + 0.9))
    if k <= 0: return img
    bl = cv2.GaussianBlur(img, (0, 0), 1 + 6 * k)
    M = np.float32([[1, 0, 3 * k * math.sin(t * 90)], [0, 1, 0]])
    bl = cv2.warpAffine(bl, M, (W, H), borderMode=cv2.BORDER_REFLECT)
    return img * (1 - k) + bl * k

# ---------------- A10: the train home
ROUTE = [(132.455, 34.385), (132.22, 34.17), (131.80, 34.05), (131.40, 34.09), (130.93, 33.95), (130.88, 33.88),
         (130.42, 33.59), (130.51, 33.38), (130.30, 33.26), (130.12, 33.21), (130.05, 32.84), (129.877, 32.75)]

@functools.lru_cache(None)
def route_stroke():
    P = [proj(*q) for q in ROUTE]
    sm = []
    for a, b in zip(P, P[1:]):
        sm += [(lerp(a[0], b[0], u), lerp(a[1], b[1], u)) for u in np.linspace(0, 1, 12)[:-1]]
    sm.append(P[-1])
    return Stroke(sm, 3.0, 0.8, 90, taper=False)

def map_static(img, cam, op):
    wsh, wl, (x0, y0) = map_layers()
    blit(img, wsh, cam, x0, y0, PAPER_D, 0.55 * op, mode='multiply')
    blit(img, wl, cam, x0, y0, SEA, 0.85 * op, mode='multiply')
    ink_strokes(img, cam, coast_strokes(), 1.0, op=0.92 * op)
    for (x, y), kanji, gr, side in ((HIRO, '広島', 'Χιροσίμα', 1), (NAGA, '長崎', 'Ναγκασάκι', -1)):
        dot = np.zeros((H, W), np.float32)
        sx, sy = cam.pt(x, y)
        cv2.circle(dot, (int(sx * 8), int(sy * 8)), int(7 * cam.z * 8), 1.0, -1, cv2.LINE_AA, shift=3)
        cv2.circle(dot, (int(sx * 8), int(sy * 8)), int(13 * cam.z * 8), 1.0, max(1, int(1.6 * cam.z)), cv2.LINE_AA, shift=3)
        paint(img, dot, INK, 0.95 * op)
        KJ = _T(kanji, F_CJK, 42, None, 0, False, 0, 0); GR = _T(gr, F_SERIF_I, 36, 500)
        lx = x + side * 34; ly = y + 12
        if side < 0:
            text_at(img, cam, KJ, lx - KJ.ink_w, ly - 6, INK, op); text_at(img, cam, GR, lx - GR.ink_w, ly + 40, INK, 0.9 * op)
        else:
            text_at(img, cam, KJ, lx, ly - 6, INK, op); text_at(img, cam, GR, lx, ly + 40, INK, 0.9 * op)

@functools.lru_cache(None)
def loco():
    R = Raster(-40, -34, 80, 60)
    R.poly([(-30, -4), (14, -4), (14, -22), (30, -22), (30, 10), (-34, 10), (-34, 2)])
    R.poly([(-22, -4), (-22, -18), (-14, -18), (-14, -4)])
    for x in (-24, -6, 14):
        R.ell((x, 14), (7, 7))
    return R.alpha(0.3, 4)

def draw_route(img, cam, t):
    dy = YA['route'] - YA['map']
    c2 = cam_off(cam, dy)
    t0, t1 = b0('a_train') - 0.2, b1('a_train') + 0.4
    op = ease(prog(t, t0 - 0.9, t0 - 0.2))
    if op <= 0: return
    map_static(img, c2, op)
    st = route_stroke()
    p = einout(prog(t, t0 + 0.2, t1))
    # dashed: draw the route then knock out every other dash with paper
    buf = np.zeros((H, W), np.float32)
    draw_strokes(buf, [st], p, c2)
    U, V = c2.uv()
    n = int(np.searchsorted(st.L, p * st.len)); n = max(1, min(n, len(st.P) - 1))
    paint(img, inkify(buf), INK, 0.9)
    tip = st.P[n]
    # smoke: puffs left behind the engine
    for j in range(6):
        u = p - 0.02 * (j + 1)
        if u <= 0: break
        m = int(np.searchsorted(st.L, u * st.len)); m = max(0, min(m, len(st.P) - 1))
        x, y = st.P[m]
        r = 6 + 5 * j
        lay = np.zeros((H, W), np.float32)
        sx, sy = c2.pt(x + 6 * j, y - 20 - 9 * j)
        cv2.circle(lay, (int(sx * 8), int(sy * 8)), int(r * cam.z * 8), 1.0, max(1, int(1.5 * cam.z)), cv2.LINE_AA, shift=3)
        paint(img, lay, INK, 0.55 * (1 - j / 6) * (1 if p < 1 else 1 - prog(t, t1, t1 + 0.5)))
    if p > 0:
        blit(img, loco(), c2, tip[0], tip[1] - 22, INK, 0.95 * min(1, p * 10), anchor=(40, 34), scale=1.25)
    for (x, y), txt, tk in ((HIRO, '7 Αυγ.', t0), (NAGA, '8 Αυγ.', t1 - 0.2)):
        T_ = _T(txt, F_MONO, 30, None, 0, True, 77 if tk == t0 else 78)
        text_at(img, c2, T_, x + (30 if tk == t0 else -150), y - 30, INK, 0.9, n=type_n(T_, t, tk, 16))

# ---------------- A11: the mother at the door
@functools.lru_cache(None)
def bandaged_figure():
    """a standing man in a doorway, back-lit; bandages as paper-white bands."""
    R = Raster(-60, -190, 120, 195)
    R.ell((0, -158), (17, 21))
    R.poly([(-8, -138), (8, -138), (9, -128), (-9, -128)])
    R.poly([(-30, -128), (30, -128), (26, -60), (22, 0), (-22, 0), (-26, -60)])
    R.line([(-30, -124), (-40, -80), (-38, -40)], 13)
    R.line([(30, -124), (40, -80), (38, -40)], 13)
    a = R.alpha(0.6, 6)
    B = Raster(-60, -190, 120, 195)
    for yy in (-166, -154):
        B.line([(-19, yy + 3), (0, yy - 2), (19, yy + 3)], 3.4)
    B.line([(-40, -92), (-34, -58)], 7)
    B.line([(26, -126), (-30, -60)], 5)
    b = B.alpha(0.4, 7) * a
    return a, b

def draw_ghost(img, cam, t):
    dy = YA['ghost'] - YA['home']
    c2 = cam_off(cam, dy)
    t0, t1 = b0('a_ghost'), b1('a_ghost')
    op = ease(prog(t, t0 - 0.9, t0 - 0.2))
    if op <= 0: return
    cx, by = 540, YA['home'] + 470
    K = 1.35
    # interior light through the open door
    k = ease(prog(t, t0 + 0.1, t0 + 0.9))
    door = Raster(cx - 110 * K - 10, by - 120 * K - 10, 64 * K + 20, 120 * K + 20)
    door.poly([(cx - 110 * K, by - 120 * K), (cx - 46 * K, by - 120 * K), (cx - 46 * K, by), (cx - 110 * K, by)])
    blit(img, wash(door.alpha(), 4, 0.2, 0.15), c2, door.x0, door.y0, AMBER, 0.9 * k, mode='multiply')
    sx, sy = c2.pt(cx - 78 * K, by - 60 * K)
    paint(img, radial(int(sx), int(sy), int(260 * cam.z)), AMBER, 0.25 * k, mode='screen')
    ink_strokes(img, c2, _house(), 1.0, op=0.92 * op)
    # the figure fades in like an apparition
    kf = ease(prog(t, t0 + 0.6, t0 + 1.8))
    if kf > 0:
        a, b = bandaged_figure()
        fx, fy = cx - 78 * K, by - 2
        shimmer = 1 + 0.03 * math.sin(t * 7)
        blit(img, a, c2, fx, fy, INK, 0.9 * kf, scale=K * shimmer, anchor=(60, 190))
        blit(img, b, c2, fx, fy, C(238, 232, 222), 0.95 * kf, scale=K * shimmer, anchor=(60, 190))

# ---------------- A12-A13: the report, the disbelief, the light
REP_W, REP_H = 700, 820

@functools.lru_cache(None)
def report_base():
    rgb, a = card_paper(REP_W, REP_H, 91, base=C(236, 230, 216), edge=0.12)
    put_text_local(rgb, _T('ΑΝΑΦΟΡΑ', F_SERIF, 54, 700, 10), REP_W / 2, 100, INK, align='center')
    put_text_local(rgb, _T('ΧΙΡΟΣΙΜΑ  ·  6.8.1945', F_MONO, 28, None, 1, True, 3), REP_W / 2, 150, INK, align='center')
    r = np.random.default_rng(5)
    for row in range(9):
        y = 210 + row * 34
        x = 70
        while x < REP_W - 90:
            L = r.uniform(50, 150)
            blk = np.zeros(a.shape, np.float32)
            for k in range(int(L // 13)):
                hh = r.uniform(9, 15)
                cv2.rectangle(blk, (int((x + k * 13) * S), int((y - hh) * S)), (int((x + k * 13 + 8) * S), int(y * S)), 0.6 + 0.3 * r.random(), -1)
            put_alpha(rgb, cv2.GaussianBlur(blk, (0, 0), 1.0), 0, 0, INK, 0.6)
            x += L + 24
    return rgb, a

def report_rgba(t):
    rgb, a = report_base(); rgb = rgb.copy()
    B = _T('1 ΒΟΜΒΑ = 1 ΠΟΛΗ', F_MONOB, 50, None, 1, True, 9)
    put_text_local(rgb, B, REP_W / 2, 600, INK, n=type_n(B, t, b0('a_onebomb') + 0.05, 9), align='center')
    # the boss's pencil: a big question mark and a wavy underline
    tq = b0('a_nobelieve') + 0.1
    q = [Stroke(bez((520, 690), (530, 650), (565, 650), 10) + bez((565, 650), (600, 652), (596, 690), 10) + bez((596, 690), (590, 716), (562, 728), 10) + [(560, 760)], 4.0, 1.0, 3),
         Stroke(circle_pts(560, 790, 4, 0, 2 * math.pi, 12), 5.0, 0.0, 4),
         Stroke([(110 + x, 625 + 7 * math.sin(x / 18)) for x in range(0, 480, 6)], 2.4, 0.5, 5)]
    pq = eout(prog(t, tq, tq + 0.9))
    if pq > 0:
        put_alpha(rgb, local_strokes(a.shape, q, pq), 0, 0, C(84, 84, 92), 0.85)
    return rgb, a

def light_amount(t):
    """'the room fills with the same white light': grows, then a second white-out."""
    t0, t1 = b0('a_light'), b1('a_light')
    grow = ease(prog(t, t0 + 0.55 * (t1 - t0), t1 + 0.1))
    out = eout(prog(t, t1 + 0.05, t1 + 0.2)) * math.exp(-max(0, t - t1 - 0.45) / 0.6)
    return grow, out

def draw_report(img, cam, t):
    y = YA['report']
    # calendar: 9 August circled
    tc = b0('a_aug9') - 0.3
    uc = spring(max(0, t - tc) / 0.8, 1.0, 5)
    rgb, a = cal_page(8, 'ΑΥΓΟΥΣΤΟΣ'); rgb = rgb.copy()
    cx9, cy9 = 30 + 3 * 45 + 12, 186 + 1 * 34 - 7
    st = Stroke([(cx9 + 21 * math.cos(q), cy9 + 16 * math.sin(q)) for q in np.linspace(-0.6, 2 * math.pi + 0.2, 50)], 2.4, 0.8, 6)
    put_alpha(rgb, local_strokes(a.shape, [st], eout(prog(t, b0('a_aug9') + 0.3, b1('a_aug9') + 0.3))), 0, 0, C(70, 70, 80), 0.85)
    if uc > 0:
        blit_rgba(img, rgb, a, cam, lerp(-300, 250, uc), y - 640, 1.0, 1.0, math.radians(lerp(-12, -4, uc)), (CAL_W / 2, CAL_H / 2), shadow=0.6, lift=0.5)
    ur = spring(max(0, t - (b0('a_boss') - 0.4)) / 0.85, 1.0, 5)
    if ur > 0:
        rgb, a = report_rgba(t)
        blit_rgba(img, rgb, a, cam, lerp(1700, 600, ur), y + 140, 1.0, 1.0, math.radians(lerp(14, 2, ur)), (REP_W / 2, REP_H / 2), shadow=0.6, lift=0.6)
    # window light: four panes thrown across the desk
    g, o = light_amount(t)
    if g > 0:
        U, V = cam.uv()
        # sheared pane grid in canvas coords
        u = (U - 120) + (V - (y - 700)) * 0.35
        v = (V - (y - 700))
        pane = ((u % 330) < 290) & ((v % 560) < 500) & (u > 0) & (u < 990) & (v > 0) & (v < 1120)
        m = cv2.GaussianBlur(pane.astype(np.float32), (0, 0), 6)
        paint(img, m, np.float32([240, 248, 255]) * 0.8 + AMBER * 0.2, 0.8 * g, mode='screen')
        paint(img, np.ones((H, W), np.float32), AMBER, 0.12 * g, mode='multiply')

# ---------------- A14: the quote
QUOTE = ['«Νόμιζα ότι το σύννεφο', 'με είχε ακολουθήσει', 'από τη Χιροσίμα.»']

def draw_quote(img, cam, t):
    y = YA['quote']
    t0, t1 = b0('a_quote'), b1('a_quote')
    a, w, h = big_cloud()
    blit(img, a, cam, 540, y + 760, INK, 0.10, anchor=(w / 2, h - 20), scale=1.0)
    for i, ln in enumerate(QUOTE):
        T_ = _T(ln, F_SERIF_I, 78, 500)
        k = eout(prog(t, t0 + 0.35 * i * (t1 - t0) / 1.6, t0 + 0.35 * i * (t1 - t0) / 1.6 + 0.8))
        text_at(img, cam, T_, 540, y - 150 + i * 108 + 16 * (1 - k), INK, k, align='center')
    A_ = _T('— Τσουτόμου Γιαμαγκούτσι', F_MONO, 30, None, 0, True, 12)
    text_at(img, cam, A_, 540, y + 250, INK, 0.85, n=type_n(A_, t, t1 - 1.0, 24), align='center')

# ---------------- A15: a life, 1916-2010
LX0, LX1 = 90, 990
def yr_x(yr): return LX0 + (yr - 1916) / (2010 - 1916) * (LX1 - LX0)

@functools.lru_cache(None)
def mini_cloud():
    import cards05
    w, h = 120, 170
    return cards05.mushroom(int(w * S), int(h * S), w * S / 2, (h - 6) * S, (h - 16) * S, 5), w, h

def draw_life(img, cam, t):
    y = YA['life'] + 120
    ta, tb = b0('a_again'), b0('a_93')
    Nm = _T('Τσουτόμου Γιαμαγκούτσι', F_SERIF_I, 48, 500)
    text_at(img, cam, Nm, LX0, y - 560, INK, 0.9 * ease(prog(t, ta - 0.6, ta)))
    Kj = _T('山口 彊', F_CJK, 40, None, 4, False, 0, 0)
    text_at(img, cam, Kj, LX0, y - 490, INK, 0.8 * ease(prog(t, ta - 0.4, ta + 0.2)))
    p1 = eout(prog(t, ta - 0.5, ta + 0.6))
    p2 = eout(prog(t, tb, b1('a_93') + 0.2))
    x45 = yr_x(1945)
    ink_strokes(img, cam, [Stroke([(LX0, y), (x45, y)], 4.0, 0.3, 1, taper=False)], p1, tip=True)
    ink_strokes(img, cam, [Stroke([(x45, y), (LX1, y)], 4.0, 0.3, 2, taper=False)], p2, tip=True)
    for yr, k in ((1916, p1), (1945, p1), (2010, p2)):
        if k <= 0.02: continue
        x = yr_x(yr)
        ink_strokes(img, cam, [Stroke([(x, y - 22), (x, y + 22)], 3.0, 0.1, yr)], 1.0, op=k)
        T_ = _T(str(yr), F_MONO, 46, None, 1, True, yr)
        text_at(img, cam, T_, x, y + 90, INK, k, align='center')
    # two small clouds over 1945: Hiroshima, then Nagasaki
    a, w, h = mini_cloud()
    for j, (lab, tk) in enumerate((('6.8', ta - 0.35), ('9.8', ta + 0.15))):
        k = prog(t, tk, tk + 0.6)
        if k <= 0: continue
        x = x45 + (j * 2 - 1) * 62
        e = eout(k)
        blit(img, a, cam, x, y - 40, INK, 0.9 * min(1, k * 3), scale=(lerp(0.5, 1, e), lerp(0.1, 1, e)), anchor=(w / 2, h - 6))
        T_ = _T(lab, F_MONO, 32, None, 0, True, 30 + j)
        text_at(img, cam, T_, x, y - 230, INK, 0.9, n=type_n(T_, t, tk + 0.2, 16), align='center')
    k = back(prog(t, b1('a_93') - 0.3, b1('a_93') + 0.3))
    if k > 0:
        N = _T('93', F_SERIF, 170, 600)
        text_at(img, cam, N, 760, y - 330, INK, min(1, k), align='center', scale=0.85 + 0.15 * k)
        ink_strokes(img, cam, [Stroke(circle_pts(760, y - 390, 130, -2.0, -2.0 + 2 * math.pi + 0.4), 3.4, 1.4, 9)],
                    eout(prog(t, b1('a_93'), b1('a_93') + 0.6)), op=0.85)
        Y_ = _T('χρονών', F_SERIF_I, 44, 500)
        text_at(img, cam, Y_, 760, y - 200, INK, min(1, k) * 0.9, align='center')
