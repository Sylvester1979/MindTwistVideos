
# ======================================================================= story B: the stoker (told as if true)
YB = {'boiler': 1000, 'name': 3000, 'ice': 5000, 'empress': 7000, 'fog': 9000, 'l1915': 11000,
      'torp': 13000, 'sink3': 15000, 'swim': 17000, 'oath': 19000}
PAPER_B = None
def paper_b():
    global PAPER_B
    if PAPER_B is None: PAPER_B = PaperStrip(12000, seed=21)
    return PAPER_B

# ---------------- ships
@functools.lru_cache(None)
def liner(nf, L=760, collier=False, seed=1):
    """ship in profile, bow to the right, waterline at local y=0. returns ink alpha, highlight alpha, anchor-offset (ox, oy).
    raster covers x -0.05L..1.07L, y -0.40L..0.10L."""
    x0, y0, w, h = -0.05 * L, -0.40 * L, 1.12 * L, 0.50 * L
    R = Raster(x0, y0, w, h)
    hull = [(0, -0.07 * L), (0.008 * L, -0.035 * L), (0.03 * L, 0.012 * L), (0.075 * L, 0.055 * L), (0.96 * L, 0.055 * L), (0.99 * L, 0.0), (1.012 * L, -0.085 * L)]
    R.poly(hull)
    Hl = Raster(x0, y0, w, h)
    if not collier:
        R.poly([(0.19 * L, -0.068 * L), (0.80 * L, -0.068 * L), (0.80 * L, -0.115 * L), (0.19 * L, -0.115 * L)])
        R.poly([(0.24 * L, -0.11 * L), (0.75 * L, -0.11 * L), (0.75 * L, -0.15 * L), (0.24 * L, -0.15 * L)])
        xs = np.linspace(0.30, 0.70, nf) if nf > 1 else [0.5]
        for xf in xs:
            fx = xf * L; fw = 0.042 * L
            R.poly([(fx - fw / 2, -0.148 * L), (fx + fw / 2, -0.148 * L), (fx + fw / 2 - 0.014 * L, -0.27 * L), (fx - fw / 2 - 0.014 * L, -0.27 * L)])
            Hl.line([(fx - fw / 2 - 0.012 * L, -0.236 * L), (fx + fw / 2 - 0.012 * L, -0.236 * L)], max(1.2, 0.005 * L))
        for xm in (0.11, 0.9):
            R.line([(xm * L, -0.06 * L), (xm * L - 0.01 * L, -0.36 * L)], max(1.4, 0.0045 * L))
        for k in range(int(0.82 * L / 16)):
            px = 0.08 * L + k * 16
            Hl.ell((px, -0.037 * L), (max(1.0, 0.0045 * L), max(1.0, 0.0045 * L)))
        Hl.line([(0.03 * L, -0.055 * L), (0.985 * L, -0.06 * L)], max(1.0, 0.004 * L))
    else:
        R.poly([(0.07 * L, -0.06 * L), (0.24 * L, -0.06 * L), (0.24 * L, -0.13 * L), (0.07 * L, -0.13 * L)])      # aft house
        R.poly([(0.13 * L, -0.13 * L), (0.19 * L, -0.13 * L), (0.18 * L, -0.25 * L), (0.12 * L, -0.25 * L)])      # funnel
        R.poly([(0.52 * L, -0.06 * L), (0.62 * L, -0.06 * L), (0.62 * L, -0.12 * L), (0.52 * L, -0.12 * L)])      # bridge
        for xm in (0.36, 0.8):
            R.line([(xm * L, -0.06 * L), (xm * L, -0.33 * L)], max(1.4, 0.006 * L))
            R.line([(xm * L, -0.24 * L), (xm * L + 0.1 * L, -0.07 * L)], max(1.0, 0.004 * L))
        Hl.line([(0.03 * L, -0.045 * L), (0.98 * L, -0.05 * L)], max(1.0, 0.004 * L))
    a = R.alpha(0.6, seed)
    hl = Hl.alpha(0.3, seed + 1) * (a > 0.5)
    return a, hl, (-x0, -y0)

def draw_ship(img, cam, nf, L, x, wl, rot=0.0, sink=0.0, op=1.0, collier=False, wipe=1.0, seed=1):
    a, hl, (ox, oy) = liner(nf, L, collier, seed)
    if wipe < 1:
        cut = int(a.shape[1] * wipe)
        a = a.copy(); hl = hl.copy(); a[:, cut:] = 0; hl[:, cut:] = 0
    anc = (ox + 0.5 * L, oy)
    blit(img, a, cam, x + 0.5 * L, wl + sink, INK, 0.95 * op, rot=rot, anchor=anc)
    blit(img, hl, cam, x + 0.5 * L, wl + sink, C(222, 214, 200), 0.9 * op, rot=rot, anchor=anc)

@functools.lru_cache(None)
def wave_strokes(y, x0=0, x1=W, rows=6, gap=15, seed=0, big=1.0):
    r = np.random.default_rng(seed)
    s = []
    for row in range(rows):
        yy = y + 10 + row * gap * big
        x = x0 + r.uniform(0, 40)
        while x < x1:
            L = r.uniform(18, 34) * big
            s.append(Stroke(bez((x, yy), (x + L / 2, yy - 5 * big), (x + L, yy), 8), max(1.0, (2.2 - row * 0.22) * big), 0.2, int(x + row * 1000 + seed)))
            x += L + r.uniform(16, 40) * big
    return s

def vband(cam, y0, y1, f=140):
    U, V = cam.uv()
    return np.clip(np.minimum((V - y0) / f, (y1 - V) / f), 0, 1).astype(np.float32)

def sea_over(img, base, cam, wl, seed=0, wash_op=0.28, rows=7, y_end=None):
    """everything below the waterline (down to y_end) becomes sea again: restore paper, wash, waves."""
    U, V = cam.uv()
    y_end = wl + 850 if y_end is None else y_end
    m = cv2.GaussianBlur(((V > wl + 2) & (V < y_end)).astype(np.float32), (0, 0), 1.0)[..., None]
    img[:] = img * (1 - m) + base * m
    paint(img, m[..., 0], SEA, wash_op, mode='multiply')
    ink_strokes(img, cam, wave_strokes(wl, 0, W, rows, 15, seed), 1.0, op=0.75)
    ink_strokes(img, cam, [Stroke([(0, wl), (W, wl)], 1.6, 0.6, seed + 7, taper=False)], 1.0, op=0.6)

@functools.lru_cache(None)
def lifebuoy():
    """red and white ring with rope: rgb, alpha (S px)."""
    d = 120
    R = Raster(0, 0, d, d).ell((d / 2, d / 2), (48, 48))
    hole = Raster(0, 0, d, d).ell((d / 2, d / 2), (27, 27)).alpha()
    a = np.clip(R.alpha() - hole, 0, 1)
    hh, ww = a.shape
    Y, X = np.mgrid[0:hh, 0:ww].astype(np.float32) / S - d / 2
    ang = np.arctan2(Y, X)
    seg = (((ang + math.pi / 4) // (math.pi / 2)) % 2).astype(np.float32)
    rgb = np.ones((hh, ww, 3), np.float32) * C(232, 226, 214)
    rgb = rgb * (1 - seg[..., None]) + VERM * seg[..., None]
    r = np.sqrt(X ** 2 + Y ** 2)
    shade = 0.78 + 0.3 * np.cos((r - 37.5) / 10.5 * math.pi / 2) * (0.6 + 0.4 * np.cos(ang + 2.2))
    rgb *= np.clip(shade, 0.5, 1.1)[..., None]
    ol = local_strokes(a.shape, [Stroke(circle_pts(d / 2, d / 2, 48, 0, 2 * math.pi, 60), 1.8, 0.2, 1, taper=False),
                                 Stroke(circle_pts(d / 2, d / 2, 27, 0, 2 * math.pi, 40), 1.4, 0.2, 2, taper=False)])
    put_alpha(rgb, ol, 0, 0, INK, 0.9)
    for k in range(4):                                  # rope grab-lines
        a0 = k * math.pi / 2 + math.pi / 4
        P = [(d / 2 + 50 * math.cos(a0 + u * 0.5), d / 2 + 50 * math.sin(a0 + u * 0.5) + 4 * math.sin(u * math.pi)) for u in np.linspace(-1, 1, 10)]
        rope = local_strokes(a.shape, [Stroke(P, 1.6, 0.1, 10 + k, taper=False)])
        put_alpha(rgb, rope, 0, 0, C(120, 100, 70), 0.9)
    return rgb, a

def draw_buoy(img, cam, t, t_on, x, y, scale=1.0):
    k = prog(t, t_on, t_on + 0.5)
    if k <= 0: return
    rgb, a = lifebuoy()
    s = scale * back(k, 2.0)
    bob = 6 * math.sin((t - t_on) * 2.6)
    blit_rgba(img, rgb, a, cam, x, y + bob, min(1, k * 3), s, math.radians(8 * math.sin((t - t_on) * 1.7)), (60, 60), shadow=0.35, lift=0.2)

def counter(img, cam, t, t0, t1, target, x, y, size=120, plus=True):
    p = eout(prog(t, t0, t1))
    if p <= 0: return
    n = int(round(target * p / 10.0)) * 10 if p < 1 else target
    s = f'{n:,}'.replace(',', '.') + ('+' if plus and p >= 1 else '')
    T_ = _T(s, F_SERIF, size, 600, 2)
    text_at(img, cam, T_, x, y, INK, min(1, p * 4), align='center')

def stamp_time(img, cam, t, t_on, txt, x, y):
    T_ = _T(txt, F_MONO, 40, None, 2, True, sum(map(ord, txt)) % 97)
    text_at(img, cam, T_, x, y, INK, 0.9, n=type_n(T_, t, t_on, 14))

# ---------------- B1: Titanic, the boiler room
@functools.lru_cache(None)
def boiler_strokes(y):
    s = []
    for i, cx in enumerate((200, 540, 880)):
        cy = y + 260
        s.append(Stroke(circle_pts(cx, cy, 160, -2.6, -2.6 + 2 * math.pi), 3.2, 0.8, 10 + i, taper=False))
        s.append(Stroke(circle_pts(cx, cy, 148, -2.6, -2.6 + 2 * math.pi), 1.2, 0.5, 20 + i, taper=False))
        for j, dx in enumerate((-70, 0, 70)):
            s.append(Stroke(circle_pts(cx + dx, cy + 70, 30, 0, 2 * math.pi, 30), 2.2, 0.3, 30 + i * 3 + j, taper=False))
        for k in range(8):                                     # rivets ring
            a = k * math.pi / 4
            s.append(Stroke(circle_pts(cx + 120 * math.cos(a), cy - 20 + 100 * math.sin(a), 3, 0, 2 * math.pi, 8), 2.0, 0, 40 + i * 8 + k, taper=False))
        s.append(Stroke([(cx - 40, cy - 160), (cx - 40, cy - 330)], 2.4, 0.4, 70 + i))   # uptakes
        s.append(Stroke([(cx + 40, cy - 160), (cx + 40, cy - 330)], 2.4, 0.4, 80 + i))
    s.append(Stroke([(0, y + 440), (W, y + 440)], 2.6, 0.6, 90, taper=False))           # stokehold plates
    for k in range(0, W, 60):
        s.append(Stroke([(k, y + 440), (k + 30, y + 470)], 1.0, 0.2, 100 + k))
    return s

@functools.lru_cache(None)
def stoker_parts():
    """silhouette: body (static) + arm/shovel (rotating about shoulder). local units, feet at (0,0)."""
    B = Raster(-80, -260, 160, 262)
    B.ell((10, -224), (19, 23))                                       # head, leaning
    B.poly([(-14, -200), (34, -196), (40, -120), (18, -70), (-26, -74), (-30, -140)])
    B.line([(-12, -76), (-40, -38), (-44, 0)], 20)                     # legs
    B.line([(14, -74), (32, -36), (44, 0)], 20)
    A = Raster(-200, -120, 320, 240)                                   # arm + shovel, pivot at (0,0)
    A.line([(0, 0), (52, 46), (96, 60)], 15)
    A.line([(70, 40), (190, 92)], 6)                                   # shaft
    A.poly([(180, 78), (226, 92), (218, 124), (174, 112)])             # blade
    return B.alpha(0.6, 3), A.alpha(0.6, 4)

def draw_boiler(img, cam, t):
    y = YB['boiler']
    t0 = b0('b_1912')
    k = eout(prog(t, t0 - 0.2, t0 + 0.6))
    T_ = _T('ΤΙΤΑΝΙΚΟΣ', F_SERIF, 92, 600, 20)
    text_at(img, cam, T_, 540, y - 700 + 14 * (1 - k), INK, k, align='center')
    D = _T('ΑΠΡΙΛΙΟΣ · 1912', F_MONO, 44, None, 2, True, 22)
    text_at(img, cam, D, 540, y - 612, INK, 0.92, n=type_n(D, t, t0 + 0.3, 14), align='center')
    draw_ship(img, cam, 4, 880, 100, y - 330, wipe=eout(prog(t, t0, t0 + 1.4)), seed=3)
    ink_strokes(img, cam, wave_strokes(y - 330, 40, 1040, 2, 12, 5), eout(prog(t, t0 + 0.6, t0 + 1.6)), op=0.6)
    # the cut-away: boilers
    ts = b0('b_stoker')
    p = eout(prog(t, ts - 0.4, ts + 1.4))
    ink_strokes(img, cam, boiler_strokes(y), p, op=0.9)
    # furnace glow, flickering
    g = ease(prog(t, ts + 0.6, ts + 1.6))
    if g > 0:
        for i, cx in enumerate((200, 540, 880)):
            for j, dx in enumerate((-70, 0, 70)):
                fl = 0.75 + 0.25 * math.sin(t * (9 + i * 2.3 + j)) * math.sin(t * (5.1 + j))
                sx, sy = cam.pt(cx + dx, y + 330)
                lay = np.zeros((H, W), np.float32)
                cv2.circle(lay, (int(sx * 8), int(sy * 8)), int(27 * cam.z * 8), 1.0, -1, cv2.LINE_AA, shift=3)
                paint(img, lay, AMBER, 0.85 * g * fl, mode='multiply')
                paint(img, radial(int(sx), int(sy), int(110 * cam.z)), AMBER, 0.20 * g * fl, mode='screen')
    # the stoker, shovelling coal into the left boiler
    ks = ease(prog(t, ts + 0.3, ts + 1.0))
    if ks > 0:
        body, arm = stoker_parts()
        fx, fy = 300, y + 440
        ph = (t - ts) / 1.15
        sw = math.sin(ph * 2 * math.pi)
        blit(img, body, cam, fx, fy, INK, 0.95 * ks, anchor=(80, 260), scale=1.0)
        blit(img, arm, cam, fx + 6, fy - 180, INK, 0.95 * ks, anchor=(200, 120), rot=math.radians(-25 + 32 * sw), scale=(-1.0, 1.0))
        # coal flying into the furnace door on each swing
        u = (ph % 1.0)
        if 0.15 < u < 0.55 and ks > 0.9:
            q = (u - 0.15) / 0.4
            for j in range(5):
                qq = clamp01(q - j * 0.05)
                px = lerp(fx - 190, 200 + 70, qq); py = lerp(fy - 120, y + 330, qq) - 90 * math.sin(qq * math.pi)
                dot = np.zeros((H, W), np.float32)
                sx, sy = cam.pt(px + 6 * j, py + 4 * (j % 2))
                cv2.circle(dot, (int(sx * 8), int(sy * 8)), int((4 + j % 3) * cam.z * 8), 1.0, -1, cv2.LINE_AA, shift=3)
                paint(img, dot, INK, 0.9 * (1 - qq * 0.3))
    # coal heap
    kh = ease(prog(t, ts, ts + 0.8))
    if kh > 0:
        U, V = cam.uv()
        heap = ((V - (y + 440)) > -110 * np.exp(-((U - 40) / 120) ** 2)) & (V < y + 442) & (U < 260)
        paint(img, heap.astype(np.float32) * hatch_uv(cam, math.radians(40), 5, 2.4, 9), INK, 0.8 * kh)

# ---------------- B2: his name
@functools.lru_cache(None)
def tower_tag():
    w, h = 560, 300
    rgb, a = card_paper(w, h, 141, base=C(218, 196, 150), edge=0.2)
    hole = Raster(0, 0, w, h).ell((40, h / 2), (13, 13)).alpha()
    a = a * (1 - hole)
    ring = local_strokes(a.shape, [Stroke(circle_pts(40, h / 2, 19, 0, 2 * math.pi, 40), 1.4, 0.1, 1)])
    put_alpha(rgb, ring, 0, 0, C(120, 100, 70), 0.7)
    hh, ww = a.shape
    soot = np.clip((fbm(hh, ww, 14 * S, 3, 3) - 0.45) * 3, 0, 1)                 # a sooty thumbprint
    Y, X = np.mgrid[0:hh, 0:ww].astype(np.float32) / S
    thumb = np.exp(-(((X - 470) / 38) ** 2 + ((Y - 240) / 48) ** 2))
    rings = 0.5 + 0.5 * np.sin(np.sqrt((X - 470) ** 2 + ((Y - 240) * 0.8) ** 2) * 1.3)
    put_alpha(rgb, thumb * rings * 0.6 + soot * 0.12, 0, 0, C(60, 56, 54), 0.8)
    return rgb, a

def tower_tag_rgba(t, t0):
    rgb, a = tower_tag(); rgb = rgb.copy()
    k = prog(t, t0 + 0.2, t0 + 0.7)
    put_text_local(rgb, _T('FRANK TOWER', F_MONOB, 54, None, 3, True, 7), 90, 100, INK, n=int(11 * k))
    N1 = _T('Φρανκ Τάουερ', F_MONO, 30, None, 0, True, 3)
    put_text_local(rgb, N1, 92, 166, INK, n=type_n(N1, t, t0 + 0.5, 20))
    N2 = _T('θερμαστής', F_MONO, 30, None, 0, True, 4)
    put_text_local(rgb, N2, 92, 216, INK, n=type_n(N2, t, t0 + 1.0, 20))
    return rgb, a

def draw_name(img, cam, t):
    y = YB['name']
    t0 = b0('b_name') - 0.3
    u = spring(max(0, t - t0) / 0.85, 1.1, 5)
    if u > 0:
        rgb, a = tower_tag_rgba(t, t0)
        blit_rgba(img, rgb, a, cam, lerp(1600, 540, u), y, 1.0, 1.45, math.radians(lerp(18, -3, u)), (280, 150), shadow=0.7, lift=0.8)

# ---------------- B3: the iceberg
@functools.lru_cache(None)
def iceberg(y):
    wl = y + 160
    top = [(700, wl), (735, wl - 120), (770, wl - 160), (800, wl - 290), (830, wl - 250), (870, wl - 330), (905, wl - 240), (950, wl - 200), (990, wl - 80), (1030, wl)]
    R = Raster(680, wl - 360, 380, 380).poly(top)
    sh = Raster(680, wl - 360, 380, 380).poly([(870, wl - 330), (905, wl - 240), (950, wl - 200), (990, wl - 80), (1030, wl), (860, wl)])
    st = [Stroke(top, 2.6, 0.6, 5, taper=False)]
    under = Stroke([(700, wl), (660, wl + 110), (760, wl + 260), (930, wl + 280), (1060, wl + 160), (1030, wl)], 1.4, 1.0, 6, taper=False)
    return R.alpha(0.6, 3), sh.alpha(0.6, 4), (680, wl - 360), st, under

def sink_motion(t, t_hit, t_sink0, t_sink1, max_rot, depth, bow_down=True):
    jolt = 0.0
    if t >= t_hit:
        q = t - t_hit
        jolt = 3.0 * math.exp(-q * 8) * math.sin(q * 60)
    p = ein(prog(t, t_sink0, t_sink1)) if t_sink1 > t_sink0 else 0
    rot = math.radians(max_rot) * min(1, p * 1.6) * (1 if bow_down else -1)
    return rot, depth * p, jolt

def draw_ice(img, cam, t):
    y = YB['ice']; wl = y + 160
    t0, t1 = b0('b_ice'), b1('b_ice')
    t_hit = t0 + 0.72 * (t1 - t0)
    base = img.copy()
    # night: a cool wash over the sky + a few stars
    paint(img, vband(cam, y - 950, y + 1000), SEA, 0.30, mode='multiply')
    stamp_time(img, cam, t, t0 + 0.2, '23:40', 160, y - 560)
    ice_a, ice_sh, (ix, iy), ist, under = iceberg(y)
    blit(img, ice_a, cam, ix, iy, C(236, 240, 242), 0.9)
    blit(img, ice_sh * hatch_layer(*ice_sh.shape, math.radians(60), 7, 1.6, 2), cam, ix, iy, INK, 0.6)
    ink_strokes(img, cam, ist, 1.0, op=0.9)
    # the ship comes in from the left and meets the ice
    xs = lerp(-420, -10, eout(prog(t, t0 - 0.5, t_hit)))
    rot, depth, jolt = sink_motion(t, t_hit, b0('b_1500') - 0.2, b1('b_1500') + 1.2, 11, 230)
    draw_ship(img, cam, 4, 760, xs + jolt, wl, rot=rot, sink=depth, seed=3)
    sea_over(img, base, cam, wl, seed=11, wash_op=0.35)
    ink_strokes(img, cam, [under], 1.0, op=0.3)
    if t > t_hit:
        k = 1 - prog(t, t_hit + 0.3, t_hit + 1.2)
        sc = [Stroke([(712 + 6 * j, wl - 40 + 14 * j), (690 + 6 * j, wl - 34 + 14 * j)], 1.8, 0.1, 80 + j) for j in range(4)]
        ink_strokes(img, cam, sc, 1.0, op=0.8 * k)
    counter(img, cam, t, b0('b_1500'), b1('b_1500') - 0.3, 1500, 540, y - 330)
    draw_buoy(img, cam, t, b0('b_saved1') - 0.05, 300, wl + 120, 1.15)

# ---------------- B4: Empress of Ireland
@functools.lru_cache(None)
def ticket_b():
    w, h = 640, 290
    rgb, a = card_paper(w, h, 171, base=C(214, 206, 186), edge=0.18)
    hole = Raster(0, 0, w, h).ell((560, 60), (14, 14)).alpha()
    a = a * (1 - hole)
    dash = [Stroke([(470, yy), (470, yy + 9)], 1.4, 0.0, yy) for yy in range(14, h - 14, 18)]
    put_alpha(rgb, local_strokes(a.shape, dash), 0, 0, INK, 0.6)
    put_text_local(rgb, _T('QUEBEC', F_SERIF, 58, 700, 3), 40, 110, INK)
    put_text_local(rgb, _T('→ LIVERPOOL', F_SERIF, 58, 700, 3), 40, 176, INK)
    put_text_local(rgb, _T('EMPRESS OF IRELAND', F_MONO, 22, None, 1, True, 8), 44, 232, INK)
    put_text_local(rgb, _T('28.5.1914', F_MONO, 22, None, 1, True, 9), 494, 150, INK, 0.8)
    put_text_local(rgb, _T('CREW', F_MONO, 22, None, 1, True, 10), 494, 190, INK, 0.8)
    return rgb, a

def draw_empress(img, cam, t):
    y = YB['empress']
    t0, t1 = b0('b_again'), b1('b_again')
    u = spring(max(0, t - (t0 - 0.3)) / 0.85, 1.0, 5)
    k = eout(prog(t, t0 - 0.1, t0 + 0.5))
    T_ = _T('1914', F_SERIF, 120, 600, 6)
    text_at(img, cam, T_, 540, y - 560 + 14 * (1 - k), INK, k, align='center')
    rgb, a = ticket_b()
    blit_rgba(img, rgb, a, cam, lerp(1500, 540, u), y - 250, 1.0, 1.0, math.radians(lerp(16, 3, u)), (320, 145), shadow=0.7, lift=0.7)
    draw_ship(img, cam, 2, 820, 130, y + 330, wipe=eout(prog(t, t0 + 0.6 * (t1 - t0) - 0.6, t1 + 0.4)), seed=5)
    ink_strokes(img, cam, wave_strokes(y + 330, 60, 1020, 3, 13, 6), eout(prog(t, t1 - 0.6, t1 + 0.6)), op=0.6)

# ---------------- B5: fog, the collier, 14 minutes
@functools.lru_cache(None)
def fog_bank():
    return fbm(1100, 2600, 220, 8, 5)

def fog_layer(cam, t, dens):
    f = fog_bank()
    U, V = cam.uv()
    u = ((U + t * 40) % 2600).astype(np.float32); v = np.abs(((V - (YB['fog'] - 550)) % 2198) - 1099).astype(np.float32)
    n = cv2.remap(f, u, v, cv2.INTER_LINEAR)
    return np.clip((n - 0.42) * 2.6 + 0.35, 0, 1) * dens

def draw_fog(img, cam, t):
    y = YB['fog']; wl = y + 200
    t0 = b0('b_fog'); th0, th1 = b0('b_hit'), b1('b_hit')
    t_hit = th0 + 0.62 * (th1 - th0)
    base = img.copy()
    paint(img, vband(cam, y - 950, y + 1000), SEA, 0.25, mode='multiply')
    stamp_time(img, cam, t, t0 + 0.2, '01:56', 160, y - 560)
    S_ = _T('ΑΓΙΟΣ ΛΑΥΡΕΝΤΙΟΣ · ΚΑΝΑΔΑΣ', F_MONO, 30, None, 2, True, 44)
    text_at(img, cam, S_, 540, y + 640, INK, 0.85, n=type_n(S_, t, t0 + 0.4, 18), align='center')
    rot, depth, jolt = sink_motion(t, t_hit, b0('b_14') - 0.3, b1('b_1000') + 0.6, -16, 260, bow_down=False)
    # the Empress, listing as she goes; the collier backs away after the blow
    draw_ship(img, cam, 2, 700, 120 + jolt, wl, rot=rot, sink=depth, seed=5)
    cx = lerp(1200, 540, eout(prog(t, t0 - 0.3, t_hit))) + 260 * ein(prog(t, t_hit + 0.4, t_hit + 3.0))
    a, hl, (ox, oy) = liner(1, 420, True, 9)
    halo = cv2.dilate(a, np.ones((9, 9), np.uint8))
    blit(img, halo, cam, cx, wl + 30, C(214, 208, 198), 0.95, anchor=(ox + 0.5 * 420, oy), scale=(-1.0, 1.0))
    blit(img, a, cam, cx, wl + 30, C(58, 54, 52), 0.95, anchor=(ox + 0.5 * 420, oy), scale=(-1.0, 1.0))
    sea_over(img, base, cam, wl + 30, seed=13, wash_op=0.32)
    if t > t_hit:
        k = 1 - prog(t, t_hit + 0.2, t_hit + 1.0)
        sx, sy = cam.pt(cx - 210, wl - 30)
        burst = [Stroke([(cx - 210 + 20 * math.cos(q), wl - 40 + 20 * math.sin(q)), (cx - 210 + 60 * math.cos(q), wl - 40 + 60 * math.sin(q))], 2.0, 0.2, int(q * 100)) for q in np.linspace(0, 2 * math.pi, 11)[:-1]]
        ink_strokes(img, cam, burst, 1.0, op=0.85 * k)
    dens = 0.9 * (1 - 0.55 * ease(prog(t, t_hit, b1('b_14'))))
    paint(img, fog_layer(cam, t, dens) * vband(cam, y - 950, y + 1000), C(226, 222, 214), 1.0)
    # 14 minutes
    k = eout(prog(t, b0('b_14') + 0.1, b0('b_14') + 0.6))
    if k > 0:
        M_ = _T('14′', F_SERIF, 170, 600)
        text_at(img, cam, M_, 540, y - 330 + 20 * (1 - k), INK, k, align='center')
    counter(img, cam, t, b0('b_1000'), b1('b_1000') - 0.3, 1000, 540, y - 150, 110)
    draw_buoy(img, cam, t, b0('b_saved2') - 0.05, 760, wl + 150, 1.15)

# ---------------- B6: 1915, Lusitania
def draw_1915(img, cam, t):
    y = YB['l1915']
    t0 = b0('b_1915')
    k = eout(prog(t, t0 - 0.1, t0 + 0.5))
    T_ = _T('1915', F_SERIF, 170, 600, 8)
    text_at(img, cam, T_, 540, y - 520 + 16 * (1 - k), INK, k, align='center')
    D = _T('ΠΡΩΤΟΣ ΠΑΓΚΟΣΜΙΟΣ ΠΟΛΕΜΟΣ', F_MONO, 38, None, 2, True, 51)
    text_at(img, cam, D, 540, y - 420, INK, 0.9, n=type_n(D, t, t0 + 0.7, 22), align='center')
    tl = b0('b_lusi')
    draw_ship(img, cam, 4, 900, 90, y + 40, wipe=eout(prog(t, tl - 0.3, tl + 1.4)), seed=7)
    ink_strokes(img, cam, wave_strokes(y + 40, 40, 1040, 3, 13, 8), eout(prog(t, tl + 0.4, tl + 1.4)), op=0.6)
    L_ = _T('LUSITANIA', F_SERIF, 54, 600, 14)
    text_at(img, cam, L_, 540, y + 160, INK, 0.9 * ease(prog(t, tl + 0.4, tl + 1.0)), align='center')
    u = spring(max(0, t - (tl + 0.6)) / 0.8, 1.1, 5)
    if u > 0:
        rgb, a = tower_tag_rgba(t, -10)
        blit_rgba(img, rgb, a, cam, lerp(1600, 640, u), y + 470, 1.0, 1.0, math.radians(lerp(18, 4, u)), (280, 150), shadow=0.7, lift=0.8)

# ---------------- B7: the torpedo (map of southern Ireland)
IL0, IL1, IA0, IA1 = -10.6, -5.6, 50.85, 52.75
IKX = 980 / (IL1 - IL0); IKY = IKX / math.cos(math.radians(51.7))
IMX, IMY = 50, YB['torp'] - 600
def iproj(lon, lat): return IMX + (lon - IL0) * IKX, IMY + (IA1 - lat) * IKY
WRECK = iproj(-8.55, 51.417); KINSALE = iproj(-8.53, 51.60)

@functools.lru_cache(None)
def ireland():
    gj = json.load(open(f'{ROOT}/map/ne_10m_land.geojson'))
    runs, rings = [], []
    m = 0.4
    for f in gj['features']:
        g = f['geometry']
        polys = g['coordinates'] if g['type'] == 'MultiPolygon' else [g['coordinates']]
        for poly in polys:
            R = np.array(poly[0])
            inb = (R[:, 0] > IL0 - m) & (R[:, 0] < IL1 + m) & (R[:, 1] > IA0 - m) & (R[:, 1] < IA1 + m)
            if not inb.any(): continue
            P = np.stack(iproj(R[:, 0], R[:, 1]), 1)
            rings.append(P)
            idx = np.where(inb)[0]; br = np.where(np.diff(idx) > 1)[0]
            for seg in np.split(idx, br + 1):
                if len(seg) < 4: continue
                Q = P[seg]
                if np.hypot(*np.diff(Q, axis=0).T).sum() < 14: continue
                runs.append(Q)
    st = [Stroke(Q, 2.4, 0.6, 500 + i, taper=False) for i, Q in enumerate(runs)]
    st.sort(key=lambda s: -s.len)
    Rr = Raster(IMX - 40, IMY - 40, 1060, (IA1 - IA0) * IKY + 80)
    for P in rings: Rr.poly(P)
    land = Rr.alpha()
    sea = 1 - (land > 0.5).astype(np.uint8)
    d = cv2.distanceTransform(sea, cv2.DIST_L2, 5) / S
    wl = np.zeros_like(land)
    for dist, op in ((7, 0.75), (14, 0.5), (22, 0.32), (31, 0.2)):
        wl = np.maximum(wl, np.clip(1.2 - np.abs(d - dist) * S * 0.6, 0, 1) * op)
    return st, wash(land, 5, 0.25, 0.3), wl, (Rr.x0, Rr.y0)

@functools.lru_cache(None)
def uboat():
    R = Raster(-70, -30, 140, 50)
    R.ell((0, 6), (64, 9)).poly([(-14, 2), (16, 2), (12, -16), (-10, -16)]).line([(2, -16), (2, -28)], 2.2)
    return R.alpha(0.3, 6)

def draw_torp(img, cam, t):
    y = YB['torp']
    t0, t1 = b0('b_torp'), b1('b_torp')
    st, wsh, wl, (x0, y0) = ireland()
    op = ease(prog(t, t0 - 0.9, t0 - 0.1))
    blit(img, wsh, cam, x0, y0, PAPER_D, 0.55 * op, mode='multiply')
    blit(img, wl, cam, x0, y0, SEA, 0.85 * op, mode='multiply')
    ink_strokes(img, cam, st, eout(prog(t, t0 - 0.9, t0 + 0.6)), op=0.92)
    T_ = _T('ΙΡΛΑΝΔΙΑ', F_SERIF, 64, 600, 16)
    text_at(img, cam, T_, 560, IMY + 260, INK, 0.85 * op, align='center')
    K_ = _T('Κινσέιλ', F_SERIF_I, 34, 500)
    text_at(img, cam, K_, KINSALE[0] + 24, KINSALE[1] - 14, INK, 0.9 * op)
    stamp_time(img, cam, t, t0 + 0.2, '14:10', 160, y - 660)
    # the liner, a small silhouette heading east along the coast
    sx = lerp(WRECK[0] - 160, WRECK[0], eout(prog(t, t0 - 0.6, t1 - 0.3)))
    a, hl, (ox, oy) = liner(4, 150, False, 7)
    blit(img, a, cam, sx, WRECK[1], INK, 0.95 * op, anchor=(ox + 75, oy))
    # U-20 to the south, the torpedo track
    ux, uy = WRECK[0] + 120, WRECK[1] + 330
    ku = ease(prog(t, t0 + 0.4, t0 + 1.0))
    blit(img, uboat(), cam, ux, uy, INK, 0.9 * ku, anchor=(70, 30), scale=1.4)
    U_ = _T('U-20', F_MONO, 30, None, 1, True, 52)
    text_at(img, cam, U_, ux + 70, uy + 10, INK, 0.9, n=type_n(U_, t, t0 + 0.9, 12))
    tt0, tt1 = t0 + 0.55 * (t1 - t0), t1 - 0.25
    p = prog(t, tt0, tt1)
    if p > 0:
        P = [(lerp(ux - 30, WRECK[0] + 20, q), lerp(uy - 10, WRECK[1] + 12, q)) for q in np.linspace(0, 1, 40)]
        dash = []
        for i in range(0, 38, 3):
            dash.append(Stroke(P[i:i + 2], 2.4, 0.1, 600 + i, taper=False))
        ink_strokes(img, cam, dash, p, op=0.9)
    if t > tt1:
        k = 1 - prog(t, tt1 + 0.2, tt1 + 1.2)
        burst = [Stroke([(WRECK[0] + 20 + 14 * math.cos(q), WRECK[1] + 10 + 14 * math.sin(q)), (WRECK[0] + 20 + 48 * math.cos(q), WRECK[1] + 10 + 48 * math.sin(q))], 2.2, 0.2, int(q * 100) + 9) for q in np.linspace(0, 2 * math.pi, 13)[:-1]]
        ink_strokes(img, cam, burst, 1.0, op=0.9 * k)
        sx_, sy_ = cam.pt(WRECK[0] + 20, WRECK[1] + 10)
        paint(img, radial(int(sx_), int(sy_), int(120 * cam.z)), AMBER, 0.5 * k, mode='screen')

# ---------------- B8: eighteen minutes
def draw_sink3(img, cam, t):
    y = YB['sink3']; wl = y + 160
    base = img.copy()
    t0 = b0('b_18')
    rot, depth, jolt = sink_motion(t, t0 - 0.6, t0 - 0.3, b1('b_1200') + 0.8, 18, 300)
    draw_ship(img, cam, 4, 820, 130 + jolt, wl, rot=rot, sink=depth, seed=7)
    sea_over(img, base, cam, wl, seed=17, wash_op=0.32)
    k = eout(prog(t, t0 - 0.05, t0 + 0.5))
    if k > 0:
        M_ = _T('18′', F_SERIF, 170, 600)
        text_at(img, cam, M_, 540, y - 400 + 20 * (1 - k), INK, k, align='center')
    counter(img, cam, t, b0('b_1200'), b1('b_1200') - 0.3, 1200, 540, y - 220, 110)

# ---------------- B9: he swims to a boat (seen from above)
@functools.lru_cache(None)
def lifeboat_top():
    R = Raster(-130, -45, 260, 90)
    R.poly([(-120, 0), (-90, -32), (90, -32), (125, 0), (90, 32), (-90, 32)])
    inner = Raster(-130, -45, 260, 90).poly([(-104, 0), (-82, -22), (84, -22), (108, 0), (84, 22), (-82, 22)]).alpha()
    a = R.alpha(0.4, 3)
    th = np.zeros_like(a)
    for k in range(-60, 80, 34):
        cv2.line(th, (int((k + 130) * S), int(23 * S)), (int((k + 130) * S), int(67 * S)), 1.0, int(5 * S))
    oars = Raster(-130, -45, 260, 90).line([(-40, -30), (-60, -44)], 4).line([(40, -30), (20, -44)], 4).line([(-40, 30), (-60, 44)], 4).line([(40, 30), (20, 44)], 4).alpha()
    return np.clip(a - inner * 0.75 + th * inner * 0.8 + oars, 0, 1)

def draw_swim(img, cam, t):
    y = YB['swim']
    t0, t1 = b0('b_swim'), b1('b_swim')
    U, V = cam.uv()
    vb = vband(cam, y - 950, y + 1000)
    paint(img, vb, SEA, 0.38, mode='multiply')
    rip = hatch_uv(cam, math.radians(8), 26, 1.6, 4, jitter=0.6)
    paint(img, rip * vb, INK, 0.12)
    bx, by = 700, y - 360
    blit(img, lifeboat_top(), cam, bx, by + 4 * math.sin(t * 1.8), INK, 0.92, anchor=(130, 45), rot=math.radians(-18), scale=1.5)
    draw_buoy(img, cam, t, t0 - 0.6, bx - 230, by + 110, 1.1)
    # swimmer: head + alternating arms + wake, from lower left to the boat
    p = einout(prog(t, t0 + 0.2, t1 - 0.2))
    sx0, sy0 = 260, y + 520
    hx, hy = lerp(sx0, bx - 120, p), lerp(sy0, by + 60, p)
    ang = math.atan2((by + 60) - sy0, (bx - 120) - sx0)
    dx, dy = math.cos(ang), math.sin(ang)
    for j in range(7):                                       # wake
        d = 44 + j * 40
        w_ = 14 + j * 12
        P = [(hx - dx * d + dy * w_, hy - dy * d - dx * w_), (hx - dx * (d - 14), hy - dy * (d - 14)), (hx - dx * d - dy * w_, hy - dy * d + dx * w_)]
        ink_strokes(img, cam, [Stroke(P, 1.6, 0.2, 700 + j, taper=False)], 1.0, op=0.6 * (1 - j / 7) * (1 if p < 1 else 1 - prog(t, t1, t1 + 0.6)))
    stroke = math.sin(t * 7.0)
    for side in (-1, 1):
        aa = ang + side * (1.2 + 0.7 * stroke * side)
        ax_, ay_ = hx + math.cos(aa) * 58, hy + math.sin(aa) * 58
        ink_strokes(img, cam, [Stroke([(hx - dy * 14 * side, hy + dx * 14 * side), (ax_, ay_)], 8.0, 0.1, 720 + side, taper=False)], 1.0, op=0.9)
    dot = np.zeros((H, W), np.float32)
    sxx, syy = cam.pt(hx, hy)
    cv2.circle(dot, (int(sxx * 8), int(syy * 8)), int(20 * cam.z * 8), 1.0, -1, cv2.LINE_AA, shift=3)
    paint(img, dot, INK, 0.95)
    for j in range(3):                                       # rings around the head
        r = 28 + ((t * 26 + j * 18) % 60)
        ring = np.zeros((H, W), np.float32)
        cv2.circle(ring, (int(sxx * 8), int(syy * 8)), int(r * cam.z * 8), 1.0, 1, cv2.LINE_AA, shift=3)
        paint(img, ring, INK, 0.5 * (1 - (r - 28) / 60))

# ---------------- B10: never again (footsteps away from the sea)
def draw_oath(img, cam, t):
    y = YB['oath']
    t0, t1 = b0('b_oath'), b1('b_oath')
    wl = y + 560
    U, V = cam.uv()
    sea = cv2.GaussianBlur(((V > wl) & (V < y + 1000)).astype(np.float32), (0, 0), 2)
    paint(img, sea, SEA, 0.45, mode='multiply')
    surf = [Stroke([(x, wl + 8 * math.sin(x / 60 + t * 1.5)) for x in range(-20, W + 40, 10)], 2.4, 0.4, 3, taper=False),
            Stroke([(x, wl + 34 + 6 * math.sin(x / 50 + t * 1.2 + 1)) for x in range(-20, W + 40, 10)], 1.6, 0.4, 4, taper=False)]
    ink_strokes(img, cam, surf, 1.0, op=0.7)
    # sand stipple
    r = np.random.default_rng(3)
    pts = np.column_stack([r.uniform(0, W, 900), wl - r.exponential(160, 900)])
    dots = np.zeros((H, W), np.float32)
    for x, yy in pts:
        sx, sy = cam.pt(x, yy)
        if 0 <= sx < W and 0 <= sy < H: dots[int(sy), int(sx)] = 1.0
    paint(img, cv2.GaussianBlur(dots, (0, 0), 0.7) * 2.5, INK, 0.5)
    # field and fence inland
    fence = [Stroke([(60, y - 520), (1020, y - 540)], 2.0, 0.6, 5, taper=False)] + [Stroke([(80 + k * 80, y - 522 - k * 1.6), (80 + k * 80, y - 580 - k * 1.6)], 2.2, 0.2, 10 + k) for k in range(12)]
    kf = eout(prog(t, t0 + 1.0, t1))
    ink_strokes(img, cam, fence, kf, op=0.6)
    U, V = cam.uv()
    field = np.clip(((y - 560) - V) / 30, 0, 1) * np.clip((V - (y - 1000)) / 60, 0, 1)
    paint(img, hatch_uv(cam, math.radians(86), 16, 1.6, 6) * field, INK, 0.35 * kf)
    # footsteps: out of the water, up the beach, inland
    fp = footprint()
    n = int(24 * eout(prog(t, t0 - 0.2, t1 + 0.3)))
    for i in range(n):
        yy = wl + 20 - i * 44
        x = 520 + 14 * math.sin(i * 0.35) + (-14 if i % 2 else 14)
        op = 0.85 * (0.5 + 0.5 * (i / 24)) if yy < wl else 0.3
        blit(img, fp, cam, x, yy, INK, op, rot=0.0, anchor=(14, 22))

def story_b_cam(t):
    Y = YB
    keys = [(b0('b_1912') - 0.3, 540, Y['boiler'] - 500, 1.0, 0.0),
            (b0('b_stoker') - 0.1, 540, Y['boiler'] - 400, 1.0, 0.0),
            (b0('b_stoker') + 0.9, 540, Y['boiler'] + 160, 1.08, 0.0),
            (b1('b_stoker') + 0.2, 470, Y['boiler'] + 210, 1.25, 0.0),
            (b0('b_name') - 0.5, 470, Y['boiler'] + 220, 1.25, 0.0),
            (b0('b_name') + 0.2, 540, Y['name'], 1.0, 0.0),
            (b1('b_name') + 0.3, 540, Y['name'], 1.05, 0.0),
            (b0('b_ice') - 0.2, 540, Y['ice'] - 100, 1.0, 0.0),
            (b1('b_ice') + 0.2, 600, Y['ice'] - 60, 1.06, 0.0),
            (b1('b_saved1') + 0.2, 520, Y['ice'] + 20, 1.08, 0.0),
            (b0('b_again') - 0.25, 520, Y['ice'] + 40, 1.08, 0.0),
            (b0('b_again') + 0.45, 540, Y['empress'], 1.0, 0.0),
            (b1('b_again') + 0.2, 540, Y['empress'] + 60, 1.04, 0.0),
            (b0('b_fog') - 0.2, 540, Y['fog'] - 40, 1.0, 0.0),
            (b1('b_hit') + 0.2, 560, Y['fog'], 1.05, 0.0),
            (b1('b_saved2') + 0.2, 540, Y['fog'] + 40, 1.08, 0.0),
            (b0('b_1915') - 0.25, 540, Y['fog'] + 50, 1.08, 0.0),
            (b0('b_1915') + 0.45, 540, Y['l1915'], 1.0, 0.0),
            (b1('b_lusi') + 0.2, 540, Y['l1915'] + 60, 1.04, 0.0),
            (b0('b_torp') - 0.2, 540, Y['torp'] - 60, 1.0, 0.0),
            (b1('b_torp') + 0.2, 560, Y['torp'] + 20, 1.12, 0.0),
            (b0('b_18') - 0.3, 560, Y['torp'] + 30, 1.12, 0.0),
            (b0('b_18') + 0.35, 540, Y['sink3'], 1.0, 0.0),
            (b1('b_1200') + 0.3, 540, Y['sink3'] + 60, 1.05, 0.0),
            (b0('b_swim') - 0.3, 540, Y['sink3'] + 70, 1.05, 0.0),
            (b0('b_swim') + 0.4, 540, Y['swim'], 1.0, 0.0),
            (b1('b_swim') + 0.1, 560, Y['swim'] - 120, 1.08, 0.0),
            (b0('b_oath') - 0.35, 560, Y['swim'] - 130, 1.08, 0.0),
            (b0('b_oath') + 0.35, 540, Y['oath'] + 120, 1.0, 0.0),
            (b1('b_oath') + 1.0, 540, Y['oath'] - 160, 1.0, 0.0)]
    return cam_path(t, keys)

def story_b(t):
    cam = story_b_cam(t)
    img = paper_b().render(cam)
    vis = lambda yc: abs(cam.cy - yc) < 2100 / cam.z
    for key, fn in (('boiler', draw_boiler), ('name', draw_name), ('ice', draw_ice), ('empress', draw_empress), ('fog', draw_fog),
                    ('l1915', draw_1915), ('torp', draw_torp), ('sink3', draw_sink3), ('swim', draw_swim), ('oath', draw_oath)):
        if vis(YB[key]): fn(img, cam, t)
    return img, cam
