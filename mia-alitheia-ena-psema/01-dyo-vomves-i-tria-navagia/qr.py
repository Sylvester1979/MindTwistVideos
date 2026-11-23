
# ======================================================================= the question, the answer, the explanations
_C5 = None
def c5():
    global _C5
    if _C5 is None:
        import cards05
        _C5 = cards05
    return _C5

@functools.lru_cache(None)
def card_final(which):
    T5 = c5()
    if which == 'A':
        return T5.make_card('Α', T5.CLAIM_A, T5.draw_two_bombs, 3)
    return T5.make_card('Β', T5.CLAIM_B, T5.draw_three_wrecks, 7)

@functools.lru_cache(None)
def card_blank(which):
    T5 = c5()
    seed = 3 if which == 'A' else 7
    img = T5.card_paper(seed)
    ring = T5.Pen(T5.CW, T5.CH)
    ring.stroke([(70 + 52 * math.cos(a), 92 + 52 * math.sin(a)) for a in np.linspace(0.2, 2 * math.pi + 0.5, 60)], 4, 2.5)
    T5.put(img, T5.ink_tex(ring.alpha(seed=seed), seed), 0, 0, T5.INKC)
    la = T5.text_alpha('Α' if which == 'A' else 'Β', T5.SERIFB, 74)
    T5.put(img, la, 70 - la.shape[1] // 2, 92 - la.shape[0] // 2 + 2, T5.INKC)
    return img

def card_building(which, p):
    """card filling itself: typed lines one after the other, then the rule and the drawing (p 0..1)."""
    import textwrap
    T5 = c5()
    claim = T5.CLAIM_A if which == 'A' else T5.CLAIM_B
    lines = textwrap.wrap(claim, 29)
    fin, blank = card_final(which), card_blank(which)
    m = np.zeros(fin.shape[:2], np.float32)
    nch = sum(len(l) for l in lines)
    n = nch * clamp01(p / 0.6)
    y = 44
    for ln in lines:
        k = clamp01(n / len(ln)); n -= len(ln)
        x1 = 150 + k * len(ln) * 23.0
        m[max(0, y - 8):y + 46, :int(x1)] = 1.0
        y += 52
    y_rule = y + 22
    pr = clamp01((p - 0.6) / 0.12)
    m[y_rule - 6:y_rule + 6, :int(40 + pr * (T5.CW - 80))] = 1.0
    pd = clamp01((p - 0.68) / 0.32)
    if pd > 0:
        X = np.arange(fin.shape[1], dtype=np.float32)[None, :]
        edge = lerp(-60, fin.shape[1] + 60, eout(pd))
        md = np.clip((edge - X) / 60, 0, 1) * np.ones((fin.shape[0], 1), np.float32)
        md[:y_rule + 8] = 0
        m = np.maximum(m, md)
    m[:y_rule - 8, :140] = 1.0
    m = m[..., None]
    return blank * (1 - m) + fin * m

def place(img, cam, card, which, dx=0, dy=0, scale=1.0, lift=0.6, drot=0.0):
    x, y, r = card_pose_final(which)
    a = np.ones(card.shape[:2], np.float32)
    blit_rgba(img, card, a, cam, x + dx, y + dy, 1.0, scale, r + drot, (CW / 2, CH / 2), shadow=0.8, lift=lift, s=1)

def header_q(img, cam, op, txt='Ποιο είναι αληθινό;', y=190, size=70):
    T1 = _T('ΜΙΑ ΑΛΗΘΕΙΑ  ·  ΕΝΑ ΨΕΜΑ', F_SERIF, 40, 500, 5)
    text_at(img, cam, T1, W / 2, 112, C(236, 226, 210), 0.85, align='center')
    if op > 0:
        T2 = _T(txt, F_SERIF, size, 500)
        text_at(img, cam, T2, W / 2, y + 16 * (1 - op), C(240, 232, 218), op, align='center')

def t_bridge_b(): return b1('b_oath') + 0.3

def desk_b_forming(t):
    cam = Cam(W / 2, H / 2, 1.0)
    img = desk_bg().copy()
    place(img, cam, card_final('A'), 'A')
    p = prog(t, t_bridge_b() + 0.2, b1('q_so') + 0.3)
    u = spring(prog(t, t_bridge_b(), t_bridge_b() + 0.9) * 1.0, 1.0, 5)
    x, y, r = card_pose_final('B')
    a = np.ones((CH, CW), np.float32)
    card = card_building('B', p)
    blit_rgba(img, card, a, cam, lerp(1800, x, u), lerp(1600, y, u), 1.0, 1.0, lerp(0.3, r, u), (CW / 2, CH / 2), shadow=0.8, lift=0.6, s=1)
    header_q(img, cam, eout(prog(t, b0('q_so') - 0.2, b0('q_so') + 0.5)))
    return img

def countdown(img, t, t0):
    """5..1 in a paper disc between the cards; a pen arc closes each second."""
    q = t - t0
    if q < 0 or q >= 5.0: return
    k = int(q); f = q - k
    num = 5 - k
    cx, cy, R = W / 2, 975, 120
    a_in = eout(prog(q, 0, 0.3))
    disc = np.zeros((H, W), np.float32)
    cv2.circle(disc, (int(cx), int(cy)), int(R * a_in), 1.0, -1, cv2.LINE_AA)
    paint(img, cv2.GaussianBlur(disc, (0, 0), 18), np.float32([0, 0, 0]), 0.5)
    paint(img, disc, PAPER, 0.97)
    cam = Cam(W / 2, H / 2, 1.0)
    st = Stroke(circle_pts(cx, cy, R - 16, -math.pi / 2, -math.pi / 2 + 2 * math.pi, 90), 5.0, 0.3, k, taper=False)
    ink_strokes(img, cam, [st], eout(f), op=0.85)
    N = _T(str(num), F_SERIF, 150, 600)
    pop = back(clamp01(f / 0.25), 2.0)
    text_at(img, cam, N, cx, cy + 52, INK, 0.95 * a_in, align='center', scale=0.8 + 0.2 * pop)

def desk_question(t):
    cam = Cam(W / 2, H / 2, 1.0)
    img = desk_bg().copy()
    tq0, tq1 = b0('q_which'), b1('q_which')
    mid = tq0 + 0.5 * (tq1 - tq0)
    la = math.sin(math.pi * prog(t, tq0 - 0.1, mid + 0.1))
    lb = math.sin(math.pi * prog(t, mid - 0.1, tq1 + 0.3))
    tc = b1('q_true') + 0.15
    sq = eout(prog(t, tc - 0.4, tc + 0.3)) * (1 - eout(prog(t, tc + 5.0, tc + 5.4)))
    for which, l in (('A', la), ('B', lb)):
        dy = (-90 if which == 'A' else 90) * sq
        place(img, cam, card_final(which), which, dy=dy, scale=(1 + 0.03 * l) * (1 - 0.12 * sq), lift=0.6 + 1.4 * l)
    header_q(img, cam, eout(prog(t, b0('q_so') - 0.2, b0('q_so') + 0.5)))
    countdown(img, t, tc)
    return img

def t_stamp(): return b0('r_true') + 0.55 * (b1('r_true') - b0('r_true'))

def reveal_cards(t, cam=None, hdr=True):
    """both cards; red stamp slams on A, ink X crosses B. returns img."""
    ca = cardanim()
    cam = cam or Cam(W / 2, H / 2, 1.0)
    img = desk_bg().copy()
    ts = t_stamp()
    A = card_final('A').copy()
    A, _ = ca.stamp_on(A, t, ts - 0.14)
    B = card_final('B').copy()
    q1 = prog(t, ts + 0.75, ts + 1.1); q2 = prog(t, ts + 1.15, ts + 1.5)
    if q1 > 0:
        B = B * lerp(1.0, 0.9, q1)
        ca.put(B, ca.diag_wipe(ca.X1, q1 * 1.05, (70, 120), (840, 620)), 0, 0, ca.INKC, 0.92)
    if q2 > 0:
        ca.put(B, ca.diag_wipe(ca.X2, q2 * 1.05, (830, 90), (90, 640)), 0, 0, ca.INKC, 0.92)
    place(img, cam, A, 'A')
    place(img, cam, B, 'B', lift=0.5)
    if hdr: header_q(img, cam, 1.0)
    k = prog(t, ts, ts + 0.3)
    if 0 < k < 1:
        amp = 9 * (1 - k) ** 2
        M = np.float32([[1, 0, amp * math.sin(t * 70)], [0, 1, amp * math.cos(t * 93)]])
        img = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT)
        img = img * (1 - 0.10 * (1 - k)) + 245 * 0.10 * (1 - k)
    return img

# ---------------- R1: officially recognised (a slip clipped to card A)
@functools.lru_cache(None)
def slip():
    w, h = 470, 170
    rgb, a = card_paper(w, h, 201, base=C(238, 234, 222), edge=0.12)
    put_text_local(rgb, _T('二重被爆', F_CJK, 54, None, 8, False, 0, 0), 30, 76, INK)
    put_text_local(rgb, _T('διπλός επιζών · 2009', F_MONO, 30, None, 0, True, 4), 32, 132, INK)
    return rgb, a

def desk_recognised(t):
    t0 = b0('r_only')
    z = lerp(1.0, 1.35, einout(prog(t, t0 - 0.2, t0 + 1.0)))
    cam = Cam(lerp(W / 2, 540, z - 1), lerp(H / 2, 700, (z - 1) / 0.35), z)
    img = reveal_cards(t, cam, hdr=False)
    u = spring(max(0, t - (t0 + 0.6)) / 0.8, 1.1, 5)
    if u > 0:
        rgb, a = slip()
        blit_rgba(img, rgb, a, cam, lerp(1500, 700, u), 880, 1.0, 1.0, math.radians(lerp(14, -5, u)), (235, 85), shadow=0.7, lift=0.6)
    return img

# ---------------- explanation scroll
YR = {'house': 1000, 'lists': 3000, 'clark': 5000, 'three': 7000, 'shore': 9000}
PAPER_R = None
def paper_r():
    global PAPER_R
    if PAPER_R is None: PAPER_R = PaperStrip(12000, seed=31)
    return PAPER_R

@functools.lru_cache(None)
def tin():
    """a small round tin of ointment, three-quarter view."""
    w, h = 150, 110
    R = Raster(0, 0, w, h).ell((75, 70), (66, 30)).poly([(9, 40), (141, 40), (141, 70), (9, 70)]).ell((75, 40), (66, 28))
    a = R.alpha(0.4, 2)
    rgb = np.ones(a.shape + (3,), np.float32) * C(176, 170, 160)
    lid = Raster(0, 0, w, h).ell((75, 40), (66, 28)).alpha()
    put_alpha(rgb, lid, 0, 0, C(212, 206, 194), 1.0)
    put_alpha(rgb, local_strokes(a.shape, [Stroke(circle_pts(75, 40, 66, 0, 2 * math.pi, 60), 1.6, 0.2, 1, taper=False),
                                          Stroke([(9, 40), (9, 70)], 1.6, 0.1, 2), Stroke([(141, 40), (141, 70)], 1.6, 0.1, 3),
                                          Stroke([(75 + 66 * math.cos(q), 70 + 30 * math.sin(q)) for q in np.linspace(0, math.pi, 30)], 1.6, 0.2, 4)]), 0, 0, INK, 0.9)
    put_text_local(rgb, _T('藥', F_CJK, 34, None, 0, False, 0, 0), 75, 52, INK, align='center')
    return rgb, a

def draw_house_r(img, cam, t):
    y = YR['house']
    dy = y - YA['home']
    c2 = cam_off(cam, dy)
    t0 = b0('r_wife')
    ink_strokes(img, c2, _house(), 1.0, op=0.92)
    # the pharmacy, along the road to the right
    px, py = 880, y + 760
    k = ease(prog(t, t0 + 0.6, t0 + 1.4))
    sign = [Stroke([(px - 60, py - 60), (px + 60, py - 60), (px + 60, py + 90), (px - 60, py + 90), (px - 60, py - 60)], 2.2, 0.4, 5)]
    ink_strokes(img, cam, sign, k, op=0.85)
    S_ = _T('藥', F_CJK, 90, None, 0, False, 0, 0)
    text_at(img, cam, S_, px, py + 50, INK, k, align='center')
    # two sets of steps leaving the house: hers, and a baby carried (only hers are visible)
    fp = footprint()
    n = int(14 * eout(prog(t, t0 + 0.2, b1('r_wife') - 0.5)))
    for i in range(n):
        u = i / 13
        x = lerp(435, 760, u) + (-10 if i % 2 else 10)
        yy = lerp(y + 520, py + 60, u)
        ang = math.atan2(py + 60 - (y + 520), 760 - 435) + math.pi / 2
        blit(img, fp, cam, x, yy, INK, 0.8, rot=ang, anchor=(14, 22), scale=0.8)
    ku = spring(max(0, t - (b1('r_wife') - 1.0)) / 0.7, 1.1, 5)
    if ku > 0:
        rgb, a = tin()
        blit_rgba(img, rgb, a, cam, px - 170, py + 40, 1.0, 1.2, math.radians(-6), (75, 55), shadow=0.6, lift=0.5)
    # behind the house: far away, the second cloud rises
    tr = b0('r_saved') - 0.3
    pc = prog(t, tr, b1('r_saved'))
    if pc > 0:
        a, w, h = big_cloud()
        e = eout(pc)
        blit(img, a, cam, 300, y - 120, INK, 0.35 * min(1, pc * 3), scale=(0.32 * lerp(0.5, 1, e), 0.32 * lerp(0.15, 1, e)), anchor=(w / 2, h - 20))
        ink_strokes(img, cam, [Stroke([(40, y - 120), (1040, y - 120)], 1.4, 0.6, 9, taper=False)], 1.0, op=0.4)

# crew lists
LIST_W, LIST_H = 620, 820
@functools.lru_cache(None)
def crew_list(title, sub, hi_row=None, hi_name=None, seed=0):
    rgb, a = card_paper(LIST_W, LIST_H, 300 + seed, base=C(236, 230, 214), edge=0.12)
    put_text_local(rgb, _T(title, F_SERIF, 44, 700, 6), LIST_W / 2, 80, INK, align='center')
    put_text_local(rgb, _T(sub, F_MONO, 24, None, 2, True, seed), LIST_W / 2, 122, INK, align='center')
    put_alpha(rgb, local_strokes(a.shape, [Stroke([(40, 146), (LIST_W - 40, 146)], 1.6, 0.3, 3)]), 0, 0, INK, 0.7)
    r = np.random.default_rng(seed)
    for row in range(18):
        y = 186 + row * 34
        if hi_row is not None and row == hi_row:
            put_text_local(rgb, _T(hi_name, F_MONO, 26, None, 0, True, 77 + seed), 50, y, INK)
            continue
        x = 50
        blk = np.zeros(a.shape, np.float32)
        for part in (r.uniform(80, 160), r.uniform(60, 120)):
            for k in range(int(part // 13)):
                hh = r.uniform(9, 15)
                cv2.rectangle(blk, (int((x + k * 13) * S), int((y - hh) * S)), (int((x + k * 13 + 8) * S), int(y * S)), 0.6 + 0.3 * r.random(), -1)
            x += part + 28
        put_text_local(rgb, _T('Fireman', F_MONO, 24, None, 0, True, 90 + row + seed), 420, y, INK, 0.7)
        put_alpha(rgb, cv2.GaussianBlur(blk, (0, 0), 1.0), 0, 0, INK, 0.6)
    return rgb, a

@functools.lru_cache(None)
def magnifier():
    R = Raster(-120, -120, 360, 360)
    ring = Raster(-120, -120, 360, 360).ell((0, 0), (92, 92))
    inner = Raster(-120, -120, 360, 360).ell((0, 0), (78, 78)).alpha()
    a = np.clip(ring.alpha() - inner, 0, 1)
    h_ = Raster(-120, -120, 360, 360).line([(62, 62), (190, 190)], 22).alpha()
    return np.clip(a + h_, 0, 1), inner

def draw_lists(img, cam, t):
    y = YR['lists']
    t0, t1 = b0('r_never'), b1('r_lists')
    sheets = [('RMS TITANIC', '1912 · CREW', 0, -260, -6), ('EMPRESS OF IRELAND', '1914 · CREW', 1, 0, 3), ('RMS LUSITANIA', '1915 · CREW', 2, 260, -2)]
    for title, sub, k, dy, rot in sheets:
        u = spring(max(0, t - (t0 - 0.4 + 0.25 * k)) / 0.8, 1.0, 5)
        if u <= 0: continue
        rgb, a = crew_list(title, sub, seed=k)
        blit_rgba(img, rgb, a, cam, lerp(-500, 380 + 120 * k, u), y + dy, 1.0, 0.95, math.radians(rot), (LIST_W / 2, LIST_H / 2), shadow=0.6, lift=0.5)
    # the magnifier sweeps the last sheet, row by row
    pm = prog(t, b0('r_lists'), b1('r_lists') - 0.4)
    if pm > 0:
        mg, inner = magnifier()
        mx = 620 + 60 * math.sin(pm * 9)
        my = y - 80 + 520 * pm
        sx, sy = cam.pt(mx, my)
        # lens: lighten + slight magnify feel
        lens = warp(inner, layer_M(cam, mx, my, 1.0, 0, (120, 120)))
        paint(img, lens, C(250, 246, 236), 0.25, mode='screen')
        blit(img, mg, cam, mx, my, INK, 0.92, anchor=(120, 120))
    # the verdict next to it: TOWER — none
    kv = prog(t, b1('r_lists') - 0.6, b1('r_lists'))
    if kv > 0:
        V_ = _T('TOWER — 0', F_MONOB, 50, None, 2, True, 5)
        text_at(img, cam, V_, 540, y + 600, INK, 0.95, n=int(9 * kv), align='center')
        ink_strokes(img, cam, [Stroke([(330, y + 620), (750, y + 600)], 3.0, 1.0, 7)], eout(prog(t, b1('r_lists'), b1('r_lists') + 0.5)), op=0.85)

def draw_clark(img, cam, t):
    y = YR['clark']
    t0, t1 = b0('r_clark'), b1('r_clark')
    for k, (title, sub, nm, row, x, rot, tk) in enumerate((('RMS TITANIC', '1912 · CREW', 'CLARK, William', 7, 300, -5, t0 + 0.3),
                                                            ('EMPRESS OF IRELAND', '1914 · CREW', 'CLARKE, William', 11, 780, 4, t0 + 1.3))):
        u = spring(max(0, t - (b0('r_legend') - 0.3 + 0.3 * k)) / 0.8, 1.0, 5)
        if u <= 0: continue
        rgb, a = crew_list(title, sub, row, nm, seed=10 + k)
        rgb = rgb.copy()
        yy = 186 + row * 34
        ring = Stroke([(30 + 290 * (0.5 + 0.5 * math.cos(q)) + 0 * q, yy - 8 + 24 * math.sin(q)) for q in np.linspace(-2.6, 2 * math.pi - 2.2, 50)], 3.0, 1.4, 11 + k)
        put_alpha(rgb, local_strokes(a.shape, [ring], eout(prog(t, tk + 0.6, tk + 1.3))), 0, 0, C(84, 84, 92), 0.9)
        blit_rgba(img, rgb, a, cam, lerp(-400 if k == 0 else 1500, x, u), y + (k * 2 - 1) * 170, 1.0, 0.82, math.radians(rot), (LIST_W / 2, LIST_H / 2), shadow=0.6, lift=0.5)
        draw_buoy(img, cam, t, tk + 1.4, x + 200, y + (k * 2 - 1) * 170 + 330, 1.0)
    N_ = _T('William Clark · θερμαστής', F_SERIF_I, 46, 500)
    text_at(img, cam, N_, 540, y - 560, INK, ease(prog(t, t0 + 0.2, t0 + 0.8)), align='center')

def draw_three(img, cam, t):
    y = YR['three']
    t0 = b0('r_two')
    specs = [(4, 1912, 90), (2, 1914, 400), (4, 1915, 710)]
    for i, (nf, yr, x) in enumerate(specs):
        k = ease(prog(t, t0 - 0.6 + 0.15 * i, t0 - 0.1 + 0.15 * i))
        if k <= 0: continue
        draw_ship(img, cam, nf, 280, x, y, op=k, seed=3 + 2 * i)
        T_ = _T(str(yr), F_MONO, 40, None, 1, True, yr)
        text_at(img, cam, T_, x + 140, y + 90, INK, k, align='center')
    kx = eout(prog(t, t0 + 0.6, t0 + 1.2))
    ink_strokes(img, cam, [Stroke([(700, y - 150), (1010, y + 120)], 9.0, 1.6, 21), Stroke([(1010, y - 150), (700, y + 120)], 8.0, 1.6, 22)], kx, op=0.9)
    N_ = _T('2 — όχι 3', F_SERIF, 110, 600)
    text_at(img, cam, N_, 540, y - 330, INK, ease(prog(t, t0, t0 + 0.6)), align='center')

def draw_shore(img, cam, t):
    """Clark: maybe he'll go to sea again. footsteps, this time *towards* the water."""
    y = YR['shore']
    t0, t1 = b0('r_sea'), b1('r_sea')
    wl = y + 560
    U, V = cam.uv()
    sea = cv2.GaussianBlur(((V > wl) & (V < y + 1000)).astype(np.float32), (0, 0), 2)
    paint(img, sea, SEA, 0.45, mode='multiply')
    surf = [Stroke([(x, wl + 8 * math.sin(x / 60 + t * 1.5)) for x in range(-20, W + 40, 10)], 2.4, 0.4, 3, taper=False),
            Stroke([(x, wl + 34 + 6 * math.sin(x / 50 + t * 1.2 + 1)) for x in range(-20, W + 40, 10)], 1.6, 0.4, 4, taper=False)]
    ink_strokes(img, cam, surf, 1.0, op=0.7)
    fp = footprint()
    n = int(22 * eout(prog(t, t0 - 0.2, t1 + 0.2)))
    for i in range(n):
        yy = y - 420 + i * 44
        x = 560 + 14 * math.sin(i * 0.35) + (-14 if i % 2 else 14)
        if yy > wl: break
        blit(img, fp, cam, x, yy, INK, 0.85, rot=math.pi, anchor=(14, 22))

def story_r_cam(t):
    Y = YR
    keys = [(b0('r_wife') - 0.6, 560, Y['house'] + 300, 1.0, 0.0),
            (b1('r_wife') - 0.5, 600, Y['house'] + 360, 1.04, 0.0),
            (b1('r_saved') + 0.2, 540, Y['house'] + 200, 1.0, 0.0),
            (b0('r_never') - 0.3, 540, Y['house'] + 220, 1.0, 0.0),
            (b0('r_never') + 0.4, 540, Y['lists'], 1.0, 0.0),
            (b1('r_lists') + 0.2, 540, Y['lists'] + 80, 1.05, 0.0),
            (b0('r_legend') - 0.25, 540, Y['lists'] + 90, 1.05, 0.0),
            (b0('r_legend') + 0.45, 540, Y['clark'], 1.0, 0.0),
            (b1('r_clark') + 0.2, 540, Y['clark'] + 40, 1.04, 0.0),
            (b0('r_two') - 0.45, 540, Y['clark'] + 50, 1.04, 0.0),
            (b0('r_two') + 0.15, 540, Y['three'] - 60, 1.0, 0.0),
            (b0('r_sea') - 0.4, 540, Y['three'] - 40, 1.02, 0.0),
            (b0('r_sea') + 0.3, 540, Y['shore'] + 60, 1.0, 0.0),
            (b1('r_sea') + 0.4, 540, Y['shore'] + 140, 1.04, 0.0)]
    return cam_path(t, keys)

def story_r(t):
    cam = story_r_cam(t)
    img = paper_r().render(cam)
    vis = lambda yc: abs(cam.cy - yc) < 2100 / cam.z
    for key, fn in (('house', draw_house_r), ('lists', draw_lists), ('clark', draw_clark), ('three', draw_three), ('shore', draw_shore)):
        if vis(YR[key]): fn(img, cam, t)
    return img, cam

def desk_end(t):
    cam = Cam(W / 2, H / 2, 1.0)
    img = reveal_cards(b1('r_true') + 3, cam, hdr=False)
    t0 = b0('r_cta')
    k = eout(prog(t, t0 - 0.2, t0 + 0.5))
    T2 = _T('Εσύ το βρήκες;', F_SERIF, 70, 500)
    text_at(img, cam, T2, W / 2, 190 + 16 * (1 - k), C(240, 232, 218), k, align='center')
    k2 = eout(prog(t, t0 + 0.9, t0 + 1.5))
    T3 = _T('Γράψ’ το στα σχόλια', F_SERIF_I, 44, 500)
    text_at(img, cam, T3, W / 2, 1790, C(236, 226, 210), 0.9 * k2, align='center')
    T1 = _T('ΜΙΑ ΑΛΗΘΕΙΑ  ·  ΕΝΑ ΨΕΜΑ', F_SERIF, 40, 500, 5)
    text_at(img, cam, T1, W / 2, 112, C(236, 226, 210), 0.85, align='center')
    return img
