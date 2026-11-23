"""Test frames for «Μία αλήθεια, ένα ψέμα»: everything drawn in code, no AI imagery.
frame 1 = the two cards (question), frame 2 = the reveal (red stamp on the truth, ink X on the lie)."""
import sys, os, math, textwrap
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'engine'))
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
import eng
from eng import W, H, fbm, Film

FS = '/usr/share/fonts/truetype/freefont/'
SERIF, SERIFB, MONO = FS + 'FreeSerif.ttf', FS + 'FreeSerifBold.ttf', FS + 'FreeMono.ttf'
INKC = np.array([34, 36, 42], np.float32)       # BGR ink
REDI = np.array([40, 44, 176], np.float32)      # BGR stamp red
PAPERC = np.array([196, 214, 226], np.float32)  # warm aged paper (BGR)

# ------------------------------------------------------------ helpers
def text_alpha(txt, path, size, track=0):
    f = ImageFont.truetype(path, size)
    w = int(sum(f.getlength(c) + track for c in txt) + size)
    im = Image.new('L', (w, int(size * 1.7)), 0); d = ImageDraw.Draw(im)
    x = size / 2
    for c in txt:
        d.text((x, size * 0.25), c, font=f, fill=255); x += f.getlength(c) + track
    a = np.asarray(im).astype(np.float32) / 255
    ys, xs = np.where(a > 0.02)
    return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

def typed(a, seed):
    h, w = a.shape
    p = 0.70 + 0.30 * fbm(h, w, 9, seed, 2)
    starve = (fbm(h, w, 3, seed + 5, 1) > 0.2).astype(np.float32) * 0.22 + 0.78
    return np.clip(cv2.GaussianBlur(a, (0, 0), 0.8) * p * starve * 1.12, 0, 1)

def put(img, a, x, y, color, op=1.0, mul=False):
    h, w = a.shape
    H_, W_ = img.shape[:2]
    x0, y0 = max(0, x), max(0, y); x1, y1 = min(W_, x + w), min(H_, y + h)
    if x1 <= x0 or y1 <= y0: return img
    m = a[y0 - y:y1 - y, x0 - x:x1 - x, None] * op
    reg = img[y0:y1, x0:x1]
    tgt = reg * (color / 255.0) if mul else color
    img[y0:y1, x0:x1] = reg * (1 - m) + tgt * m
    return img

def wobble(mask, amp, scale, seed):
    """hand-drawn wobble: displace a mask with smooth noise."""
    h, w = mask.shape
    dx = (fbm(h, w, scale, seed, 2) - 0.5) * 2 * amp
    dy = (fbm(h, w, scale, seed + 9, 2) - 0.5) * 2 * amp
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    return cv2.remap(mask, x + dx, y + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

def hatch(h, w, ang, spacing, width, seed):
    """parallel pen lines (slightly irregular) as an alpha field."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    c, s = math.cos(ang), math.sin(ang)
    u = x * c + y * s + (fbm(h, w, 60, seed, 2) - 0.5) * spacing * 0.8
    d = np.abs(((u / spacing) % 1.0) - 0.5) * spacing
    wv = width * (0.7 + 0.6 * fbm(h, w, 25, seed + 3, 2))
    return np.clip((wv / 2 - d) / 1.2 + 0.5, 0, 1)

class Pen:
    """supersampled mask canvas with simple primitives."""
    def __init__(self, w, h, ss=3):
        self.ss, self.w, self.h = ss, w, h
        self.m = np.zeros((h * ss, w * ss), np.uint8)
    def P(self, pts):
        return [np.round(np.array(pts, np.float64) * self.ss * 16).astype(np.int32)]
    def poly(self, pts, v=255):
        cv2.fillPoly(self.m, self.P(pts), v, cv2.LINE_AA, shift=4)
    def ell(self, c, ax, ang=0, v=255):
        cv2.ellipse(self.m, (int(c[0] * self.ss), int(c[1] * self.ss)), (int(ax[0] * self.ss), int(ax[1] * self.ss)),
                    ang, 0, 360, v, -1, cv2.LINE_AA)
    def stroke(self, pts, w0, w1=None, v=255):
        """tapered polyline stroke."""
        w1 = w0 if w1 is None else w1
        pts = np.array(pts, np.float64); n = len(pts)
        for i in range(n - 1):
            a, b = pts[i], pts[i + 1]
            wa = w0 + (w1 - w0) * i / (n - 1); wb = w0 + (w1 - w0) * (i + 1) / (n - 1)
            d = b - a; L = np.hypot(*d) + 1e-6; nx, ny = -d[1] / L, d[0] / L
            self.poly([(a[0] + nx * wa / 2, a[1] + ny * wa / 2), (b[0] + nx * wb / 2, b[1] + ny * wb / 2),
                       (b[0] - nx * wb / 2, b[1] - ny * wb / 2), (a[0] - nx * wa / 2, a[1] - ny * wa / 2)], v)
            cv2.circle(self.m, (int(b[0] * self.ss), int(b[1] * self.ss)), max(1, int(wb / 2 * self.ss)), v, -1, cv2.LINE_AA)
    def line(self, a, b, w, v=255):
        self.stroke([a, b], w, w, v)
    def alpha(self, wob=1.2, seed=1):
        a = cv2.resize(self.m, (self.w, self.h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        return wobble(a, wob, 14, seed) if wob else a

def bez(p0, p1, p2, n=24):
    t = np.linspace(0, 1, n)[:, None]
    return ((1 - t) ** 2 * np.array(p0) + 2 * (1 - t) * t * np.array(p1) + t ** 2 * np.array(p2)).tolist()

def ink_tex(a, seed):
    h, w = a.shape
    return a * (0.80 + 0.20 * fbm(h, w, 7, seed, 2))

# ------------------------------------------------------------ drawings (alpha layers, card-local px)
def draw_pyramid_mammoth(w, h, seed=3):
    """returns (ink alpha, paper-highlight alpha) for a w*h box."""
    ink = Pen(w, h); hi = Pen(w, h); hat = Pen(w, h)
    gy = h - 40
    # --- pyramid, far back-left
    ax, ay, bl, br = 200, 40, 10, 400
    mid = 250
    ink.stroke([(bl, gy - 60), (ax, ay), (br, gy - 60)], 4)                     # outline
    ink.line((ax, ay), (mid, gy - 60), 3)                                         # ridge
    ink.line((bl - 30, gy - 60), (br + 40, gy - 60), 3)                           # horizon under pyramid
    hat.poly([(ax, ay), (mid, gy - 60), (br, gy - 60)])                           # shaded face
    for k in range(1, 9):                                                         # stone courses, lit face
        yy = ay + (gy - 60 - ay) * k / 9
        xl = ax + (bl - ax) * k / 9; xm = ax + (mid - ax) * k / 9
        ink.line((xl + 6, yy), (xm - 4, yy), 1.6)
    # --- ground
    for k in range(14):
        x = 40 + k * 48 + (k * 37) % 19
        ink.line((x, gy + 8 + (k % 3) * 6), (x + 26 + (k % 4) * 6, gy + 8 + (k % 3) * 6), 2.2)
    # --- mammoth, front-right, facing right
    ox, oy, s = 330, 70, 1.15
    def T(p): return (ox + p[0] * s, oy + p[1] * s)
    body = Pen(w, h)
    body.ell(T((170, 150)), (118 * s, 78 * s))
    body.ell(T((235, 100)), (62 * s, 60 * s))
    body.ell(T((290, 108)), (52 * s, 56 * s))
    body.ell(T((282, 72)), (34 * s, 30 * s))                                      # head dome
    for lx in (88, 132, 220, 262):
        body.poly([T((lx - 18, 170)), T((lx + 18, 170)), T((lx + 16, 268)), T((lx - 17, 268))])
        body.ell(T((lx, 266)), (19 * s, 7 * s))
    body.stroke([T(p) for p in bez((322, 120), (352, 200), (330, 250)) + bez((330, 250), (322, 270), (342, 262), 8)],
                30 * s, 10 * s)                                                   # trunk
    body.stroke([T(p) for p in bez((58, 130), (40, 170), (44, 205), 10)], 8 * s, 4 * s)   # tail
    # shaggy fringe under belly
    for k in range(22):
        x0 = 70 + k * 9.5
        body.poly([T((x0, 200)), T((x0 + 10, 200)), T((x0 + 4, 232 + (k * 7) % 14))])
    bm = body.alpha(wob=1.6, seed=seed + 1)
    # tusks: ink outline then paper inside
    tusk = [T(p) for p in bez((300, 168), (350, 238), (402, 168))]
    ink.stroke(tusk, 17 * s, 8 * s)
    hi.stroke(tusk, 10 * s, 3 * s)
    # fur highlights (woodcut): short wavy paper strokes on the upper body
    r = np.random.default_rng(seed)
    for _ in range(70):
        cx = r.uniform(80, 310); cy = r.uniform(60, 190)
        if ((cx - 170) / 112) ** 2 + ((cy - 150) / 72) ** 2 > 1 and ((cx - 260) / 70) ** 2 + ((cy - 100) / 60) ** 2 > 1:
            continue
        L = r.uniform(14, 30)
        hi.stroke([T((cx, cy)), T((cx - 3, cy + L * 0.5)), T((cx + 1, cy + L))], 2.6, 1.0)
    hi.ell(T((300, 92)), (4.5 * s, 4 * s))                                        # eye
    a_ink = np.maximum(ink.alpha(seed=seed), bm * 0.96)
    hm = hat.alpha(wob=0.8, seed=seed + 2) * hatch(h, w, math.radians(62), 9, 2.4, seed)
    a_ink = np.maximum(a_ink, hm * (1 - bm))
    # mammoth shadow on ground
    sh = Pen(w, h); sh.ell((ox + 205 * s, oy + 270 * s), (175 * s, 12 * s))
    a_ink = np.maximum(a_ink, sh.alpha(seed=seed + 4) * hatch(h, w, math.radians(-20), 7, 2.0, seed + 5) * 0.9)
    return ink_tex(a_ink, seed), hi.alpha(wob=0.6, seed=seed + 6)

def draw_colossus(w, h, seed=7):
    ink = Pen(w, h); hi = Pen(w, h); hat = Pen(w, h); fig = Pen(w, h)
    base = h - 62
    s = (base - 12) / 430.0                      # figure+pedestal+flame = 430 units tall
    cx = w * 0.40
    def T(x, y): return (cx + x * s, base - y * s)
    # sun, upper right, hatched disc + short rays
    sx, sy, sr = w * 0.80, h * 0.24, 56 * s / 0.8
    hat.ell((sx, sy), (sr, sr))
    for k in range(16):
        a = k * math.pi / 8
        ink.line((sx + (sr + 12) * math.cos(a), sy + (sr + 12) * math.sin(a)), (sx + (sr + 28) * math.cos(a), sy + (sr + 28) * math.sin(a)), 2.4)
    # sea
    for row in range(4):
        yy = base + 10 + row * 12
        for k in range(11):
            x0 = 8 + k * 60 + (row * 23) % 40
            ink.stroke(bez((x0, yy), (x0 + 15, yy - 5), (x0 + 32, yy), 8), 2.0)
    # small ancient ship, far right, for scale
    shx, shy = w * 0.80, base + 2
    fig.poly([(shx - 46, shy - 14), (shx + 50, shy - 14), (shx + 36, shy), (shx - 34, shy)])
    fig.stroke(bez((shx + 50, shy - 14), (shx + 60, shy - 22), (shx + 56, shy - 30), 6), 4, 2)
    fig.line((shx + 2, shy - 14), (shx + 2, shy - 78), 3)
    fig.poly([(shx - 26, shy - 74), (shx + 30, shy - 74), (shx + 26, shy - 30), (shx - 22, shy - 30)])
    # pedestal
    fig.poly([T(-78, 0), T(78, 0), T(64, 58), T(-64, 58)])
    fig.poly([T(-56, 58), T(56, 58), T(50, 80), T(-50, 80)])
    # legs: left straight, right relaxed (contrapposto)
    fig.stroke([T(-20, 80), T(-19, 130), T(-16, 182)], 17 * s, 26 * s)
    fig.stroke([T(20, 80), T(28, 128), T(16, 182)], 16 * s, 26 * s)
    # torso: hips -> waist -> broad shoulders
    fig.poly([T(-34, 180), T(34, 180), T(28, 212), T(46, 262), T(40, 272), T(-40, 272), T(-46, 262), T(-28, 212)])
    fig.stroke([T(0, 270), T(0, 286)], 14 * s)                        # neck
    fig.ell(T(0, 302), (17 * s, 21 * s))                              # head
    for k in range(7):                                                # crown of rays
        a = math.radians(-160 + k * 23.3)
        fig.stroke([(T(0, 302)[0] + 14 * s * math.cos(a), T(0, 302)[1] + 14 * s * math.sin(a)),
                    (T(0, 302)[0] + 42 * s * math.cos(a), T(0, 302)[1] + 42 * s * math.sin(a))], 6 * s, 1.5)
    # raised right arm (viewer's left) with torch
    fig.stroke([T(-40, 266), T(-62, 318), T(-66, 366)], 15 * s, 12 * s)
    fig.poly([T(-80, 366), T(-52, 366), T(-58, 392), T(-74, 392)])
    fl = [(-66 + 13 * math.sin(a) * (1 - a / math.pi) ** 0.5 * (1 if a < math.pi else 1), 0) for a in []]
    fig.poly([T(-66 + x, 392 + y) for x, y in [(-13, 6), (-15, 16), (-10, 26), (-12, 34), (-4, 30), (-2, 44), (4, 32), (10, 38), (9, 26), (14, 16), (12, 6)]])   # flame
    # lowered left arm and cloak (chlamys) hanging from it
    fig.stroke([T(40, 266), T(56, 222), T(58, 186)], 14 * s, 12 * s)
    fig.poly([T(36, 268), T(52, 262), T(78, 184), T(70, 112), T(56, 126), T(52, 184)])
    fm = fig.alpha(wob=1.2, seed=seed + 1)
    # highlights: a few contour lines, like a woodcut
    hi.line(T(-56, 58), T(56, 58), 2.2)
    hi.line(T(-50, 80), T(50, 80), 1.8)
    hi.stroke([T(x, y) for x, y in bez((-26, 248), (0, 238), (26, 248), 10)], 2.2)   # chest
    hi.stroke([T(x, y) for x, y in bez((-18, 214), (0, 206), (18, 214), 10)], 2.0)   # abdomen
    hi.stroke([T(-18, 130), T(-14, 136)], 2.0); hi.stroke([T(24, 128), T(28, 134)], 2.0)   # knees
    for k in range(3):                                                                # cloak folds
        hi.stroke([T(58 + k * 5, 240 - k * 12), T(64 + k * 4, 150)], 1.8, 1.0)
    hm = hat.alpha(seed=seed + 2) * hatch(h, w, math.radians(35), 8, 2.2, seed)
    a_ink = np.maximum(np.maximum(ink.alpha(seed=seed), fm * 0.96), hm * (1 - fm))
    return ink_tex(a_ink, seed), hi.alpha(wob=0.5, seed=seed + 3)


def outline_of(mask, wpx):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * wpx + 1, 2 * wpx + 1))
    return np.clip(cv2.dilate(mask, k) - mask, 0, 1)

def mushroom(pen_w, pen_h, cx, ground, ht, seed):
    """mushroom cloud: returns (ink, highlight) alphas."""
    r = np.random.default_rng(seed)
    m = Pen(pen_w, pen_h); bill = Pen(pen_w, pen_h)
    capy, capr = ground - ht * 0.78, ht * 0.27
    # stem
    m.poly([(cx - capr * 0.22, capy + capr * 0.3), (cx + capr * 0.22, capy + capr * 0.3),
            (cx + capr * 0.34, ground - ht * 0.12), (cx - capr * 0.34, ground - ht * 0.12)])
    # collar ring around the stem
    m.ell((cx, ground - ht * 0.45), (capr * 0.55, capr * 0.13))
    # base surge
    for k in range(7):
        bx = cx + (k - 3) * capr * 0.36
        m.ell((bx, ground - capr * 0.18), (capr * 0.34, capr * 0.22))
    # cap: cauliflower of billows
    puffs = [(0, -0.35, 0.55), (-0.5, -0.1, 0.48), (0.5, -0.1, 0.48), (-0.85, 0.2, 0.36), (0.85, 0.2, 0.36),
             (-0.3, 0.25, 0.42), (0.3, 0.25, 0.42), (0, 0.05, 0.5), (-0.25, -0.6, 0.35), (0.3, -0.62, 0.33)]
    for dx, dy, rr in puffs:
        c = (cx + dx * capr * 1.25, capy + dy * capr * 0.8); R = rr * capr * (0.95 + 0.1 * r.random())
        m.ell(c, (R, R * 0.92))
        bill.stroke([(c[0] + R * math.cos(a), c[1] + R * 0.92 * math.sin(a)) for a in np.linspace(math.pi * 1.05, math.pi * 1.85, 14)], 2.4, 1.2)
    for k in range(3):                                         # stem billows
        yy = ground - ht * (0.2 + 0.08 * k)
        bill.stroke(bez((cx - capr * 0.2, yy), (cx, yy - 10), (cx + capr * 0.2, yy), 10), 2.0)
    ma = m.alpha(wob=1.8, seed=seed)
    ba = bill.alpha(wob=1.0, seed=seed + 1) * ma
    ol = outline_of(ma, 2)
    y, x = np.mgrid[0:pen_h, 0:pen_w].astype(np.float32)
    side = np.clip((x - cx) / (capr * 0.9) + 0.25, 0, 1)        # shade the right side, light from the left
    under = np.clip((y - capy) / (capr * 0.6), 0, 1)
    sh = ma * np.clip(0.35 + 0.65 * np.maximum(side, under * 0.8), 0, 1)
    h1 = hatch(pen_h, pen_w, math.radians(55), 9, 2.0, seed) * sh
    h2 = hatch(pen_h, pen_w, math.radians(-35), 10, 1.8, seed + 2) * ma * np.clip(side * 1.4 - 0.4, 0, 1)
    ink = np.maximum.reduce([ol, ba, h1 * 0.9, h2 * 0.85])
    return ink

def draw_two_bombs(w, h, seed=3):
    ink = np.zeros((h, w), np.float32)
    gy = h - 64
    for i, (cx, ht) in enumerate([(w * 0.27, h * 0.76), (w * 0.73, h * 0.70)]):
        ink = np.maximum(ink, mushroom(w, h, cx, gy, ht, seed + i * 10))
    g = Pen(w, h)
    g.line((10, gy), (w - 10, gy), 3)
    r = np.random.default_rng(seed)
    for cx in (w * 0.27, w * 0.73):                            # small city silhouettes on the ground
        x = cx - 150
        while x < cx + 150:
            bw = r.uniform(14, 30); bh = r.uniform(8, 26)
            g.poly([(x, gy), (x + bw, gy), (x + bw, gy - bh), (x, gy - bh)]); x += bw + r.uniform(2, 8)
    # dashed arc: the trip home between the two cities
    arc = bez((w * 0.36, gy - 20), (w * 0.5, gy - 120), (w * 0.64, gy - 20), 40)
    for k in range(0, 38, 3):
        g.stroke(arc[k:k + 2], 2.6)
    ink = np.maximum(ink, g.alpha(seed=seed + 5))
    hi = np.zeros_like(ink)
    for cx, txt in ((w * 0.27, '6.8.1945'), (w * 0.73, '9.8.1945')):
        a = typed(text_alpha(txt, MONO, 30), seed + int(cx))
        y0 = gy + 14; x0 = int(cx - a.shape[1] / 2)
        ink[y0:y0 + a.shape[0], x0:x0 + a.shape[1]] = np.maximum(ink[y0:y0 + a.shape[0], x0:x0 + a.shape[1]], a)
    return ink_tex(ink, seed), hi

def ship_pen(p, hi, x0, wl, L, funnels, ang, sink):
    """liner in profile, bow to the right, tilted by ang (deg, + = bow down), sunk by `sink` px.
    p = ink silhouette pen, hi = paper highlight pen (portholes, hull band)."""
    def R(x, y):
        a = math.radians(ang); cx, cy = x0 + L * 0.5, wl
        dx, dy = x - cx, y - cy
        return (cx + dx * math.cos(a) - dy * math.sin(a), cy + dx * math.sin(a) + dy * math.cos(a) + sink)
    H_ = L * 0.10
    hull = [(x0, wl - H_), (x0 + L * 0.97, wl - H_ * 1.08), (x0 + L, wl - H_ * 1.12), (x0 + L * 0.95, wl + H_ * 0.5),
            (x0 + L * 0.07, wl + H_ * 0.5), (x0 - L * 0.03, wl - H_ * 0.25)]
    p.poly([R(*q) for q in hull])
    p.poly([R(*q) for q in [(x0 + L * 0.16, wl - H_), (x0 + L * 0.80, wl - H_), (x0 + L * 0.78, wl - H_ * 1.7), (x0 + L * 0.18, wl - H_ * 1.7)]])
    p.poly([R(*q) for q in [(x0 + L * 0.24, wl - H_ * 1.7), (x0 + L * 0.72, wl - H_ * 1.7), (x0 + L * 0.71, wl - H_ * 2.2), (x0 + L * 0.25, wl - H_ * 2.2)]])
    for k in range(funnels):
        fx = x0 + L * (0.30 + 0.38 * k / max(1, funnels - 1)) if funnels > 1 else x0 + L * 0.48
        fw = L * 0.05
        p.poly([R(fx - fw / 2, wl - H_ * 2.1), R(fx + fw / 2, wl - H_ * 2.1), R(fx + fw / 2 - L * 0.012, wl - H_ * 3.7), R(fx - fw / 2 - L * 0.012, wl - H_ * 3.7)])
        hi.line(R(fx - fw / 2 - L * 0.004, wl - H_ * 3.25), R(fx + fw / 2 - L * 0.008, wl - H_ * 3.25), max(1.5, L * 0.006))
    p.line(R(x0 + L * 0.08, wl - H_ * 1.05), R(x0 + L * 0.065, wl - H_ * 3.9), max(1.6, L * 0.006))
    p.line(R(x0 + L * 0.90, wl - H_ * 1.1), R(x0 + L * 0.885, wl - H_ * 3.9), max(1.6, L * 0.006))
    hi.line(R(x0 + L * 0.02, wl - H_ * 0.55), R(x0 + L * 0.97, wl - H_ * 0.62), max(1.4, L * 0.006))  # hull band
    for k in range(int(L / 22)):                                                                       # portholes
        px = x0 + L * 0.08 + k * 22
        if px > x0 + L * 0.9: break
        hi.ell(R(px, wl - H_ * 1.35), (max(1.2, L * 0.005), max(1.2, L * 0.005)))
    return R

def draw_three_wrecks(w, h, seed=7):
    ships = Pen(w, h); hi = Pen(w, h)
    hz = h * 0.42                                                   # horizon for the two far ships
    wl = h * 0.80                                                   # waterline of the near ship
    specs = [(w * 0.03, hz, w * 0.22, 4, 14, 6), (w * 0.75, hz, w * 0.21, 2, -10, 7)]
    for x0, y0, L, fn, ang, sink in specs:
        ship_pen(ships, hi, x0, y0, L, fn, ang, sink)
    far = ships.alpha(wob=0.6, seed=seed)
    y = np.arange(h, dtype=np.float32)[:, None]
    far = far * (y < hz + 4)
    near = Pen(w, h)
    ship_pen(near, hi, w * 0.25, wl, w * 0.50, 4, 10, 12)
    na = near.alpha(wob=1.0, seed=seed + 1) * (y < wl + 5)
    hia = hi.alpha(wob=0.4, seed=seed + 2) * (y < wl + 3)
    sea = Pen(w, h)
    sea.line((4, hz + 5), (w - 4, hz + 5), 2.2)
    for row in range(3):                                            # far sea: fine lines
        yy = hz + 16 + row * 12
        for k in range(16):
            x0 = 4 + k * (w / 15.5) + (row * 23) % 40
            sea.stroke(bez((x0, yy), (x0 + 10, yy - 3), (x0 + 22, yy), 6), 1.5)
    for row in range(5):                                            # near sea: bigger waves
        yy = wl + 8 + row * 16
        for k in range(10):
            x0 = 4 + k * (w / 9.5) + (row * 37) % 60
            sea.stroke(bez((x0, yy), (x0 + 18, yy - 7), (x0 + 40, yy), 8), 2.6 - row * 0.2)
    ink = np.maximum.reduce([far * 0.9, na * 0.96, sea.alpha(seed=seed + 3)])
    labels = [(w * 0.14, hz + 34, '1912'), (w * 0.86, hz + 34, '1914'), (w * 0.50, h - 30, '1915')]
    for i, (cx, cy, yr) in enumerate(labels):
        a = typed(text_alpha(yr, MONO, 30), seed + i)
        x0 = int(cx - a.shape[1] / 2); y0 = int(cy)
        reg = ink[y0:y0 + a.shape[0], x0:x0 + a.shape[1]]
        # clear the waves under the label, then type it
        ink[y0 - 4:y0 + a.shape[0] + 4, x0 - 8:x0 + a.shape[1] + 8] *= 0.0
        ink[y0:y0 + a.shape[0], x0:x0 + a.shape[1]] = a
    return ink_tex(ink, seed), hia

# ------------------------------------------------------------ cards
CW, CH = 900, 690

def card_paper(seed):
    img = np.ones((CH, CW, 3), np.float32) * PAPERC
    img *= (0.90 + 0.10 * fbm(CH, CW, 220, seed, 4))[..., None]
    img *= (0.95 + 0.05 * fbm(CH, CW, 14, seed + 1, 3))[..., None]
    y, x = np.mgrid[0:CH, 0:CW].astype(np.float32)
    edge = np.minimum.reduce([x, y, CW - 1 - x, CH - 1 - y])
    img *= (1 - 0.16 * np.exp(-edge / 26))[..., None]                       # darker aged edges
    return img

def make_card(letter, claim, drawing, seed):
    img = card_paper(seed)
    # letter in a hand-inked circle
    ring = Pen(CW, CH); ring.stroke([(70 + 52 * math.cos(a), 92 + 52 * math.sin(a)) for a in np.linspace(0.2, 2 * math.pi + 0.5, 60)], 4, 2.5)
    put(img, ink_tex(ring.alpha(seed=seed), seed), 0, 0, INKC)
    la = text_alpha(letter, SERIFB, 74)
    put(img, la, 70 - la.shape[1] // 2, 92 - la.shape[0] // 2 + 2, INKC)
    # typed claim
    y = 44
    for ln in textwrap.wrap(claim, 29):
        a = typed(text_alpha(ln, MONO, 38), seed + y)
        put(img, a, 150, y, INKC, 0.95); y += 52
    # thin rule
    rule = Pen(CW, CH); rule.line((40, y + 22), (CW - 40, y + 22), 1.8)
    put(img, rule.alpha(seed=seed + 2), 0, 0, INKC, 0.7)
    # drawing in the lower part
    dh = CH - (y + 40) - 20
    ia, ha = drawing(CW - 60, dh, seed)
    put(img, ia, 30, y + 40, INKC, 0.94)
    put(img, ha, 30, y + 40, img[y + 40:y + 40 + dh, 30:CW - 30].mean((0, 1)), 0.95)
    return img

def stamp_alpha(word, seed, size=116):
    t = text_alpha(word, SERIFB, size, track=10)
    pad = 34
    h, w = t.shape[0] + pad * 2, t.shape[1] + pad * 2
    a = np.zeros((h, w), np.float32)
    a[pad:pad + t.shape[0], pad:pad + t.shape[1]] = t
    cv2.rectangle(a, (6, 6), (w - 7, h - 7), 1.0, 7, cv2.LINE_AA)
    cv2.rectangle(a, (18, 18), (w - 19, h - 19), 1.0, 2, cv2.LINE_AA)
    a = wobble(a, 1.5, 20, seed)
    ink = 0.55 + 0.45 * fbm(h, w, 30, seed, 3)                              # uneven pressure
    holes = (fbm(h, w, 4, seed + 1, 2) > 0.30).astype(np.float32)          # starved specks
    return np.clip(a * ink * (0.55 + 0.45 * holes) * 1.25, 0, 1)

def rotate_rgba(img, alpha, deg):
    h, w = alpha.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * s + w * c), int(h * c + w * s)
    M[0, 2] += nw / 2 - w / 2; M[1, 2] += nh / 2 - h / 2
    ri = cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LINEAR) if img is not None else None
    ra = cv2.warpAffine(alpha, M, (nw, nh), flags=cv2.INTER_LINEAR)
    return ri, ra

def place_card(bg, card, cx, cy, deg, lift=1.0):
    a0 = np.ones(card.shape[:2], np.float32)
    ri, ra = rotate_rgba(card, a0, deg)
    h, w = ra.shape
    x, y = int(cx - w / 2), int(cy - h / 2)
    sh = cv2.GaussianBlur(ra, (0, 0), 18 * lift)
    put(bg, sh, x + int(10 * lift), y + int(18 * lift), np.array([8, 8, 10], np.float32), 0.75)
    H_, W_ = bg.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W_, x + w), min(H_, y + h)
    if x1 > x0 and y1 > y0:
        m = ra[y0 - y:y1 - y, x0 - x:x1 - x, None]
        bg[y0:y1, x0:x1] = bg[y0:y1, x0:x1] * (1 - m) + ri[y0 - y:y1 - y, x0 - x:x1 - x] * m
    return (x, y, w, h)

def desk():
    v = 0.10 + 0.07 * cv2.resize(fbm(H // 8, W, 30, 9, 4), (W, H)) + 0.05 * fbm(H, W, 90, 10, 4)
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    spot = np.exp(-(((x - W * 0.48) / (W * 0.7)) ** 2 + ((y - H * 0.45) / (H * 0.5)) ** 2))
    v = v * (0.5 + 0.8 * spot)
    return np.clip(v[..., None] * eng.PAPER * 1.0, 0, 255)

def header(img, txt, y, size=64, op=1.0):
    a = text_alpha(txt, SERIF, size, track=2)
    put(img, a, (W - a.shape[1]) // 2, y, np.array([210, 224, 234], np.float32), op)

CLAIM_A = 'Ένας άνθρωπος επέζησε από δύο ατομικές βόμβες: Χιροσίμα και, τρεις μέρες μετά, Ναγκασάκι.'
CLAIM_B = 'Ένας θερμαστής επέζησε από τρία ναυάγια: Τιτανικός, Έμπρες οφ Άιρλαντ, Λουζιτάνια.'

def frame(reveal):
    bg = desk()
    header(bg, 'ΜΙΑ ΑΛΗΘΕΙΑ  ·  ΕΝΑ ΨΕΜΑ', 92, 40, 0.75)
    header(bg, 'Ποιο είναι αληθινό;', 150, 70)
    A = make_card('Α', CLAIM_A, draw_two_bombs, 3)
    B = make_card('Β', CLAIM_B, draw_three_wrecks, 7)
    if reveal:
        # the lie: crossed out with two hand ink strokes, card slightly in shadow
        x = Pen(CW, CH)
        x.stroke(bez((70, 120), (430, 330), (840, 620), 30), 16, 9)
        x.stroke(bez((830, 90), (480, 360), (90, 640), 30), 15, 8)
        B = put(B * 0.9, ink_tex(x.alpha(wob=2.5, seed=21), 21), 0, 0, INKC, 0.92)
        s = stamp_alpha('ΑΛΗΘΕΙΑ', 31)
        _, sa = rotate_rgba(None, s, 9)
        put(A, sa, CW - sa.shape[1] - 20, 210, REDI, 0.93, mul=True)
    place_card(bg, A, W / 2 - 6, 610, 1.3)
    place_card(bg, B, W / 2 + 6, 1335, -1.0, lift=0.7 if reveal else 1.0)
    out = Film(seed=5).apply(bg, 3, 0.0, strength=0.8, weave=False)
    return np.clip(out, 0, 255).astype(np.uint8)

if __name__ == '__main__':
    od = sys.argv[1] if len(sys.argv) > 1 else '/home/claude/w'
    cv2.imwrite(f'{od}/tl_frame1_question.png', frame(False))
    cv2.imwrite(f'{od}/tl_frame2_reveal.png', frame(True))
    print('ok')
