"""Animated test: card A builds itself (typewriter, pen, clouds rising), slides up, card B arrives,
red stamp slams on the truth, ink X crosses the lie. Everything procedural. -> anim_test.mp4"""
import sys, os, math, subprocess, textwrap
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
import cards05 as T
from cards05 import Pen, bez, hatch, typed, put, text_alpha, ink_tex, INKC, REDI, CW, CH, MONO, SERIFB
from eng import W, H, ease, ease2, lerp, clamp01, Film

FPS = 30
DUR = 11.5
rng = np.random.default_rng(4)

def seg(t, a, b):
    return clamp01((t - a) / (b - a))

# ------------------------------------------------------------ card A, as separate layers
FONT_SZ, LINE_H, TX, TY0 = 38, 52, 150, 44
lines = textwrap.wrap(T.CLAIM_A, 29)

def typed_line(txt, seed):
    """uncropped typed line + x of each char boundary (for the typewriter reveal)."""
    from PIL import Image, ImageDraw, ImageFont
    f = ImageFont.truetype(MONO, FONT_SZ)
    w = int(f.getlength(txt) + 20); h = int(FONT_SZ * 1.5)
    im = Image.new('L', (w, h), 0); d = ImageDraw.Draw(im)
    d.text((4, 2), txt, font=f, fill=255)
    a = np.asarray(im).astype(np.float32) / 255
    xs = [4 + f.getlength(txt[:i]) for i in range(len(txt) + 1)]
    return typed(a, seed), xs

TL = [typed_line(ln, 100 + i) for i, ln in enumerate(lines)]
NCH = sum(len(ln) for ln in lines)
y_rule = TY0 + len(lines) * LINE_H + 22
DX, DY = 30, TY0 + len(lines) * LINE_H + 40
DW, DH = CW - 60, CH - DY - 20

base_A = T.card_paper(3)
ring = Pen(CW, CH); ring.stroke([(70 + 52 * math.cos(a), 92 + 52 * math.sin(a)) for a in np.linspace(0.2, 2 * math.pi + 0.5, 60)], 4, 2.5)
put(base_A, ink_tex(ring.alpha(seed=3), 3), 0, 0, INKC)
la = text_alpha('Α', SERIFB, 74); put(base_A, la, 70 - la.shape[1] // 2, 92 - la.shape[0] // 2 + 2, INKC)
rule = Pen(CW, CH); rule.line((40, y_rule), (CW - 40, y_rule), 1.8); RULE = rule.alpha(seed=5)

gy = DH - 64
CLOUDS = [(DW * 0.27, DH * 0.76), (DW * 0.73, DH * 0.70)]
CL = [ink_tex(T.mushroom(DW, DH, cx, gy, ht, 3 + i * 10), 3 + i) for i, (cx, ht) in enumerate(CLOUDS)]
g = Pen(DW, DH); g.line((10, gy), (DW - 10, gy), 3)
r = np.random.default_rng(3)
for cx, _ in CLOUDS:
    x = cx - 150
    while x < cx + 150:
        bw = r.uniform(14, 30); bh = r.uniform(8, 26)
        g.poly([(x, gy), (x + bw, gy), (x + bw, gy - bh), (x, gy - bh)]); x += bw + r.uniform(2, 8)
GROUND = ink_tex(g.alpha(seed=8), 8)
arc = bez((DW * 0.36, gy - 20), (DW * 0.5, gy - 120), (DW * 0.64, gy - 20), 40)
ap = Pen(DW, DH)
for k in range(0, 38, 3):
    ap.stroke(arc[k:k + 2], 2.6)
ARC = ap.alpha(seed=9)
DATES = [typed_line(d, 300 + i) for i, d in enumerate(('6.8.1945', '9.8.1945'))]

XX = np.arange(DW, dtype=np.float32)[None, :]
XC = np.arange(CW, dtype=np.float32)[None, :]

def wipe_x(a, p, x0, x1, soft=12):
    xx = np.arange(a.shape[1], dtype=np.float32)[None, :]
    edge = lerp(x0 - soft, x1 + soft, p)
    return a * np.clip((edge - xx) / soft, 0, 1)

def grow(a, cx, p):
    """cloud rises from the ground: scale anchored at (cx, gy), fades in."""
    if p <= 0: return None
    e = ease(p)
    sy, sx = lerp(0.12, 1.0, e ** 0.8), lerp(0.45, 1.0, e)
    M = np.float32([[sx, 0, cx * (1 - sx)], [0, sy, gy * (1 - sy)]])
    out = cv2.warpAffine(a, M, (DW, DH), flags=cv2.INTER_LINEAR)
    return out * clamp01(p * 4)

def card_A(t):
    img = base_A.copy()
    # typewriter: 0.6 -> 3.4 s
    n = int(NCH * seg(t, 0.6, 3.4))
    for i, (a, xs) in enumerate(TL):
        k = min(n, len(xs) - 1)
        if k > 0:
            cut = a[:, :int(xs[k]) + 1]
            put(img, cut, TX - 4, TY0 + i * LINE_H - 2, INKC, 0.95)
        n -= len(xs) - 1
        if n <= 0: break
    put(img, wipe_x(RULE, ease(seg(t, 3.4, 3.8)), 40, CW - 40), 0, 0, INKC, 0.7)
    d = np.zeros((DH, DW), np.float32)
    d = np.maximum(d, wipe_x(GROUND, ease(seg(t, 3.7, 4.3)), 0, DW))
    c1 = grow(CL[0], CLOUDS[0][0], seg(t, 4.0, 5.2))
    if c1 is not None: d = np.maximum(d, c1)
    d = np.maximum(d, wipe_x(ARC, seg(t, 5.3, 6.0), DW * 0.36, DW * 0.64, soft=4))
    c2 = grow(CL[1], CLOUDS[1][0], seg(t, 6.0, 7.2))
    if c2 is not None: d = np.maximum(d, c2)
    for i, ((a, xs), (cx, _), t0) in enumerate(zip(DATES, CLOUDS, (4.6, 6.6))):
        k = int((len(xs) - 1) * seg(t, t0, t0 + 0.5))
        if k > 0:
            cut = a[:, :int(xs[k]) + 1]
            dd = np.zeros_like(d); y0 = gy + 10; x0 = int(cx - xs[-1] / 2)
            hh, ww = cut.shape; dd[y0:y0 + hh, x0:x0 + ww] = cut[:DH - y0, :DW - x0]
            d = np.maximum(d, dd)
    put(img, d, DX, DY, INKC, 0.94)
    return img

# ------------------------------------------------------------ card B (finished), stamp, X
card_B = T.make_card('Β', T.CLAIM_B, T.draw_three_wrecks, 7)
st = T.stamp_alpha('ΑΛΗΘΕΙΑ', 31)
_, STAMP = T.rotate_rgba(None, st, 9)
SX, SY = CW - STAMP.shape[1] - 20, 210
xs_ = Pen(CW, CH); xs_.stroke(bez((70, 120), (430, 330), (840, 620), 30), 16, 9); X1 = ink_tex(xs_.alpha(wob=2.5, seed=21), 21)
xs_ = Pen(CW, CH); xs_.stroke(bez((830, 90), (480, 360), (90, 640), 30), 15, 8); X2 = ink_tex(xs_.alpha(wob=2.5, seed=22), 22)
YC, XCg = np.mgrid[0:CH, 0:CW].astype(np.float32)

def diag_wipe(a, p, p0, p1):
    """reveal along the stroke direction p0 -> p1."""
    d = np.array(p1, np.float32) - np.array(p0, np.float32); L = np.hypot(*d); u = d / L
    proj = ((XCg - p0[0]) * u[0] + (YC - p0[1]) * u[1]) / L
    return a * np.clip((p - proj) / 0.03, 0, 1)

def stamp_on(img, t, t0):
    p = seg(t, t0, t0 + 0.14)
    if p <= 0: return img, 0.0
    s = lerp(1.7, 1.0, p ** 2)
    hh, ww = STAMP.shape
    sa = cv2.resize(STAMP, (int(ww * s), int(hh * s)))
    cx, cy = SX + ww / 2, SY + hh / 2
    put(img, sa, int(cx - sa.shape[1] / 2), int(cy - sa.shape[0] / 2), T.REDI, 0.93 * clamp01(p * 1.5), mul=True)
    impact = 1.0 if p >= 1 else 0.0
    return img, impact

# ------------------------------------------------------------ scene
DESK = T.desk()
film = Film(seed=5)

def header(img, op):
    if op <= 0: return
    T.header(img, 'ΜΙΑ ΑΛΗΘΕΙΑ  ·  ΕΝΑ ΨΕΜΑ', 92, 40, 0.75 * op)
    T.header(img, 'Ποιο είναι αληθινό;', 150, 70, op)

def frame(fi):
    t = fi / FPS
    bg = DESK.copy()
    # card A: arrives (0-0.6), sits centered while building, slides up (7.6-8.5)
    pin = ease2(seg(t, 0.0, 0.6)); pup = ease2(seg(t, 7.6, 8.5))
    ay = lerp(lerp(1300, 960, pin), 610, pup)
    arot = lerp(lerp(-4.0, -0.6, pin), 1.3, pup) + 0.25 * math.sin(t * 0.9)
    A = card_A(t)
    A, _ = stamp_on(A, t, 9.4)
    T.place_card(bg, A, W / 2 - 6 + 3 * math.sin(t * 0.7), ay, arot, lift=1.0 + 0.6 * math.sin(math.pi * pup))
    # card B slides in from below (7.9-8.8)
    pb = ease2(seg(t, 7.9, 8.8))
    if pb > 0:
        B = card_B.copy()
        q1 = seg(t, 9.9, 10.25); q2 = seg(t, 10.3, 10.65)
        if q1 > 0:
            B = B * lerp(1.0, 0.9, q1)
            put(B, diag_wipe(X1, q1 * 1.05, (70, 120), (840, 620)), 0, 0, INKC, 0.92)
        if q2 > 0:
            put(B, diag_wipe(X2, q2 * 1.05, (830, 90), (90, 640)), 0, 0, INKC, 0.92)
        T.place_card(bg, B, W / 2 + 6, lerp(2400, 1335, pb), lerp(6.0, -1.0, pb), lift=0.8)
    header(bg, ease(seg(t, 8.3, 9.0)))
    # impact: shake + faint flash
    k = seg(t, 9.54, 9.85)
    if 0 < k < 1:
        amp = 9 * (1 - k) ** 2
        M = np.float32([[1, 0, amp * math.sin(fi * 2.3)], [0, 1, amp * math.cos(fi * 3.1)]])
        bg = cv2.warpAffine(bg, M, (W, H), borderMode=cv2.BORDER_REFLECT)
        bg = bg * (1 - 0.10 * (1 - k)) + 245 * 0.10 * (1 - k)
    out = film.apply(bg, fi, t, strength=0.8, weave=True)
    return np.clip(out, 0, 255).astype(np.uint8)

if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else '/home/claude/w/anim_test.mp4'
    n = int(DUR * FPS)
    p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}',
                          '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '22',
                          '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out], stdin=subprocess.PIPE)
    only = [int(x) for x in sys.argv[2:]]
    for fi in range(n):
        if only:
            break
        p.stdin.write(frame(fi).tobytes())
        if fi % 60 == 0: print(fi, flush=True)
    p.stdin.close(); p.wait()
    for fi in only:
        cv2.imwrite(f'/home/claude/w/af_{fi}.png', frame(fi))
    print('done')
