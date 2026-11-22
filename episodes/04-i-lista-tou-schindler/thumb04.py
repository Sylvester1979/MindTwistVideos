"""Ep04 thumbnails, 1080x1920, drawn entirely in code (no AI imagery).
  V1 'list'  : a typed page of the list, close; «1.200» typed large, «ονόματα»; one red pencil ring
               around No. 173 (Poldek Pfefferberg's number on the list) - the only colour.
  V2 'number': black, a faint typed list behind, «1.200» in serif, a thin red rule, «ονόματα. μία λίστα.»
-> $MT_ROOT/thumb_v1.jpg, thumb_v2.jpg"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'engine'))
import cv2, numpy as np
from PIL import Image, ImageDraw
from eng import *

INKC = np.array([38, 40, 46], np.float32)                 # BGR ink
REDP = np.array([42, 44, 178], np.float32)                # BGR red pencil
rng = np.random.default_rng(173)


def alpha_text(txt, kind, size, wght=None, track=0):
    f = font(kind, size, wght)
    w = int(sum(f.getlength(c) + track for c in txt) + size)
    im = Image.new('L', (w, int(size * 1.6)), 0); d = ImageDraw.Draw(im)
    x = size / 2
    for c in txt:
        d.text((x, size * 0.2), c, font=f, fill=255); x += f.getlength(c) + track
    a = np.asarray(im).astype(np.float32) / 255
    ys, xs = np.where(a > 0.02)
    return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def typed(a, seed):
    """typewriter ink: uneven pressure, slight bleed, a few starved spots."""
    h, w = a.shape
    p = 0.72 + 0.28 * fbm(h, w, 9, seed, 2)
    starve = (fbm(h, w, 3, seed + 5, 1) > 0.18).astype(np.float32) * 0.25 + 0.75
    a = cv2.GaussianBlur(a, (0, 0), 0.9) * p * starve
    return np.clip(a * 1.15, 0, 1)


def put(img, a, x, y, color, op=1.0):
    """paste alpha a with its top-left at (x, y)."""
    h, w = a.shape
    x0, y0 = max(0, x), max(0, y); x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0: return img
    m = a[y0 - y:y1 - y, x0 - x:x1 - x, None] * op
    img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - m) + color * m
    return img


def paper():
    base = np.array([196, 214, 226], np.float32)          # warm, aged
    img = np.ones((H, W, 3), np.float32) * base
    img *= (0.88 + 0.12 * fbm(H, W, 260, 7, 4))[..., None]                 # broad tone
    img *= (0.95 + 0.05 * fbm(H, W, 18, 8, 3))[..., None]                  # fibres
    st = np.clip((fbm(H, W, 420, 9, 3) - 0.55) * 4, 0, 1)                  # a few stains
    img = img * (1 - 0.10 * st[..., None]) + np.array([120, 150, 175], np.float32) * 0.10 * st[..., None]
    # one old fold across the page
    y = np.arange(H, dtype=np.float32)[:, None]
    fold = np.exp(-((y - 1180) / 5) ** 2) * 0.10 + np.exp(-((y - 1188) / 14) ** 2) * -0.04
    img *= (1 - fold)[..., None]
    return img


def list_rows(img, n0, n1, y0, step, x_num, alpha_ink=0.9, seed=1, color=INKC):
    """numbered rows: typewriter numbers, names as illegible typed marks (no invented names)."""
    r = np.random.default_rng(seed)
    pos = {}
    for i, n in enumerate(range(n0, n1)):
        y = int(y0 + i * step)
        a = typed(alpha_text(f'{n}.', 'type', 44), seed + n)
        put(img, a, x_num - a.shape[1], y - a.shape[0] // 2, color, alpha_ink)
        pos[n] = (x_num - a.shape[1], y, a.shape[1], a.shape[0])
        x = x_num + 60
        for _ in range(int(r.integers(2, 4))):
            L = int(r.integers(140, 300))
            lay = np.zeros((40, L), np.float32)
            for k in range(L // 19):
                hgt = int(r.integers(16, 27)); t = 20 - hgt // 2 + int(r.integers(-2, 3))
                cv2.rectangle(lay, (k * 19, t), (k * 19 + 12, t + hgt), float(0.5 + 0.4 * r.random()), -1)
            put(img, typed(lay, seed + n * 7 + x), x, y - 20, color, alpha_ink)
            x += L + 46
            if x > W - 140: break
    return pos


def pencil_ring(img, cx, cy, rx, ry, seed=3):
    """a quick hand-drawn red pencil ring: 1.15 turns, wobbling radius, grainy stroke."""
    r = np.random.default_rng(seed)
    lay = np.zeros((H, W), np.float32)
    t = np.linspace(0, 2 * math.pi * 1.15, 400) - 0.6
    wob = 1 + 0.05 * np.sin(t * 3 + r.random() * 6) + 0.03 * np.sin(t * 7 + r.random() * 6)
    x = cx + rx * wob * np.cos(t) + np.linspace(-6, 8, len(t))
    y = cy + ry * wob * np.sin(t) * (1 + 0.04 * np.cos(t))
    pts = np.stack([x, y], 1)
    for k in range(3):                                      # a few passes = pencil body
        j = pts + r.normal(0, 1.6, pts.shape) + r.normal(0, 3, 2)
        cv2.polylines(lay, [np.int32(j * 4)], False, 0.8, 12, cv2.LINE_AA, shift=4 // 2)
    grain = (fbm(H, W, 2, seed + 9, 1) > 0.25).astype(np.float32) * 0.35 + 0.65
    lay = np.clip(cv2.GaussianBlur(lay, (0, 0), 0.8) * grain, 0, 0.92)[..., None]
    return img * (1 - lay) + REDP * lay


def finish(img, name, grain=0.45):
    img = img * vignette()
    img = Film(seed=4).apply(img, 2, 0.1, strength=grain, weave=False)
    cv2.imwrite(f'{ROOT}/{name}', np.clip(img, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 93])
    print('thumb ->', f'{ROOT}/{name}')


# ---------------------------------------------------------------- V1: the page
img = paper()
# «1.200» typed, with a small full stop (the typewriter's own stop is a heavy blob at this size)
_one, _dot, _two = alpha_text('1', 'type', 330), alpha_text('.', 'type', 170), alpha_text('200', 'type', 330)
_gap = 26
_hh = max(_one.shape[0], _two.shape[0])
big = np.zeros((_hh, _one.shape[1] + _gap + _dot.shape[1] + _gap + _two.shape[1]), np.float32)
big[_hh - _one.shape[0]:, :_one.shape[1]] = _one                                   # bottom-aligned
big[_hh - _dot.shape[0] - 4:_hh - 4, _one.shape[1] + _gap:_one.shape[1] + _gap + _dot.shape[1]] = _dot
big[_hh - _two.shape[0]:, _one.shape[1] + 2 * _gap + _dot.shape[1]:] = _two
big = typed(big, 11)
put(img, big, (W - big.shape[1]) // 2, 230, INKC, 0.96)
word = alpha_text('ονόματα', 'serif', 132, 520, track=3)
put(img, word, (W - word.shape[1]) // 2, 230 + big.shape[0] + 70, INKC, 0.92)
rule = np.zeros((6, 760), np.float32); rule[2:4] = 1
put(img, typed(rule, 21), (W - 760) // 2, 230 + big.shape[0] + 70 + word.shape[0] + 70, INKC, 0.8)
pos = list_rows(img, 167, 181, 1010, 66, 270, 0.85, seed=5)
x, y, w, h = pos[173]
# shallow focus: the list goes soft away from No. 173 (and the typed 'names' are never crisp)
yy = np.arange(H, dtype=np.float32)[:, None]; xx = np.arange(W, dtype=np.float32)[None, :]
sig = np.clip(np.abs(yy - y) / 95, 0, 6) + np.where(xx > x + w + 40, 1.1, 0.0)
sig = np.where(yy < 960, 0.0, sig)
levels = [0, 1, 2, 3.5, 5.5]
blurs = [img] + [cv2.GaussianBlur(img, (0, 0), l) for l in levels[1:]]
out = np.zeros_like(img)
for i, l in enumerate(levels):
    lo = levels[i - 1] if i else -1; hi = levels[i + 1] if i + 1 < len(levels) else 99
    wgt = np.clip(1 - np.abs(sig - l) / np.where(sig < l, l - lo, hi - l), 0, 1)[..., None]
    out += blurs[i] * wgt
img = out
img = pencil_ring(img, x + w / 2 + 4, y + 2, w / 2 + 44, h / 2 + 30)
finish(img, 'thumb_v1.jpg')

# ---------------------------------------------------------------- V2: the number
img = np.ones((H, W, 3), np.float32) * np.array([16, 17, 19], np.float32)
img *= (0.85 + 0.3 * fbm(H, W, 300, 12, 3))[..., None]
list_rows(img, 1, 31, 120, 60, 210, 0.13, seed=9, color=np.array([200, 210, 218], np.float32))   # the list, barely there
img = band(img, 520, 1300, 0.75, 220)
num = alpha_text('1.200', 'serif', 330, 600, track=4)
put(img, cv2.GaussianBlur(num, (0, 0), 0.6), (W - num.shape[1]) // 2, 640, np.array([230, 236, 240], np.float32), 1.0)
ry = 640 + num.shape[0] + 70
rl = np.zeros((8, 320), np.float32); rl[2:6] = 1
put(img, rl, (W - 320) // 2, ry, REDP * 1.25, 0.95)
sub = alpha_text('ονόματα σε μια λίστα', 'serif', 84, 480, track=2)
put(img, sub, (W - sub.shape[1]) // 2, ry + 70, np.array([205, 212, 218], np.float32), 1.0)
finish(img, 'thumb_v2.jpg', 0.35)
