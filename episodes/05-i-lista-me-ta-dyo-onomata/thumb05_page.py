"""Ep05 thumbnail «the page», 1080x1920, drawn entirely in code (third version, 26/11):
close on the sheet itself, as ep. 4's thumb_v1: «ΖΑΚΥΝΘΟΣ 1943» set at the head of the page in the video's serif,
the two names typed large as the hero (wording exactly as the source gives it: Δ. Βίτσος), lamplight from the upper left,
shallow focus on the names, paper fibres, a fold, film grain.
usage: MT_ROOT=<dir with fonts/EBGaramond.ttf> python3 thumb05_page.py <out.jpg>"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'engine'))
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
from eng import fbm, Film, font, W, H

OUT = sys.argv[1]
MONO = '/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf'     # the video's typewriter face, bold
INKC = np.array([30, 32, 40], np.float32)                        # BGR typewriter ink
rng = np.random.default_rng(1943)
# the source's wording; the line breaks are layout only
ENTRIES = [['Ο Ζακύνθου', 'Χρυσόστομος'], ['Δήμαρχος', 'Ζακυνθίων', 'Λουκάς Καρρέρ']]


def glyph(c, f):
    im = Image.new('L', (int(f.size * 1.4), int(f.size * 1.6)), 0)
    ImageDraw.Draw(im).text((f.size * 0.2, f.size * 0.15), c, font=f, fill=255)
    return np.asarray(im).astype(np.float32) / 255


def typed_line(txt, f, seed):
    """a typewriter line: each key struck on its own (baseline jitter, uneven pressure, a little bleed)."""
    r = np.random.default_rng(seed)
    adv = f.getlength('M')
    w = int(adv * len(txt) + f.size + 12); h = int(f.size * 1.7) + 12
    lay = np.zeros((h, w), np.float32)
    for i, c in enumerate(txt):
        if c == ' ': continue
        g = glyph(c, f)
        dy = int(r.normal(0, 1.6)); dx = int(r.normal(0, 0.8))
        pressure = float(np.clip(r.normal(0.92, 0.07), 0.72, 1.0))
        x0 = 6 + int(i * adv) + dx; y0 = 6 + dy
        hh, ww = g.shape
        sub = lay[max(0, y0):y0 + hh, x0:x0 + ww]
        sub[:] = np.maximum(sub, g[:sub.shape[0], :sub.shape[1]] * pressure)
    starve = (fbm(h, w, 3, seed + 5, 1) > 0.2).astype(np.float32) * 0.2 + 0.8
    tex = 0.80 + 0.20 * fbm(h, w, 6, seed, 2)
    a = np.clip(cv2.GaussianBlur(lay, (0, 0), 0.9) * starve * tex * 1.12, 0, 1)
    ys, xs = np.where(a > 0.03)
    return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def put(img, a, x, y, color, op=1.0):
    h, w = a.shape
    x0, y0 = max(0, x), max(0, y); x1, y1 = min(img.shape[1], x + w), min(img.shape[0], y + h)
    m = a[y0 - y:y1 - y, x0 - x:x1 - x, None] * op
    img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - m) + color * m


# ---------------------------------------------------------------- the page, a little larger than the frame
PW, PH = 1300, 2200
page = np.ones((PH, PW, 3), np.float32) * np.array([212, 226, 236], np.float32)      # the plates' paper
page *= (0.90 + 0.10 * fbm(PH, PW, 300, 7, 4))[..., None]                            # broad tone
page *= (0.955 + 0.045 * fbm(PH, PW, 14, 8, 3))[..., None]                            # fibres
st = np.clip((fbm(PH, PW, 380, 9, 3) - 0.56) * 4, 0, 1)                               # an old stain or two
page = page * (1 - 0.12 * st[..., None]) + np.array([110, 145, 175], np.float32) * 0.12 * st[..., None]
yy = np.arange(PH, dtype=np.float32)[:, None]
FOLD = 1610
page *= (1 - (np.exp(-((yy - FOLD) / 4) ** 2) * 0.10 + np.exp(-((yy - FOLD - 7) / 16) ** 2) * -0.035))[..., None]

ox, oy = 110, 140                                    # the frame's top-left inside the page (before rotation)
# heading: the video's serif, in ink, as if printed at the head of the page
ttl_f = font('serif', 150, 620)
tl = Image.new('L', (1400, 260), 0); d = ImageDraw.Draw(tl)
x = 40
for c in 'ΖΑΚΥΝΘΟΣ 1943':
    d.text((x, 30), c, font=ttl_f, fill=255); x += ttl_f.getlength(c) + 8
ttl = np.asarray(tl).astype(np.float32) / 255
ys, xs = np.where(ttl > 0.02); ttl = ttl[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
if ttl.shape[1] > 860: ttl = cv2.resize(ttl, (860, int(ttl.shape[0] * 860 / ttl.shape[1])), interpolation=cv2.INTER_AREA)
ty = oy + 170
put(page, cv2.GaussianBlur(ttl, (0, 0), 0.6) * (0.88 + 0.12 * fbm(*ttl.shape, 5, 3, 2)), ox + (W - ttl.shape[1]) // 2 - 22, ty, INKC, 0.94)   # -22: centred after the page's turn
ry = ty + ttl.shape[0] + 55
put(page, np.ones((3, 620), np.float32), ox + (W - 620) // 2 - 20, ry, INKC, 0.55)

# the two names: typed large, left-aligned like a typed page
mono = ImageFont.truetype(MONO, 110)
lines = [l for e in ENTRIES for l in e]
widest = max(mono.getlength('M') * len(l) for l in lines)
sc = min(1.0, 880 / widest)
mono = ImageFont.truetype(MONO, int(110 * sc))
xl = ox + 95
y = ry + 240
centre = []
for i, e in enumerate(ENTRIES):
    for j, l in enumerate(e):
        a = typed_line(l, mono, 100 + i * 10 + j)
        put(page, a, xl, y, INKC, 0.97)
        centre.append(y + a.shape[0] / 2)
        y += int(mono.size * 1.32)
    y += int(mono.size * 0.9)
focus_y = (centre[0] + centre[-1]) / 2

# ---------------------------------------------------------------- into the frame: a slight turn, then light and focus
M = cv2.getRotationMatrix2D((ox + W / 2, oy + H / 2), -2.2, 1.0)
M[:, 2] -= (ox, oy)
img = cv2.warpAffine(page, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
fy = focus_y - oy
# shallow focus: the names sharp, the page softening above and below (the heading kept readable)
Y = np.arange(H, dtype=np.float32)[:, None] * np.ones((1, W), np.float32)
sig = np.clip((np.abs(Y - fy) - 520) / 220, 0, 3.2)
sig = np.where(Y < ry - oy + 40, np.minimum(sig, 0.6), sig)
levels = [0, 1, 2, 3.2, 4.5]
blurs = [img] + [cv2.GaussianBlur(img, (0, 0), l * 1.5) for l in levels[1:]]
out = np.zeros_like(img)
for i, l in enumerate(levels):
    lo = levels[i - 1] if i else -1; hi = levels[i + 1] if i + 1 < len(levels) else 99
    wgt = np.clip(1 - np.abs(sig - l) / np.where(sig < l, l - lo, hi - l), 0, 1)[..., None]
    out += blurs[i] * wgt
img = out
# lamplight: warm from the upper left, falling off into a deep vignette
Yg, Xg = np.mgrid[0:H, 0:W].astype(np.float32)
dd = np.sqrt(((Xg - W * 0.30) / 1050) ** 2 + ((Yg - H * 0.40) / 1250) ** 2)
light = np.clip(1.30 - dd, 0.16, 1.0) ** 1.05
warm = np.array([0.86, 0.96, 1.04], np.float32)                                      # BGR: a touch of lamp
img = img * light[..., None] * warm
img *= (1 - 0.35 * np.clip((np.abs(Xg - W / 2) / (W / 2)) ** 3, 0, 1))[..., None]     # darker side edges
img = Film(seed=7).apply(img, 2, 0.1, strength=0.38, weave=False)
cv2.imwrite(OUT, np.clip(img, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 94])
print('thumb ->', OUT)
