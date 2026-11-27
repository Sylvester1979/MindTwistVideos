"""Ep05 thumbnail «the list», 1080x1920, drawn entirely in code (the user's idea, 26/11):
«ΖΑΚΥΝΘΟΣ 1943» at the very top; the main subject is the list with its two typed names,
«ΜΗΤΡΟΠΟΛΙΤΗΣ ΖΑΚΥΝΘΟΥ ΧΡΥΣΟΣΤΟΜΟΣ» and «ΔΗΜΑΡΧΟΣ ΖΑΚΥΝΘΙΩΝ ΛΟΥΚΑΣ ΚΑΡΡΕΡ» (the user's wording).
Same paper, typewriter ink and film finish as ep. 4's thumb_v1.
usage: MT_ROOT=<dir with fonts/EBGaramond.ttf> python3 thumb05_list.py <out.jpg>"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'engine'))
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
from eng import fbm, vignette, Film, font, W, H

OUT = sys.argv[1]
MONO = '/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf'    # the video's typewriter face (FreeMono), bold so it reads on a phone
INKC = np.array([38, 40, 46], np.float32)                       # BGR ink, as ep. 4


def alpha(txt, f, track=0):
    w = int(sum(f.getlength(c) + track for c in txt) + f.size)
    im = Image.new('L', (w, int(f.size * 1.6)), 0); d = ImageDraw.Draw(im)
    x = f.size / 2
    for c in txt:
        d.text((x, f.size * 0.2), c, font=f, fill=255); x += f.getlength(c) + track
    a = np.asarray(im).astype(np.float32) / 255
    ys, xs = np.where(a > 0.02)
    return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def typed(a, seed):
    """typewriter ink: uneven pressure, slight bleed, a few starved spots (ep. 4)."""
    h, w = a.shape
    p = 0.72 + 0.28 * fbm(h, w, 9, seed, 2)
    starve = (fbm(h, w, 3, seed + 5, 1) > 0.18).astype(np.float32) * 0.25 + 0.75
    a = cv2.GaussianBlur(a, (0, 0), 0.9) * p * starve
    return np.clip(a * 1.15, 0, 1)


def put(img, a, x, y, color, op=1.0):
    h, w = a.shape
    x0, y0 = max(0, x), max(0, y); x1, y1 = min(W, x + w), min(H, y + h)
    m = a[y0 - y:y1 - y, x0 - x:x1 - x, None] * op
    img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - m) + color * m


def paper():                                                    # ep. 4's aged page
    img = np.ones((H, W, 3), np.float32) * np.array([196, 214, 226], np.float32)
    img *= (0.88 + 0.12 * fbm(H, W, 260, 7, 4))[..., None]
    img *= (0.95 + 0.05 * fbm(H, W, 18, 8, 3))[..., None]
    st = np.clip((fbm(H, W, 420, 9, 3) - 0.55) * 4, 0, 1)
    img = img * (1 - 0.10 * st[..., None]) + np.array([120, 150, 175], np.float32) * 0.10 * st[..., None]
    y = np.arange(H, dtype=np.float32)[:, None]
    fold = np.exp(-((y - 1340) / 5) ** 2) * 0.10 + np.exp(-((y - 1348) / 14) ** 2) * -0.04
    return img * (1 - fold)[..., None]


img = paper()
# «ΖΑΚΥΝΘΟΣ 1943» at the very top, in the video's serif
ttl = alpha('ΖΑΚΥΝΘΟΣ 1943', font('serif', 128, 600), track=6)
if ttl.shape[1] > 920: ttl = cv2.resize(ttl, (920, int(ttl.shape[0] * 920 / ttl.shape[1])), interpolation=cv2.INTER_AREA)
put(img, ttl, (W - ttl.shape[1]) // 2, 150, INKC, 0.95)
rule = np.zeros((6, 760), np.float32); rule[2:4] = 1
put(img, typed(rule, 21), (W - 760) // 2, 150 + ttl.shape[0] + 60, INKC, 0.8)

# the list: two numbered entries, each name typed large on two lines
NAMES = [('ΜΗΤΡΟΠΟΛΙΤΗΣ ΖΑΚΥΝΘΟΥ', 'ΧΡΥΣΟΣΤΟΜΟΣ'), ('ΔΗΜΑΡΧΟΣ ΖΑΚΥΝΘΙΩΝ', 'ΛΟΥΚΑΣ ΚΑΡΡΕΡ')]
mono = ImageFont.truetype(MONO, 76)
num_f = ImageFont.truetype(MONO, 76)
xl = 140                                                        # left edge of the names
widest = max(alpha(l, mono).shape[1] for n in NAMES for l in n)
scale = min(1.0, (W - xl - 40) / widest)
y = 640
for i, lines in enumerate(NAMES):
    n = typed(alpha(f'{i + 1}.', num_f), 40 + i)
    put(img, n, xl - 30 - n.shape[1], y, INKC, 0.92)
    for j, l in enumerate(lines):
        a = alpha(l, mono)
        if scale < 1: a = cv2.resize(a, (int(a.shape[1] * scale), int(a.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        put(img, typed(a, 50 + i * 10 + j), xl, y + j * 110, INKC, 0.96)
    y += 360

img = img * vignette()
img = Film(seed=5).apply(img, 2, 0.1, strength=0.45, weave=False)
cv2.imwrite(OUT, np.clip(img, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 93])
print('thumb ->', OUT)
