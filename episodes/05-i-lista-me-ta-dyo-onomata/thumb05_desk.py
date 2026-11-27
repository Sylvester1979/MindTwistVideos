"""Ep05 thumbnail «the list on the desk», 1080x1920, drawn entirely in code (the user's idea, 26/11, second version):
the typed sheet with the two names lies on a dark desk in the pool of a desk lamp, slightly turned, with its shadow;
«ΖΑΚΥΝΘΟΣ 1943» at the very top, in the video's serif, on the dark.
usage: MT_ROOT=<dir with fonts/EBGaramond.ttf> python3 thumb05_desk.py <out.jpg>"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'engine'))
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
from eng import fbm, Film, font, W, H

OUT = sys.argv[1]
MONO = '/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf'     # the video's typewriter face, bold for a phone screen
INKC = np.array([34, 36, 42], np.float32)                        # BGR typewriter ink
LAMP = np.array([150, 205, 255], np.float32) / 255               # BGR warm lamplight
# the wording as the source gives it (Δ. Βίτσος, «Οι Ζακυνθινοί Εβραίοι»; as on the video's typed page), no numbering:
# the user (26/11) wants it historically exact rather than his own wording
NAMES = [('Ο Ζακύνθου', 'Χρυσόστομος'), ('Δήμαρχος Ζακυνθίων', 'Λουκάς Καρρέρ')]


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
    """typewriter ink: uneven pressure, slight bleed, a few starved spots (as ep. 4)."""
    h, w = a.shape
    p = 0.74 + 0.26 * fbm(h, w, 9, seed, 2)
    starve = (fbm(h, w, 3, seed + 5, 1) > 0.18).astype(np.float32) * 0.22 + 0.78
    return np.clip(cv2.GaussianBlur(a, (0, 0), 0.8) * p * starve * 1.15, 0, 1)


def put(img, a, x, y, color, op=1.0):
    h, w = a.shape
    x0, y0 = max(0, x), max(0, y); x1, y1 = min(img.shape[1], x + w), min(img.shape[0], y + h)
    m = a[y0 - y:y1 - y, x0 - x:x1 - x, None] * op
    img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - m) + color * m


# ---------------------------------------------------------------- the sheet (flat), A4 proportions
PW, PH = 1000, 1414
sheet = np.ones((PH, PW, 3), np.float32) * np.array([200, 218, 230], np.float32)
sheet *= (0.90 + 0.10 * fbm(PH, PW, 240, 7, 4))[..., None]
sheet *= (0.95 + 0.05 * fbm(PH, PW, 16, 8, 3))[..., None]
yy = np.arange(PH, dtype=np.float32)[:, None]
sheet *= (1 - (np.exp(-((yy - PH * 0.5) / 4) ** 2) * 0.08 + np.exp(-((yy - PH * 0.5 - 6) / 12) ** 2) * -0.03))[..., None]  # one fold
mono = ImageFont.truetype(MONO, 84)
xl, y = 110, 250
widest = max(alpha(l, mono).shape[1] for n in NAMES for l in n)
sc = min(1.0, (PW - xl - 110) / widest)
for i, lines in enumerate(NAMES):
    for j, l in enumerate(lines):
        a = alpha(l, mono)
        if sc < 1: a = cv2.resize(a, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        put(sheet, typed(a, 50 + i * 10 + j), xl, y + j * 112, INKC, 0.97)
    y += 360

# ---------------------------------------------------------------- the desk, the sheet on it, its shadow, the lamp
desk = np.ones((H, W, 3), np.float32) * np.array([22, 26, 32], np.float32)
grain = cv2.resize(fbm(H // 8, W, 40, 3, 4), (W, H), interpolation=cv2.INTER_CUBIC)          # long wood grain
desk *= (0.75 + 0.5 * grain)[..., None]
cx, cy, ang = W * 0.53, 1250, -5.0                                                           # where the sheet lies
S = 1.06                                                                                    # sheet scale: it runs off the frame, the names fill it
src = np.float32([[0, 0], [PW, 0], [PW, PH], [0, PH]])
c, s = np.cos(np.radians(ang)), np.sin(np.radians(ang))
def place(px, py):
    x, y = (px - PW / 2) * S, (py - PH / 2) * S
    y *= 1 - 0.06 * (py / PH - 0.5)                                                        # a little perspective: far edge narrower
    x *= 1 - 0.05 * (0.5 - py / PH)
    return cx + x * c - y * s, cy + x * s + y * c
dst = np.float32([place(*p) for p in src])
M = cv2.getPerspectiveTransform(src, dst)
warped = cv2.warpPerspective(sheet, M, (W, H), flags=cv2.INTER_CUBIC)
mask = cv2.warpPerspective(np.ones((PH, PW), np.float32), M, (W, H), flags=cv2.INTER_LINEAR)
mask = cv2.GaussianBlur(mask, (0, 0), 0.7)
sh = cv2.GaussianBlur(np.roll(np.roll(mask, 26, 0), 18, 1), (0, 0), 22)                    # soft shadow, lamp top-left
img = desk * (1 - 0.65 * sh)[..., None]
curl = cv2.warpPerspective(np.tile(np.linspace(0.93, 1.0, PW, dtype=np.float32), (PH, 1)), M, (W, H))
img = img * (1 - mask[..., None]) + warped * (curl * mask)[..., None]
# lamp: a warm pool on the sheet, falling off to near black at the frame
Y, X = np.mgrid[0:H, 0:W].astype(np.float32)
d = np.sqrt(((X - W * 0.42) / 820) ** 2 + ((Y - 980) / 900) ** 2)
light = np.clip(1.22 - d, 0.05, 1.0) ** 1.35
img = img * (light[..., None] * (0.55 + 0.45 * LAMP))
img *= (0.92 + 0.08 * fbm(H, W, 300, 11, 3))[..., None]

# ---------------------------------------------------------------- «ΖΑΚΥΝΘΟΣ 1943» at the very top
ttl = alpha('ΖΑΚΥΝΘΟΣ 1943', font('serif', 132, 600), track=6)
if ttl.shape[1] > 900: ttl = cv2.resize(ttl, (900, int(ttl.shape[0] * 900 / ttl.shape[1])), interpolation=cv2.INTER_AREA)
glow = cv2.GaussianBlur(ttl, (0, 0), 12)
tx, ty = (W - ttl.shape[1]) // 2, 170
put(img, glow, tx, ty, np.array([4, 4, 6], np.float32), 0.8)
put(img, ttl, tx, ty, np.array([214, 226, 236], np.float32), 1.0)                          # the plates' paper colour

img = Film(seed=5).apply(img, 2, 0.1, strength=0.35, weave=False)
cv2.imwrite(OUT, np.clip(img, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 93])
print('thumb ->', OUT)
