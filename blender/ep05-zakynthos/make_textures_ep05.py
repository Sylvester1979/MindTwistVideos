"""Textures for the ep05 Blender clips (all drawn here, no AI):
  names.png  the paper with the two names: typed Greek, exactly as D. Vitsos records them; below, two lines of pen
             scribble standing for the German version (its wording is not recorded, so it is never legible)
  list.png   a page of the list: typed header and rows of neutral 'typed' marks (no real names)
  wood.png   a dark desk top (fine grain), only for luminance — the finish turns everything to ink and paper
usage: python3 make_textures_ep05.py [out_dir]"""
import sys, os, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'textures')
os.makedirs(OUT, exist_ok=True)
MONO = '/usr/share/fonts/truetype/freefont/FreeMono.ttf'
rng = np.random.default_rng(5)

def fbm(h, w, scale, seed, oct=4):
    r = np.random.default_rng(seed); out = np.zeros((h, w), np.float32); amp, tot = 1.0, 0.0
    for o in range(oct):
        s = max(2, int(scale / 2 ** o))
        g = r.random((h // s + 2, w // s + 2)).astype(np.float32)
        out += amp * np.asarray(Image.fromarray((g * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC), np.float32) / 255
        tot += amp; amp *= 0.5
    return out / tot

def paper(w, h, seed):
    base = np.float32([232, 222, 200])                     # aged paper (RGB)
    n = 0.90 + 0.10 * fbm(h, w, 160, seed) + 0.04 * (fbm(h, w, 6, seed + 1, 2) - 0.5)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    edge = np.minimum.reduce([x, y, w - 1 - x, h - 1 - y])
    n *= 1 - 0.18 * np.exp(-edge / (0.03 * w))
    return np.clip(base[None, None] * n[..., None], 0, 255)

def typed(img, xy, text, size, seed, ink=(30, 28, 30)):
    """typewriter text: each glyph slightly offset and with uneven ink."""
    f = ImageFont.truetype(MONO, size); d = ImageDraw.Draw(img); r = np.random.default_rng(seed)
    x, y = xy
    for ch in text:
        a = int(170 + 85 * r.random())
        d.text((x + r.normal(0, 0.6), y + r.normal(0, 0.8)), ch, font=f, fill=ink + (a,))
        x += f.getlength(ch)
    return x

def scribble(d, x0, y, length, seed, h=34, width=3):
    """a line of cursive-looking handwriting (illegible on purpose): loops of varying height, slanted, in words."""
    r = np.random.default_rng(seed); x = x0
    while x < x0 + length:
        pts = []; nlet = int(r.integers(3, 9))
        for _ in range(nlet):
            s_ = r.uniform(16, 24); a = h * (2.0 if r.random() < 0.22 else r.uniform(0.7, 1.0))
            for t in np.linspace(0, 1, 14):
                yy = -a * (0.5 - 0.5 * math.cos(2 * math.pi * t))
                xx = x + s_ * t + 0.35 * s_ * math.sin(2 * math.pi * t) - 0.28 * yy      # loop + slant
                pts.append((xx, y + yy))
            x += s_
        d.line(pts, fill=(25, 24, 30, 230), width=width, joint='curve')
        x += r.uniform(24, 40)

def save(arr, name):
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(os.path.join(OUT, name))

# ---- the paper with the two names (15 x 21 cm, 1500 x 2100 px)
W, H = 1500, 2100
img = Image.fromarray(paper(W, H, 11).astype(np.uint8)).convert('RGBA')
ink = Image.new('RGBA', img.size, (0, 0, 0, 0))
typed(ink, (130, 560), 'Ο Ζακύνθου Χρυσόστομος', 76, 1)
typed(ink, (130, 700), 'Δήμαρχος Ζακυνθίων Λουκάς Καρρέρ', 62, 2)
d = ImageDraw.Draw(ink)
scribble(d, 150, 1000, 900, 3)
scribble(d, 150, 1120, 1100, 4)
ink = ink.filter(ImageFilter.GaussianBlur(0.7))
img = Image.alpha_composite(img, ink).convert('RGB')
save(np.asarray(img, np.float32), 'names.png')

# ---- a page of the list (A4, 1400 x 1980 px): header + rows of neutral typed marks
W, H = 1400, 1980
img = Image.fromarray(paper(W, H, 21).astype(np.uint8)).convert('RGBA')
ink = Image.new('RGBA', img.size, (0, 0, 0, 0)); d = ImageDraw.Draw(ink)
f = ImageFont.truetype(MONO, 44)
typed(ink, (110, 140), 'ΟΝΟΜΑ', 44, 7); typed(ink, (620, 140), 'ΔΙΕΥΘΥΝΣΙΣ', 44, 8); typed(ink, (1080, 140), 'ΠΕΡΙΟΥΣΙΑ', 44, 9)
d.line([(100, 210), (1300, 210)], fill=(30, 28, 30, 220), width=3)
r = np.random.default_rng(12)
ALPHA = 'ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ'
for row in range(26):                                     # random letters: they read as typed entries, they are nobody
    y = 250 + row * 62
    for x0, n0, n1 in ((110, 5, 10), (620, 6, 12), (1080, 3, 7)):
        txt = ''.join(r.choice(list(ALPHA)) for _ in range(int(r.integers(n0, n1))))
        if r.random() < 0.6 and x0 != 1080:
            txt += ' ' + ''.join(r.choice(list(ALPHA)) for _ in range(int(r.integers(3, 7))))
        typed(ink, (x0, y), txt, 34, int(r.integers(1e6)))
ink = ink.filter(ImageFilter.GaussianBlur(0.8))
img = Image.alpha_composite(img, ink).convert('RGB')
save(np.asarray(img, np.float32), 'list.png')

# ---- desk top (dark, fine grain)
W, H = 2048, 2048
y = np.arange(H, dtype=np.float32)[:, None]
grain = 0.5 + 0.5 * np.sin(y * 0.09 + 6 * fbm(H, W, 300, 33))
v = 0.55 + 0.25 * grain + 0.2 * fbm(H, W, 120, 34)
save(np.float32([70, 52, 38])[None, None] * v[..., None], 'wood.png')
print('textures ->', OUT)
