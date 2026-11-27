"""Textures for the ep05 thumbnail scene (Blender 5.2, Cycles). Plain PIL + numpy, run once before mt_ep05_thumb.py.
  paper_col.png / paper_bump.png : the sheet, A4 at 300 dpi. The two names in the wording the source gives
                                   (Δ. Βίτσος): «Ο Ζακύνθου Χρυσόστομος», «Δήμαρχος Ζακυνθίων Λουκάς Καρρέρ»,
                                   handwritten with a fountain pen (hand05.py: pen strokes + nib + ink, all code).
                                   Written large so they read on a phone; line breaks are layout only.
  folder_col.png / folder_bump.png: a manila folder cover with a red rubber stamp «ΖΑΚΥΝΘΟΣ 1943» (the title, in the scene).
usage: python3 make_thumb_tex.py <outdir> <unused, kept for the old call> <EBGaramond.ttf>   (hand05.py beside it)"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hand05
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

OUT, MONO, SERIF = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(1943)


def noise(h, w, scale, seed, octaves=4):
    """smooth value noise in [0,1] (sum of upsampled random grids)."""
    r = np.random.default_rng(seed); out = np.zeros((h, w), np.float32); amp = 1.0; tot = 0.0
    for o in range(octaves):
        s = max(1, int(scale / (2 ** o)))
        g = r.random((h // s + 2, w // s + 2)).astype(np.float32)
        im = Image.fromarray((g * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
        out += amp * np.asarray(im, np.float32) / 255; tot += amp; amp *= 0.5
    return out / tot


def fibres(h, w, seed, n=9000):
    """paper fibres: short thin random strokes."""
    im = Image.new('L', (w, h), 0); d = ImageDraw.Draw(im); r = np.random.default_rng(seed)
    for _ in range(n):
        x, y = r.random() * w, r.random() * h; a = r.random() * np.pi; L = 6 + r.random() * 28
        d.line([(x, y), (x + L * np.cos(a), y + L * np.sin(a))], fill=int(40 + r.random() * 90), width=1)
    return np.asarray(im.filter(ImageFilter.GaussianBlur(0.6)), np.float32) / 255


# ---------------------------------------------------------------- the sheet: A4 at 300 dpi
H, W = 3508, 2480
base = np.array([236, 228, 210], np.float32) / 255                               # aged off-white (RGB)
col = np.ones((H, W, 3), np.float32) * base
col *= (0.94 + 0.06 * noise(H, W, 400, 1))[..., None]
fb = fibres(H, W, 2)
col *= (1 - 0.05 * fb)[..., None]
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
edge = np.minimum.reduce([xx, yy, W - 1 - xx, H - 1 - yy]) / 90.0                 # edges yellowed by handling
col *= (1 - 0.10 * np.exp(-edge))[..., None] * np.array([1.0, 0.97, 0.90])[None, None]
fox = np.clip((noise(H, W, 40, 5, 3) - 0.80) * 6, 0, 1) * (noise(H, W, 900, 6, 2) > 0.62)   # a little foxing
col = col * (1 - 0.18 * fox[..., None]) + np.array([0.62, 0.48, 0.30]) * 0.18 * fox[..., None]
# the two names (the source's wording), handwritten: x-height 100 px (8.5 mm), left margin 28 mm, first baseline 47 mm
lines = ['Ο Ζακύνθου', 'Χρυσόστομος', '', 'Δήμαρχος Ζακυνθίων', 'Λουκάς Καρρέρ']
D, PR, DRY = hand05.write(lines, 100, (H, W), (330, 560), seed=1943, line_gap=2.6)
grain = np.clip(0.6 * noise(H, W, 2, 31, 2) + 0.4 * fb * 2.5, 0, 1)                  # fibre tops (dry strokes, take-up)
fibre_map = np.clip(0.5 * noise(H, W, 6, 32, 2) + 0.5 * fb * 2.5, 0, 1)             # where the ink creeps a little
D = hand05.ink_on_paper(D, DRY, fibre_map, grain)
col = col * np.exp(-D[..., None] * hand05.INK_BROWN_BLACK)                        # brown-black, translucent dye
Image.fromarray((np.clip(col, 0, 1) * 255).astype(np.uint8)).save(f'{OUT}/paper_col.png')
cov = np.clip(D, 0, 1)
dent = np.asarray(Image.fromarray((PR * cov * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.5)), np.float32) / 255
bump = 0.5 + 0.10 * (noise(H, W, 6, 9, 3) - 0.5) + 0.12 * fb - 0.10 * cov * 0.4 - 0.16 * dent   # the nib presses a groove
Image.fromarray((np.clip(bump, 0, 1) * 255).astype(np.uint8)).save(f'{OUT}/paper_bump.png')
Image.fromarray((np.clip(cov, 0, 1) * 255).astype(np.uint8)).save(f'{OUT}/paper_ink.png')   # ink mask (sheen)

# ---------------------------------------------------------------- the folder cover: 300 x 240 mm at 10 px/mm
FH, FW = 3000, 2400
mb = np.array([214, 178, 118], np.float32) / 255                                   # manila
fc = np.ones((FH, FW, 3), np.float32) * mb
fc *= (0.90 + 0.10 * noise(FH, FW, 300, 11))[..., None]
ff = fibres(FH, FW, 12, 14000)
fc *= (1 - 0.07 * ff)[..., None]
fy, fx = np.mgrid[0:FH, 0:FW].astype(np.float32)
fedge = np.minimum.reduce([fx, fy, FW - 1 - fx, FH - 1 - fy]) / 60.0
fc *= (1 - 0.22 * np.exp(-fedge))[..., None]                                         # worn, darker edges
# the stamp: «ΖΑΚΥΝΘΟΣ 1943» in the video's serif, bold, inside a double frame; uneven red ink
st = Image.new('L', (FW, FH), 0); d = ImageDraw.Draw(st)
sf = ImageFont.truetype(SERIF, 176)
try: sf.set_variation_by_axes([800])
except Exception: pass
txt = 'ΖΑΚΥΝΘΟΣ 1943'
tw = sf.getlength(txt) + 12 * (len(txt) - 1)
cx, cy = FW / 2, 470
x = cx - tw / 2
for c in txt:
    d.text((x, cy - 118), c, font=sf, fill=255); x += sf.getlength(c) + 12
pad = 60
d.rectangle([cx - tw / 2 - pad, cy - 140, cx + tw / 2 + pad, cy + 140], outline=255, width=12)
d.rectangle([cx - tw / 2 - pad + 22, cy - 118, cx + tw / 2 + pad - 22, cy + 118], outline=255, width=4)
st = st.rotate(-3.0, resample=Image.BICUBIC, center=(cx, cy))
sa = np.asarray(st.filter(ImageFilter.GaussianBlur(1.6)), np.float32) / 255
# a real stamp is never perfect (the user, 26/11): uneven pressure across it, ink starved in patches, a faint second
# strike where the hand moved, the ink bleeding a little into the card
ghost = np.roll(np.roll(sa, 7, 0), -5, 1) * 0.22
sa = np.maximum(sa, ghost)
press = np.clip(0.50 + 0.65 * (fx - (cx - tw / 2)) / tw, 0.45, 1.05) * (0.75 + 0.35 * noise(FH, FW, 160, 24, 2))
grit = np.clip(0.30 + 0.95 * noise(FH, FW, 4, 21, 3), 0, 1) * (noise(FH, FW, 70, 22, 3) * 0.55 + 0.45)
voids = (noise(FH, FW, 26, 25, 2) > 0.62).astype(np.float32)
grit *= 1 - 0.75 * np.asarray(Image.fromarray((voids * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(3)), np.float32) / 255
sa = np.clip(sa * grit * press * 1.25, 0, 0.93)
sa = np.maximum(sa, np.asarray(Image.fromarray((sa * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(4)), np.float32) / 255 * 0.25)
red = np.array([0.62, 0.10, 0.09])
fc = fc * (1 - sa[..., None]) + red * sa[..., None] * fc / mb.max() * 1.0
Image.fromarray((np.clip(fc, 0, 1) * 255).astype(np.uint8)).save(f'{OUT}/folder_col.png')
fbmp = 0.5 + 0.12 * (noise(FH, FW, 5, 23, 3) - 0.5) + 0.15 * ff - 0.08 * sa
Image.fromarray((np.clip(fbmp, 0, 1) * 255).astype(np.uint8)).save(f'{OUT}/folder_bump.png')
print('textures ->', OUT)
