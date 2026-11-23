"""Ep05 thumbnail, 1080x1920, drawn entirely in code (no AI imagery), in the language of the video.
Two cards on the dark desk, the claims reduced to their hooks («2 ΑΤΟΜΙΚΕΣ ΒΟΜΒΕΣ» / «3 ΝΑΥΑΓΙΑ»), and one red
round stamp with a question mark between them. The answer is not shown. Red is used twice only: «ΕΝΑ ΨΕΜΑ» and the stamp.
  thumb05.py [out.jpg]"""
import os, sys, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
import cards05 as T
from eng import W, H, fbm, Film

ROOT = os.environ.get('MT_ROOT', '/home/claude/ep05')
SERIF = f'{ROOT}/fonts/EBGaramond.ttf'
INK = np.array([34, 36, 42], np.float32)
PAPER = np.array([196, 214, 226], np.float32)
LIGHT = np.array([214, 228, 238], np.float32)
RED = np.array([40, 44, 182], np.float32)

def serif_alpha(txt, size, wght=700, track=0):
    f = ImageFont.truetype(SERIF, size); f.set_variation_by_axes([wght])
    w = int(sum(f.getlength(c) + track for c in txt) + size); h = int(size * 1.6)
    im = Image.new('L', (w, h), 0); d = ImageDraw.Draw(im); x = size / 2
    for c in txt:
        d.text((x, size * 0.2), c, font=f, fill=255); x += f.getlength(c) + track
    a = np.asarray(im).astype(np.float32) / 255
    ys, xs = np.where(a > 0.02)
    return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

def inky(a, seed, k=0.12):
    h, w = a.shape
    return np.clip(a * (1 - k + k * fbm(h, w, 6, seed, 2)) * 1.05, 0, 1)

def paste(img, a, cx, cy, color, op=1.0, shadow=0.0):
    h, w = a.shape; x0, y0 = int(cx - w / 2), int(cy - h / 2)
    x1, y1 = x0 + w, y0 + h
    ax0, ay0 = max(0, -x0), max(0, -y0)
    X0, Y0, X1, Y1 = max(0, x0), max(0, y0), min(img.shape[1], x1), min(img.shape[0], y1)
    m = a[ay0:ay0 + Y1 - Y0, ax0:ax0 + X1 - X0]
    if shadow:
        sm = cv2.GaussianBlur(m, (0, 0), 8) * shadow
        img[Y0:Y1, X0:X1] *= (1 - sm[..., None] * op)
    img[Y0:Y1, X0:X1] = img[Y0:Y1, X0:X1] * (1 - m[..., None] * op) + color * m[..., None] * op
    return img

def paper(w, h, seed):
    img = np.ones((h, w, 3), np.float32) * PAPER
    img *= (0.90 + 0.10 * fbm(h, w, 220, seed, 4))[..., None]
    img *= (0.95 + 0.05 * fbm(h, w, 14, seed + 1, 3))[..., None]
    Y, X = np.mgrid[0:h, 0:w].astype(np.float32)
    e = np.minimum.reduce([X, Y, w - 1 - X, h - 1 - Y])
    img *= (1 - 0.18 * np.exp(-e / 26))[..., None]
    return img

def card(letter, small, big, drawing, seed, w=960, h=640):
    img = paper(w, h, seed)
    ring = T.Pen(w, h)
    ring.stroke([(82 + 58 * math.cos(a), 92 + 58 * math.sin(a)) for a in np.linspace(0.2, 2 * math.pi + 0.5, 70)], 5, 3)
    T.put(img, T.ink_tex(ring.alpha(seed=seed), seed), 0, 0, INK)
    la = serif_alpha(letter, 84, 700)
    paste(img, la, 82, 94, INK)
    sa = inky(serif_alpha(small, 46, 600, 6), seed)
    T.put(img, sa, 170, 50, INK, 0.9)
    ba = inky(serif_alpha(big, 84, 800, 1), seed + 1)
    if ba.shape[1] > w - 190:
        s = (w - 190) / ba.shape[1]; ba = cv2.resize(ba, (int(ba.shape[1] * s), int(ba.shape[0] * s)), interpolation=cv2.INTER_AREA)
    T.put(img, ba, 168, 112, INK)
    rule = T.Pen(w, h); rule.line((40, 222), (w - 40, 220), 2.0)
    T.put(img, rule.alpha(seed=seed + 2), 0, 0, INK, 0.7)
    ia, ha = drawing(w - 60, h - 250, seed)
    T.put(img, ia, 30, 236, INK, 0.95)
    return img

def place(bg, im, cx, cy, deg, lift=1.0):
    a0 = np.ones(im.shape[:2], np.float32)
    ri, ra = T.rotate_rgba(im, a0, deg)
    h, w = ra.shape; x, y = int(cx - w / 2), int(cy - h / 2)
    sh = cv2.GaussianBlur(ra, (0, 0), 22 * lift)
    T.put(bg, sh, x + int(12 * lift), y + int(22 * lift), np.array([6, 6, 8], np.float32), 0.85)
    H_, W_ = bg.shape[:2]
    X0, Y0, X1, Y1 = max(0, x), max(0, y), min(W_, x + w), min(H_, y + h)
    m = ra[Y0 - y:Y1 - y, X0 - x:X1 - x, None]
    bg[Y0:Y1, X0:X1] = bg[Y0:Y1, X0:X1] * (1 - m) + ri[Y0 - y:Y1 - y, X0 - x:X1 - x] * m

def stamp_q(seed=5, d=330):
    """round red rubber stamp with a question mark (ink texture, starved specks)."""
    a = np.zeros((d, d), np.float32)
    c = d // 2
    cv2.circle(a, (c, c), c - 10, 1.0, 12, cv2.LINE_AA)
    cv2.circle(a, (c, c), c - 32, 1.0, 4, cv2.LINE_AA)
    q = serif_alpha('?', 230, 800)
    h, w = q.shape
    a[c - h // 2:c - h // 2 + h, c - w // 2:c - w // 2 + w] = np.maximum(a[c - h // 2:c - h // 2 + h, c - w // 2:c - w // 2 + w], q)
    a = T.wobble(a, 1.6, 22, seed)
    ink = 0.55 + 0.45 * fbm(d, d, 34, seed, 3)
    holes = (fbm(d, d, 4, seed + 1, 2) > 0.30).astype(np.float32)
    return np.clip(a * ink * (0.55 + 0.45 * holes) * 1.3, 0, 1)

def desk():
    v = 0.09 + 0.07 * cv2.resize(fbm(H // 8, W, 30, 9, 4), (W, H)) + 0.05 * fbm(H, W, 90, 10, 4)
    Y, X = np.mgrid[0:H, 0:W].astype(np.float32)
    spot = np.exp(-(((X - W * 0.5) / (W * 0.8)) ** 2 + ((Y - H * 0.47) / (H * 0.42)) ** 2))
    return np.clip((v * (0.4 + 0.95 * spot))[..., None] * PAPER * 1.05, 0, 255)

def main(out):
    bg = desk()
    # title: two lines, the lie in red
    t1 = inky(serif_alpha('ΜΙΑ ΑΛΗΘΕΙΑ', 112, 700, 10), 3, 0.08)
    t2 = inky(serif_alpha('ΕΝΑ ΨΕΜΑ', 112, 700, 10), 4, 0.08)
    paste(bg, t1, W / 2, 150, LIGHT, 1.0, shadow=0.8)
    paste(bg, t2, W / 2, 285, RED, 1.0, shadow=0.8)
    A = card('Α', 'ΕΠΕΖΗΣΕ ΑΠΟ', '2 ΑΤΟΜΙΚΕΣ ΒΟΜΒΕΣ', T.draw_two_bombs, 3)
    B = card('Β', 'ΕΠΕΖΗΣΕ ΑΠΟ', '3 ΝΑΥΑΓΙΑ', T.draw_three_wrecks, 7)
    place(bg, A, W / 2 - 4, 770, 1.6)
    place(bg, B, W / 2 + 4, 1392, -1.2, lift=0.45)
    # the red question stamp, pressed across both cards
    s = stamp_q()
    _, sr = T.rotate_rgba(None, s, -12)
    reg_y, reg_x = 1110, 830
    h, w = sr.shape
    sub = bg[reg_y - h // 2:reg_y - h // 2 + h, reg_x - w // 2:reg_x - w // 2 + w]
    m = sr[:sub.shape[0], :sub.shape[1], None]
    bg[reg_y - h // 2:reg_y - h // 2 + h, reg_x - w // 2:reg_x - w // 2 + w] = sub * (1 - m * 0.95) + sub * (RED / 255.0) * 1.35 * m * 0.95
    img = Film(seed=5).apply(bg, 3, 0.0, strength=0.6, weave=False)
    cv2.imwrite(out, np.clip(img, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 94])
    print('->', out)

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else f'{ROOT}/thumb_ep05.jpg')
