"""Core engine: plates with continuous depth parallax, fog, rain, light, film look, transitions, text."""
import cv2, numpy as np, os, functools
from PIL import Image, ImageDraw, ImageFont

ROOT = os.environ.get('MT_ROOT', '/home/user/ww1')
W, H, FPS = 1080, 1920, 30
PAPER = np.array([214, 226, 236], np.float32)
INK = np.array([16, 17, 20], np.float32)
RED = np.array([38, 40, 150], np.float32)       # dried-blood ink red (BGR)
FONT = {
    'serif': f'{ROOT}/fonts/EBGaramond.ttf',
    'hand': f'{ROOT}/fonts/Caveat.ttf',
    'type': f'{ROOT}/fonts/SpecialElite.ttf',
}

def ease(x):
    x = min(1.0, max(0.0, x)); return x * x * (3 - 2 * x)

def ease2(x):
    x = min(1.0, max(0.0, x)); return 0.5 - 0.5 * np.cos(np.pi * x)

def lerp(a, b, u):
    return a + (b - a) * u

def clamp01(x):
    return min(1.0, max(0.0, x))

# ---------------------------------------------------------------- noise
def smooth_noise(h, w, scale, seed):
    r = np.random.default_rng(seed)
    sh, sw = max(2, int(h / scale) + 3), max(2, int(w / scale) + 3)
    n = r.random((sh, sw)).astype(np.float32)
    return cv2.resize(n, (int(sw * scale), int(sh * scale)), interpolation=cv2.INTER_CUBIC)[:h, :w]

def fbm(h, w, scale, seed, oct=4):
    acc = np.zeros((h, w), np.float32); amp = 1.0; tot = 0
    for o in range(oct):
        acc += amp * smooth_noise(h, w, max(2, scale / 2 ** o), seed + o * 17); tot += amp; amp *= 0.5
    return acc / tot

@functools.lru_cache(None)
def grain_bank():
    r = np.random.default_rng(7)
    return [cv2.GaussianBlur(r.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 0.7) for _ in range(12)]

@functools.lru_cache(None)
def vignette():
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt(((x - W / 2) / (W * 0.62)) ** 2 + ((y - H / 2) / (H * 0.6)) ** 2)
    return np.clip(1 - 0.55 * np.clip(d - 0.35, 0, 1) ** 1.6, 0, 1)[..., None]

@functools.lru_cache(None)
def boil_maps():
    out = []
    for s in range(3):
        bx = (fbm(H // 4, W // 4, 18, 100 + s, 2) - 0.5) * 2
        by = (fbm(H // 4, W // 4, 18, 200 + s, 2) - 0.5) * 2
        out.append((cv2.resize(bx, (W, H)), cv2.resize(by, (W, H))))
    return out

@functools.lru_cache(None)
def fog_tex(seed=3):
    # large tileable-ish fog texture
    return fbm(1400, 2400, 260, seed, 5)

@functools.lru_cache(None)
def paper_tex():
    n = fbm(H, W, 6, 51, 3)
    fib = fbm(H, W, 40, 52, 3)
    return (0.93 + 0.07 * (0.6 * n + 0.4 * fib))[..., None]

# ---------------------------------------------------------------- plates
class Plate:
    def __init__(self, key):
        self.key = key
        self.img = cv2.imread(f'{ROOT}/plates/{key}.png').astype(np.float32)
        self.ph, self.pw = self.img.shape[:2]
        d = f'{ROOT}/depth/{key}.npy'
        self.depth = np.load(d).astype(np.float32) if os.path.exists(d) else np.full((self.ph, self.pw), 0.5, np.float32)
        wm = f'{ROOT}/plates/{key}_warm.png'
        self.warm = cv2.imread(wm, 0).astype(np.float32) / 255 if os.path.exists(wm) else None
        self.blur = None

    def base_maps(self, cx, cy, zoom, rot=0.0):
        """output pixel -> plate pixel (no parallax). zoom 1 = plate width fills output width."""
        s = self.pw / W / zoom
        ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
        xs -= W / 2; ys -= H / 2
        if rot:
            c, si = np.cos(rot), np.sin(rot)
            xs, ys = xs * c - ys * si, xs * si + ys * c
        return (xs * s + cx).astype(np.float32), (ys * s + cy).astype(np.float32)

    def render(self, t, fi, cam, par=(0, 0, 0), focus=0.5, boil=1.0, dof=0.0, fog=0.0,
               fog_speed=(18, -3), haze=0.0, warm_flicker=0.0, light=1.0):
        """cam=(cx,cy,zoom[,rot]) in plate px; par=(px,py,pz): parallax shift (plate px per unit depth) & dolly."""
        cx, cy, zoom = cam[:3]; rot = cam[3] if len(cam) > 3 else 0.0
        mx, my = self.base_maps(cx, cy, zoom, rot)
        if boil:
            bx, by = boil_maps()[(fi // 3) % 3]
            mx = (mx + bx * float(1.1 * boil)).astype(np.float32); my = (my + by * float(1.1 * boil)).astype(np.float32)
        px, py, pz = par
        if px or py or pz:
            for _ in range(2):
                d = cv2.remap(self.depth, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT) - focus
                ox = d * px - d * pz * (mx - cx) / max(1.0, self.pw)
                oy = d * py - d * pz * (my - cy) / max(1.0, self.pw)
                mx2, my2 = (mx - ox).astype(np.float32), (my - oy).astype(np.float32)
            mx, my = mx2, my2
        out = cv2.remap(self.img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        if os.environ.get('BORDERCHK'):
            pm = cv2.imread(f'{ROOT}/plates/{self.key}_paper.png', 0).astype(np.float32) / 255
            v = cv2.remap(pm, mx[::8, ::8].copy(), my[::8, ::8].copy(), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=1)
            print('BORDER', self.key, round(t, 2), round(float(v.mean()), 4), flush=True)
        dmap = None
        if dof or fog or haze:
            dmap = cv2.remap(self.depth, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        if dof:
            bl = cv2.GaussianBlur(out, (0, 0), dof)
            a = np.clip(np.abs(dmap - focus) * 2.2, 0, 1)[..., None]
            out = out * (1 - a) + bl * a
        if self.warm is not None and warm_flicker:
            wm = cv2.remap(self.warm, mx, my, cv2.INTER_LINEAR)
            fl = 1 + warm_flicker * (0.18 * np.sin(t * 13.1) + 0.12 * np.sin(t * 7.3 + 1) + 0.08 * np.sin(t * 23.7))
            glow = cv2.GaussianBlur(wm, (0, 0), 25)[..., None]
            out = out * (1 + (fl - 1) * wm[..., None]) + np.array([60, 140, 255], np.float32) * glow * 0.35 * fl
        if haze:
            far = np.clip(1 - dmap, 0, 1)[..., None] ** 1.5
            out = out * (1 - haze * far) + PAPER * 0.92 * haze * far
        if fog:
            out = apply_fog(out, t, fog, dmap, fog_speed)
        if light != 1.0:
            out = out * light
        return out

def apply_fog(out, t, amt, dmap=None, speed=(18, -3), seed=3):
    tex = fog_tex(seed)
    th, tw = tex.shape
    ox = int(t * speed[0]) % (tw - W); oy = int(abs(t * speed[1])) % (th - H) if th > H else 0
    f1 = tex[oy:oy + H, ox:ox + W] if th >= H + oy else cv2.resize(tex, (W, H))
    f2 = cv2.resize(fog_tex(seed + 5), (W, H))
    sh = int(t * speed[0] * 0.55) % W
    f2 = np.roll(f2, sh, axis=1)
    f = np.clip((0.6 * f1 + 0.4 * f2 - 0.35) * 1.8, 0, 1)
    if dmap is not None:
        f = f * (0.35 + 0.65 * np.clip(1 - dmap, 0, 1))
    a = (f * amt)[..., None]
    return out * (1 - a) + PAPER * 0.95 * a

# ---------------------------------------------------------------- weather / particles
class Rain:
    def __init__(self, n=900, seed=1, angle=0.18, speed=2600, length=38, alpha=0.18):
        r = np.random.default_rng(seed)
        self.x = r.random(n) * W * 1.3 - W * 0.15; self.y0 = r.random(n) * H
        self.z = 0.4 + r.random(n) * 0.6
        self.angle, self.speed, self.length, self.alpha = angle, speed, length, alpha
    def render(self, t):
        layer = np.zeros((H, W), np.float32)
        y = (self.y0 + t * self.speed * self.z) % (H + 100) - 50
        x = self.x + y * self.angle
        L = self.length * self.z
        for xi, yi, li, zi in zip(x, y, L, self.z):
            cv2.line(layer, (int(xi), int(yi)), (int(xi + self.angle * li), int(yi + li)), float(zi), 1, cv2.LINE_AA)
        return cv2.GaussianBlur(layer, (0, 0), 0.6)
    def apply(self, img, t, amt=1.0):
        l = self.render(t)[..., None] * self.alpha * amt
        return img * (1 - l) + PAPER * l

class Motes:
    def __init__(self, n=70, seed=2):
        r = np.random.default_rng(seed)
        self.p = r.random((n, 2)) * [W, H]; self.v = (r.random((n, 2)) - 0.5) * [14, 10] + [6, 4]
        self.s = 1 + r.random(n) * 2.5; self.ph = r.random(n) * 6.28
    def apply(self, img, t, amt=1.0, dark=False):
        layer = np.zeros((H, W), np.float32)
        p = (self.p + self.v * t + np.stack([np.sin(t * 0.7 + self.ph) * 8, np.cos(t * 0.5 + self.ph) * 6], 1)) % [W, H]
        for (x, y), s, ph in zip(p, self.s, self.ph):
            cv2.circle(layer, (int(x), int(y)), int(s), 0.5 + 0.5 * np.sin(t * 2 + ph), -1, cv2.LINE_AA)
        layer = cv2.GaussianBlur(layer, (0, 0), 1.2)[..., None] * 0.5 * amt
        c = INK if dark else PAPER * 1.05
        return img * (1 - layer) + c * layer

# ---------------------------------------------------------------- film look
class Film:
    def __init__(self, seed=5):
        self.r = np.random.default_rng(seed)
        self.scr = []
    def apply(self, img, fi, t, strength=1.0, weave=True, flicker=1.0, grain=1.0):
        g = grain_bank()[fi % 12][..., None]
        fl = 1 + flicker * (0.025 * np.sin(fi * 1.7) + 0.02 * np.sin(fi * 0.53) + 0.015 * (self.r.random() - 0.5))
        out = img * fl
        lum = np.clip(out.mean(2, keepdims=True) / 255, 0, 1)
        out = out + g * (10 + 10 * (1 - np.abs(lum - 0.5) * 2)) * grain * strength
        out = out * vignette()
        if weave:
            dx = 0.8 * np.sin(fi * 0.37) + 0.5 * np.sin(fi * 1.31); dy = 0.9 * np.sin(fi * 0.29 + 1)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            out = cv2.warpAffine(out, M, (W, H), borderMode=cv2.BORDER_REFLECT)
        # dust & hair
        rr = np.random.default_rng(fi * 31 + 3)
        if strength > 0:
            for _ in range(rr.integers(0, 4)):
                x, y = rr.integers(0, W), rr.integers(0, H)
                cv2.circle(out, (int(x), int(y)), int(rr.integers(1, 4)), (30, 30, 30) if rr.random() < 0.6 else (225, 232, 238), -1, cv2.LINE_AA)
            if rr.random() < 0.04:
                x = int(rr.integers(80, W - 80))
                cv2.line(out, (x, 0), (x + int(rr.integers(-30, 30)), H), (200, 210, 215), 1, cv2.LINE_AA)
        return out

# ---------------------------------------------------------------- transitions
@functools.lru_cache(None)
def _ink_noise(seed):
    return fbm(H, W, 140, seed, 5)

def ink_mask(p, seed=11, center=(0.5, 0.5), soft=0.06):
    """organic ink-bleed reveal mask: 0 -> 1 as p goes 0 -> 1."""
    n = _ink_noise(seed)
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt(((x / W - center[0]) * 0.75) ** 2 + (y / H - center[1]) ** 2)
    d = d / d.max()
    v = 0.6 * d + 0.4 * n
    thr = lerp(-0.05, 1.05, p)
    return np.clip((thr - v) / soft, 0, 1)[..., None]

def crossfade(a, b, u):
    return a * (1 - u) + b * u

def flash(img, amt, color=(235, 240, 245)):
    return img * (1 - amt) + np.array(color, np.float32) * amt

# ---------------------------------------------------------------- text
@functools.lru_cache(None)
def font(kind, size, wght=None):
    f = ImageFont.truetype(FONT[kind], size)
    if wght is not None:
        try:
            f.set_variation_by_axes([wght])
        except Exception:
            pass
    return f

def text_layer(lines, kind='serif', size=60, wght=500, spacing=1.15, align='center', track=0):
    """returns alpha (h,w) float for multi-line text."""
    if kind in ('type', 'hand') and any('\u0370' <= ch <= '\u03ff' or '\u1f00' <= ch <= '\u1fff' for ln in lines for ch in ln):
        kind, wght = 'serif', 600
    f = font(kind, size, wght)
    widths, hs = [], []
    for ln in lines:
        bb = f.getbbox(ln)
        widths.append(bb[2] - bb[0] + track * len(ln)); hs.append(size)
    w = int(max(widths) + size); h = int(size * spacing * len(lines) + size * 0.6)
    im = Image.new('L', (w, h), 0); d = ImageDraw.Draw(im)
    y = size * 0.2
    for ln, lw in zip(lines, widths):
        x = (w - lw) / 2 if align == 'center' else size * 0.5
        if track:
            for ch in ln:
                d.text((x, y), ch, font=f, fill=255); x += f.getlength(ch) + track
        else:
            d.text((x, y), ln, font=f, fill=255)
        y += size * spacing
    return np.asarray(im).astype(np.float32) / 255

def paste_alpha(img, alpha, cx, cy, color, opacity=1.0, shadow=0.0, rough=True):
    h, w = alpha.shape
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    x1, y1 = x0 + w, y0 + h
    ax0, ay0 = max(0, -x0), max(0, -y0)
    x0c, y0c, x1c, y1c = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1c <= x0c or y1c <= y0c:
        return img
    a = alpha[ay0:ay0 + (y1c - y0c), ax0:ax0 + (x1c - x0c)]
    if rough:
        n = _ink_noise(77)[y0c:y1c, x0c:x1c]
        a = a * (0.82 + 0.18 * n)
    if shadow:
        sa = cv2.GaussianBlur(a, (0, 0), 6) * shadow
        img[y0c:y1c, x0c:x1c] *= (1 - sa[..., None] * opacity)
    a = (a * opacity)[..., None]
    img[y0c:y1c, x0c:x1c] = img[y0c:y1c, x0c:x1c] * (1 - a) + np.array(color, np.float32) * a
    return img

def band(img, y0, y1, amt=0.55, feather=60):
    m = np.zeros((H, 1), np.float32)
    for y in range(max(0, y0 - feather), min(H, y1 + feather)):
        if y < y0: v = (y - (y0 - feather)) / feather
        elif y > y1: v = ((y1 + feather) - y) / feather
        else: v = 1
        m[y] = ease(v)
    return img * (1 - amt * m[:, None, :])
