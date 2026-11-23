"""Ep05 «Μία αλήθεια, ένα ψέμα»: drawing + motion kit. Everything procedural, no AI imagery.

Coordinate systems
  canvas : the story 'scroll' (units = px at zoom 1, width 1080). Strokes, props and layers live here.
  screen : 1080x1920 output. Cam maps canvas -> screen (centre, zoom, roll).
Layers are cached rasters at S px per canvas unit (S=2) so the camera can push in without blur.
Strokes are drawn live in screen space (sharp at any zoom) with a progress parameter (pen drawing)."""
import os, sys, math, functools
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'engine'))
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
import eng
from eng import W, H, fbm, ease, ease2, lerp, clamp01, Film, vignette

ROOT = os.environ.get('MT_ROOT', '/home/claude/ep05')
F_SERIF = f'{ROOT}/fonts/EBGaramond.ttf'
F_SERIF_I = f'{ROOT}/fonts/EBGaramond-Italic.ttf'
F_MONO = '/usr/share/fonts/truetype/freefont/FreeMono.ttf'
F_MONOB = '/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf'
F_CJK = '/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc'
S = 2                                                     # layer px per canvas unit

# ---------------------------------------------------------------- palette (BGR, 0-255)
def C(r, g, b): return np.array([b, g, r], np.float32)
PAPER = C(226, 214, 196)          # warm aged paper
PAPER_D = C(205, 190, 168)        # paper in shadow / land wash
INK = C(42, 36, 34)
INK_BLUE = C(38, 58, 92)          # drafting ink
AMBER = C(236, 158, 66)           # warm light (flash, lamp, furnace)
SEA = C(118, 138, 150)            # faded grey-blue wash
VERM = C(196, 54, 40)             # hanko vermilion / lifebuoy
STAMP = C(176, 44, 40)            # the ΑΛΗΘΕΙΑ stamp
DESK = C(30, 26, 24)

# ---------------------------------------------------------------- easing
def prog(t, a, b):
    return clamp01((t - a) / (b - a)) if b > a else float(t >= a)
def eout(x):
    x = clamp01(x); return 1 - (1 - x) ** 3
def ein(x):
    x = clamp01(x); return x ** 3
def einout(x):
    x = clamp01(x); return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2
def back(x, s=1.6):
    x = clamp01(x); return 1 + (s + 1) * (x - 1) ** 3 + s * (x - 1) ** 2
def spring(x, freq=3.2, damp=5.0):
    """0->1 with a damped overshoot (for things that land)."""
    x = max(0.0, x); return 1 - math.exp(-damp * x) * math.cos(freq * 2 * math.pi * x) if x < 4 else 1.0

# ---------------------------------------------------------------- camera
class Cam:
    def __init__(self, cx, cy, z=1.0, rot=0.0):
        self.cx, self.cy, self.z, self.rot = cx, cy, z, rot
        self._uv = None
    def M(self):
        c, s = math.cos(self.rot), math.sin(self.rot)
        A = np.float32([[c, -s], [s, c]]) * self.z
        t = np.float32([W / 2, H / 2]) - A @ np.float32([self.cx, self.cy])
        return np.hstack([A, t[:, None]]).astype(np.float32)
    def pt(self, x, y):
        M = self.M(); return M[0, 0] * x + M[0, 1] * y + M[0, 2], M[1, 0] * x + M[1, 1] * y + M[1, 2]
    def pts(self, P):
        M = self.M(); P = np.asarray(P, np.float32)
        return P @ M[:, :2].T + M[:, 2]
    def uv(self):
        """canvas coords of every screen pixel (cached per camera)."""
        if self._uv is None:
            Mi = cv2.invertAffineTransform(self.M())
            X, Y = _grid()
            self._uv = (Mi[0, 0] * X + Mi[0, 1] * Y + Mi[0, 2], Mi[1, 0] * X + Mi[1, 1] * Y + Mi[1, 2])
        return self._uv

@functools.lru_cache(None)
def _grid():
    Y, X = np.mgrid[0:H, 0:W].astype(np.float32)
    return X, Y

def cam_path(t, keys):
    """keys: list of (time, cx, cy, z, rot). Eased (in-out cubic) between consecutive keys, holds outside."""
    if t <= keys[0][0]: return Cam(*keys[0][1:])
    for k0, k1 in zip(keys, keys[1:]):
        if t <= k1[0]:
            u = einout((t - k0[0]) / max(1e-6, k1[0] - k0[0]))
            return Cam(*[lerp(a, b, u) for a, b in zip(k0[1:], k1[1:])])
    return Cam(*keys[-1][1:])

# ---------------------------------------------------------------- compositing
def paint(img, a, color, op=1.0, mode='normal'):
    if op <= 0: return img
    a = (a * op)[..., None] if a.ndim == 2 else a * op
    if mode == 'normal':
        img[:] = img * (1 - a) + color * a
    elif mode == 'multiply':
        img[:] = img * (1 - a + a * (np.asarray(color, np.float32) / 255.0))
    elif mode == 'add':
        img[:] = img + a * color
    elif mode == 'screen':
        img[:] = 255 - (255 - img) * (1 - a * np.asarray(color, np.float32) / 255.0)
    return img

def warp(src, M, interp=cv2.INTER_LINEAR):
    return cv2.warpAffine(src, M, (W, H), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

def layer_M(cam, x0, y0, scale=1.0, rot=0.0, anchor=(0, 0), s=S):
    """layer raster (s px/unit) placed so its local point `anchor` (units) sits at canvas (x0,y0), rotated/scaled about it.
    scale may be (sx, sy) for flips."""
    c, si = math.cos(rot), math.sin(rot)
    sx, sy = (scale, scale) if np.isscalar(scale) else scale
    L = np.float32([[c * sx / s, -si * sy / s, 0], [si * sx / s, c * sy / s, 0]])
    L[:, 2] = np.float32([x0, y0]) - L[:, :2] @ np.float32([anchor[0] * s, anchor[1] * s])
    Mc = cam.M()
    A = Mc[:, :2] @ L[:, :2]; t = Mc[:, :2] @ L[:, 2] + Mc[:, 2]
    return np.hstack([A, t[:, None]]).astype(np.float32)

def blit(img, alpha, cam, x0, y0, color, op=1.0, scale=1.0, rot=0.0, anchor=(0, 0), mode='normal', s=S):
    if op <= 0: return img
    a = warp(alpha, layer_M(cam, x0, y0, scale, rot, anchor, s))
    return paint(img, a, color, op, mode)

def blit_rgba(img, rgb, alpha, cam, x0, y0, op=1.0, scale=1.0, rot=0.0, anchor=(0, 0), shadow=0.0, lift=1.0, s=S):
    """prop with its own colours and a soft drop shadow (lift = height above paper)."""
    if op <= 0: return img
    M = layer_M(cam, x0, y0, scale, rot, anchor, s)
    a = warp(alpha, M)
    if shadow > 0:
        z = cam.z * (scale if np.isscalar(scale) else scale[0])
        off = (6 + 10 * lift) * z
        sh = cv2.GaussianBlur(a, (0, 0), (4 + 10 * lift) * z + 0.5)
        sh = np.roll(np.roll(sh, int(off * 0.6), 1), int(off), 0)
        img[:] = img * (1 - (sh * shadow * op)[..., None])
    c = warp(rgb, M)
    img[:] = img * (1 - (a * op)[..., None]) + c * (a * op)[..., None]
    return img

# ---------------------------------------------------------------- noise helpers in screen space
@functools.lru_cache(None)
def ink_grain():
    return (0.80 + 0.20 * fbm(H, W, 5, 77, 2)).astype(np.float32)

def inkify(a):
    """pen ink: slight bleed + uneven density."""
    return np.clip(cv2.GaussianBlur(a, (0, 0), 0.55) * ink_grain() * 1.08, 0, 1)

# ---------------------------------------------------------------- strokes (live, screen space)
def resample(P, step=3.0):
    P = np.asarray(P, np.float64)
    d = np.hypot(*np.diff(P, axis=0).T)
    L = np.concatenate([[0], np.cumsum(d)])
    if L[-1] < 1e-6: return P
    n = max(2, int(L[-1] / step) + 1)
    s = np.linspace(0, L[-1], n)
    return np.stack([np.interp(s, L, P[:, 0]), np.interp(s, L, P[:, 1])], 1)

def hand(P, amp=1.0, seed=0, step=3.0, freq=0.02):
    """hand-drawn wobble: low-frequency perpendicular drift along the stroke."""
    P = resample(P, step)
    if amp <= 0 or len(P) < 3: return P
    r = np.random.default_rng(seed)
    n = len(P)
    k = max(4, int(n * step * freq) + 4)
    ctrl = r.normal(0, 1, k)
    off = np.interp(np.linspace(0, k - 1, n), np.arange(k), ctrl) * amp
    T = np.gradient(P, axis=0); T /= (np.hypot(*T.T)[:, None] + 1e-9)
    N = np.stack([-T[:, 1], T[:, 0]], 1)
    return P + N * off[:, None]

class Stroke:
    def __init__(self, P, w=2.2, amp=1.0, seed=0, closed=False, taper=True):
        P = np.asarray(P, np.float64)
        if closed: P = np.vstack([P, P[:1]])
        self.P = hand(P, amp, seed)
        d = np.hypot(*np.diff(self.P, axis=0).T)
        self.L = np.concatenate([[0], np.cumsum(d)])
        self.len = self.L[-1]
        self.w = w; self.taper = taper
        r = np.random.default_rng(seed + 991)
        n = len(self.P)
        k = max(3, n // 25 + 3)
        self.press = np.interp(np.linspace(0, k - 1, n), np.arange(k), 0.75 + 0.5 * r.random(k))

def draw_strokes(buf, strokes, p, cam, tip=False, width_scale=1.0):
    """draw strokes into screen alpha buf; p in [0,1] = fraction of the *total* length drawn (in order)."""
    total = sum(s.len for s in strokes) or 1
    budget = p * total
    for s in strokes:
        if budget <= 0: break
        upto = min(s.len, budget); budget -= s.len
        n = int(np.searchsorted(s.L, upto)) + 1
        if n < 2: continue
        Q = cam.pts(s.P[:n].astype(np.float32))
        wpx = s.w * cam.z * width_scale
        if wpx < 1.6:
            lay = np.zeros_like(buf)
            cv2.polylines(lay, [np.round(Q * 8).astype(np.int32)], False, 1.0, 1, cv2.LINE_AA, shift=3)
            np.maximum(buf, lay * min(1.0, wpx / 1.2), out=buf)
        else:
            # pressure-varying width: draw in a few chunks
            chunks = max(1, n // 24)
            idx = np.linspace(0, n - 1, chunks + 1).astype(int)
            for i0, i1 in zip(idx[:-1], idx[1:]):
                if i1 <= i0: continue
                pr = s.press[i0:i1 + 1].mean()
                if s.taper:
                    f0 = s.L[i0] / max(1e-6, s.len)
                    pr *= 0.55 + 0.45 * min(1, f0 * 8, (1 - f0) * 8 + 0.2)
                th = max(1, int(round(wpx * pr)))
                cv2.polylines(buf, [np.round(Q[i0:i1 + 1] * 8).astype(np.int32)], False, 1.0, th, cv2.LINE_AA, shift=3)
        if tip and upto < s.len:
            x, y = Q[-1]
            cv2.circle(buf, (int(x * 8), int(y * 8)), max(1, int(wpx * 0.9 * 8)), 1.0, -1, cv2.LINE_AA, shift=3)
    return buf

def strokes_len(strokes):
    return sum(s.len for s in strokes)

# ---------------------------------------------------------------- shapes as layers (cached rasters)
class Raster:
    """supersampled mask builder in canvas units, output at S px/unit, origin (x0,y0)."""
    def __init__(self, x0, y0, w, h, ss=2):
        self.x0, self.y0, self.w, self.h, self.ss = x0, y0, w, h, ss
        self.k = S * ss
        self.m = np.zeros((int(h * self.k), int(w * self.k)), np.uint8)
    def _p(self, P):
        P = (np.asarray(P, np.float64) - [self.x0, self.y0]) * self.k * 16
        return [np.round(P).astype(np.int32)]
    def poly(self, P, v=255):
        cv2.fillPoly(self.m, self._p(P), v, cv2.LINE_AA, shift=4); return self
    def ell(self, c, ax, ang=0, v=255):
        cx, cy = (c[0] - self.x0) * self.k, (c[1] - self.y0) * self.k
        cv2.ellipse(self.m, (int(cx), int(cy)), (max(1, int(ax[0] * self.k)), max(1, int(ax[1] * self.k))), ang, 0, 360, v, -1, cv2.LINE_AA)
        return self
    def line(self, P, w, v=255):
        cv2.polylines(self.m, self._p(P), False, v, max(1, int(w * self.k)), cv2.LINE_AA, shift=4); return self
    def alpha(self, wob=0.0, seed=0):
        a = cv2.resize(self.m, (int(self.w * S), int(self.h * S)), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        if wob:
            h, w = a.shape
            dx = (fbm(h, w, 30, seed, 2) - 0.5) * 2 * wob * S
            dy = (fbm(h, w, 30, seed + 7, 2) - 0.5) * 2 * wob * S
            Y, X = np.mgrid[0:h, 0:w].astype(np.float32)
            a = cv2.remap(a, X + dx, Y + dy, cv2.INTER_LINEAR)
        return a

def hatch_uv(cam, ang, spacing, width, seed=0, jitter=0.25):
    """parallel pen lines in *canvas* coordinates (they move with the paper)."""
    U, V = cam.uv()
    c, s = math.cos(ang), math.sin(ang)
    u = U * c + V * s
    if jitter:
        n = _jit_field(seed)
        u = u + n * spacing * jitter
    d = np.abs(((u / spacing) % 1.0) - 0.5) * spacing
    px = 1.0 / cam.z
    return np.clip((width / 2 - d) / px + 0.5, 0, 1)

@functools.lru_cache(8)
def _jit_field(seed):
    return (fbm(H, W, 90, seed, 2) - 0.5) * 2

def hatch_layer(h, w, ang, spacing, width, seed=0):
    """hatch pattern as raster in layer px (S/unit)."""
    Y, X = np.mgrid[0:h, 0:w].astype(np.float32) / S
    c, s = math.cos(ang), math.sin(ang)
    u = X * c + Y * s + (fbm(h, w, 60, seed, 2) - 0.5) * spacing * 0.5
    d = np.abs(((u / spacing) % 1.0) - 0.5) * spacing
    return np.clip((width / 2 - d) * S + 0.5, 0, 1)

def wash(mask, seed=0, edge=0.35, gran=0.25):
    """watercolour wash from a mask (layer px): soft edges, darker pooled rim, granulation."""
    a = cv2.GaussianBlur(mask, (0, 0), 2.2 * S)
    h, w = a.shape
    g = fbm(h, w, 18 * S, seed, 3)
    rim = np.clip(mask - cv2.GaussianBlur(mask, (0, 0), 6 * S), 0, 1)
    return np.clip(a * (1 - gran + gran * g * 1.6) + rim * edge, 0, 1)

# ---------------------------------------------------------------- text
@functools.lru_cache(None)
def _font(path, size, wght=None, index=0):
    f = ImageFont.truetype(path, int(size), index=index)
    if wght is not None:
        try: f.set_variation_by_axes([wght])
        except Exception: pass
    return f

class Text:
    """text raster (S px/unit) + per-character x boundaries (for typing). size in canvas units."""
    def __init__(self, txt, path=F_MONO, size=40, wght=None, track=0.0, typed=True, seed=0, index=0):
        f = _font(path, size * S, wght, index)
        self.txt = txt
        adv = [f.getlength(ch) + track * S for ch in txt]
        w = int(sum(adv) + size * S * 0.6); h = int(size * S * 1.6)
        im = Image.new('L', (w, h), 0); d = ImageDraw.Draw(im)
        x = size * S * 0.2; self.xs = [x]
        asc = f.getmetrics()[0]
        y = int(size * S * 0.25)
        for ch, a in zip(txt, adv):
            d.text((x, y), ch, font=f, fill=255); x += a; self.xs.append(x)
        a = np.asarray(im).astype(np.float32) / 255
        if typed:
            hh, ww = a.shape
            pr = 0.70 + 0.30 * fbm(hh, ww, 9 * S, seed, 2)
            st = (fbm(hh, ww, 2 * S, seed + 5, 1) > 0.2).astype(np.float32) * 0.22 + 0.78
            a = np.clip(cv2.GaussianBlur(a, (0, 0), 0.45 * S) * pr * st * 1.12, 0, 1)
        self.a = a
        self.w = w / S; self.h = h / S
        self.base = (y + asc) / S                # baseline in units from top
        self.ink_w = (x - size * S * 0.2) / S    # advance width in units
    def upto(self, n):
        """alpha with only the first n characters (typewriter)."""
        if n >= len(self.txt): return self.a
        if n <= 0: return np.zeros_like(self.a)
        out = self.a.copy(); out[:, int(self.xs[n]):] = 0
        return out

def text_at(img, cam, T, x, y, color, op=1.0, n=None, align='left', rot=0.0, scale=1.0):
    """place Text so that its baseline-left (or centre) is at canvas (x,y)."""
    a = T.a if n is None else T.upto(n)
    ax = T.ink_w / 2 + 0.2 * (T.h / 1.6) if align == 'center' else 0.2 * (T.h / 1.6)
    return blit(img, a, cam, x, y, color, op, scale=scale, rot=rot, anchor=(ax, T.base))

def type_n(T, t, t0, cps=22.0):
    return int(max(0, (t - t0) * cps))

# ---------------------------------------------------------------- paper
class PaperStrip:
    """long sheet of paper in canvas space (1 px/unit), warped by the camera."""
    def __init__(self, height, seed=1, base=PAPER):
        h, w = int(height), W
        tex = np.ones((h, w, 3), np.float32) * base
        tex *= (0.92 + 0.08 * fbm(h, w, 260, seed, 4))[..., None]
        tex *= (0.95 + 0.05 * fbm(h, w, 14, seed + 1, 3))[..., None]
        st = np.clip((fbm(h, w, 480, seed + 2, 3) - 0.56) * 4, 0, 1)
        tex = tex * (1 - 0.08 * st[..., None]) + C(170, 140, 100) * 0.08 * st[..., None]
        self.tex = np.clip(tex, 0, 255).astype(np.uint8)
    def render(self, cam):
        out = cv2.warpAffine(self.tex, cam.M(), (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return out.astype(np.float32)

# ---------------------------------------------------------------- props
def card_paper(w, h, seed, base=PAPER, edge=0.16):
    """paper prop texture at S px/unit: returns rgb, alpha."""
    hh, ww = int(h * S), int(w * S)
    img = np.ones((hh, ww, 3), np.float32) * base
    img *= (0.91 + 0.09 * fbm(hh, ww, 200 * S / 2, seed, 4))[..., None]
    img *= (0.95 + 0.05 * fbm(hh, ww, 12 * S / 2, seed + 1, 3))[..., None]
    Y, X = np.mgrid[0:hh, 0:ww].astype(np.float32)
    e = np.minimum.reduce([X, Y, ww - 1 - X, hh - 1 - Y]) / S
    img *= (1 - edge * np.exp(-e / 22))[..., None]
    # slightly irregular edge
    n = fbm(hh, ww, 6 * S, seed + 3, 2)
    a = np.clip((e - 0.6 * n) * 1.5, 0, 1).astype(np.float32)
    return img, a

def put_alpha(dst, a, x, y, color, op=1.0, mode='normal'):
    """paste alpha into a local raster (S px) at local units (x,y)."""
    x, y = int(x * S), int(y * S)
    h, w = a.shape; H_, W_ = dst.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W_, x + w), min(H_, y + h)
    if x1 <= x0 or y1 <= y0: return dst
    m = a[y0 - y:y1 - y, x0 - x:x1 - x, None] * op
    reg = dst[y0:y1, x0:x1]
    tgt = reg * (np.asarray(color, np.float32) / 255) if mode == 'multiply' else color
    dst[y0:y1, x0:x1] = reg * (1 - m) + tgt * m
    return dst

def put_text_local(dst, T, x, y, color, op=1.0, n=None, align='left'):
    a = T.a if n is None else T.upto(n)
    ax = T.ink_w / 2 + 0.2 * (T.h / 1.6) if align == 'center' else 0.2 * (T.h / 1.6)
    return put_alpha(dst, a, x - ax, y - T.base, color, op)

# ---------------------------------------------------------------- light
@functools.lru_cache(maxsize=24)
def radial(cx, cy, r):
    """gaussian light blob in screen space (bounded cache: positions change every frame when the camera moves)."""
    r = max(1, r)
    x0, x1 = max(0, int(cx - 3 * r)), min(W, int(cx + 3 * r) + 1)
    y0, y1 = max(0, int(cy - 3 * r)), min(H, int(cy + 3 * r) + 1)
    out = np.zeros((H, W), np.float32)
    if x1 > x0 and y1 > y0:
        Y, X = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        d2 = ((X - cx) ** 2 + (Y - cy) ** 2) / (r * r)
        out[y0:y1, x0:x1] = np.exp(-d2)
    return out

def grade(img, t, fi, film, warmth=0.0, vig=1.0, grain=0.85):
    """final look: slight warm split, vignette, film."""
    out = img
    if warmth:
        out = out * (1 + warmth * np.float32([-0.04, 0.0, 0.05]))
    out = film.apply(out, fi, t, strength=grain, weave=True)
    return np.clip(out, 0, 255).astype(np.uint8)

def bez(p0, p1, p2, n=24):
    """quadratic bezier as a list of points."""
    t = np.linspace(0, 1, n)[:, None]
    return ((1 - t) ** 2 * np.array(p0, np.float64) + 2 * (1 - t) * t * np.array(p1, np.float64) + t ** 2 * np.array(p2, np.float64)).tolist()
