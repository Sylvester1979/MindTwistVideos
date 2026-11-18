"""Shadow-puppet theatre: a backlit linen screen, cut-out scenery taken from images,
and articulated cut-paper puppets with real walk cycles.

Everything that is black is a cut-out casting a shadow on the screen. Depth reads as
blur and density: pressed on the screen = sharp and dark, further back = softer and paler."""
import math, functools, cv2, numpy as np
from eng import W, H, fbm, ease, ease2, clamp01, lerp

# ------------------------------------------------------------------ the screen
AMBER = np.array([62, 148, 236], np.float32)     # BGR lamp-lit linen
DUSK = np.array([78, 92, 112], np.float32)       # drained, cold grey-brown
INKSH = np.array([10, 9, 12], np.float32)        # shadow colour

@functools.lru_cache(None)
def linen():
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    weave = 0.5 + 0.25 * np.sin(x * 2.1) * np.sin(y * 0.35) + 0.25 * np.sin(y * 2.3) * np.sin(x * 0.31)
    slub = cv2.resize(fbm(H // 6, W // 2, 14, 301, 3), (W, H))           # thicker threads
    blot = fbm(H, W, 220, 302, 4)                                          # uneven thickness
    t = 0.90 + 0.05 * weave + 0.05 * slub + 0.06 * (blot - 0.5)
    return t[..., None]

@functools.lru_cache(None)
def _radial(cx, cy, rx, ry):
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    d = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2
    return np.exp(-d)[..., None]

def screen(t, warmth=1.0, lamp=(0.5, 0.42), power=1.0, flicker=1.0):
    """warmth 1 = amber lamp, 0 = drained grey. lamp = hotspot position (fractions)."""
    col = DUSK + (AMBER - DUSK) * warmth
    fl = 1 + flicker * (0.035 * math.sin(t * 9.1) + 0.025 * math.sin(t * 13.7 + 1.3) + 0.02 * math.sin(t * 4.3))
    hot = _radial(int(W * lamp[0]), int(H * lamp[1]), W * 0.95, H * 0.62)
    lum = (0.30 + 0.85 * hot) * power * fl
    img = col * lum * linen()
    return np.clip(img, 0, 255)

def cast(img, alpha, blur=0.0, density=1.0, color=INKSH):
    """lay a shadow (alpha HxW float) on the screen. blur in px = distance from the screen."""
    a = alpha
    if blur > 0.3:
        a = cv2.GaussianBlur(a, (0, 0), blur)
    a = (a * density)[..., None]
    return img * (1 - a) + color * a

# ------------------------------------------------------------------ scenery from images
def cutout(path, lo=None, hi=None, size=None):
    """black regions of an image -> alpha (1 = card). Otsu threshold with a soft edge."""
    g = cv2.imread(path, 0)
    if size: g = cv2.resize(g, size, interpolation=cv2.INTER_AREA)
    if lo is None:
        thr, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        lo, hi = thr * 0.55, thr * 1.05
    a = np.clip((hi - g.astype(np.float32)) / max(1, hi - lo), 0, 1)
    return a

def place_alpha(a, cx, cy, scale, rot=0.0):
    h, w = a.shape
    c, s = math.cos(rot) * scale, math.sin(rot) * scale
    M = np.float32([[c, -s, 0], [s, c, 0]])
    M[:, 2] = np.float32([cx, cy]) - M[:, :2] @ np.float32([w / 2, h / 2])
    return cv2.warpAffine(a, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)

# ------------------------------------------------------------------ puppets
SS = 3  # supersampling for clean cut edges

def _rot(p, a):
    c, s = math.cos(a), math.sin(a); return (p[0] * c - p[1] * s, p[0] * s + p[1] * c)

def _add(a, b): return (a[0] + b[0], a[1] + b[1])

class Canvas:
    def __init__(self):
        self.m = np.zeros((H * SS, W * SS), np.uint8)
        self.holes = []
    def poly(self, pts, v=255):
        cv2.fillPoly(self.m, [np.round(np.array(pts) * SS * 4).astype(np.int32)], v, cv2.LINE_AA, shift=2)
    def circle(self, c, r, v=255, th=-1):
        cv2.circle(self.m, (int(c[0] * SS * 4), int(c[1] * SS * 4)), int(r * SS * 4), v, int(th * SS) if th > 0 else -1, cv2.LINE_AA, shift=2)
    def ellipse(self, c, ax, ang, v=255):
        cv2.ellipse(self.m, (int(c[0] * SS), int(c[1] * SS)), (int(ax[0] * SS), int(ax[1] * SS)), ang, 0, 360, v, -1, cv2.LINE_AA)
    def limb(self, a, b, w0, w1, v=255):
        dx, dy = b[0] - a[0], b[1] - a[1]; L = math.hypot(dx, dy) + 1e-6
        nx, ny = -dy / L, dx / L
        self.poly([(a[0] + nx * w0, a[1] + ny * w0), (b[0] + nx * w1, b[1] + ny * w1),
                   (b[0] - nx * w1, b[1] - ny * w1), (a[0] - nx * w0, a[1] - ny * w0)], v)
        self.circle(a, w0, v); self.circle(b, w1, v)
    def rod(self, a, b, w=1.6):
        cv2.line(self.m, (int(a[0] * SS), int(a[1] * SS)), (int(b[0] * SS), int(b[1] * SS)), 150, max(1, int(w * SS)), cv2.LINE_AA)
    def alpha(self):
        return cv2.resize(self.m, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255

def walk_pose(phase, stride=1.0):
    """angles (radians) for one leg, phase in cycles. positive = forward (towards +x)."""
    p = phase * 2 * math.pi
    thigh = math.radians(24) * stride * math.sin(p)
    swing = math.cos(p) > 0                       # leg moving forward
    knee = math.radians(38) * stride * max(0.0, math.sin(p + 0.9)) if swing else math.radians(6) * stride
    return thigh, knee

def puppet(cv, x, y, s, phase, kind='child', facing=1, hold=None, bundle=False, hair='bob', hat=None,
           stride=1.0, lean=0.0, rods=True, arm_pose=None):
    """draw one articulated puppet. (x,y) = hip on screen, s = scale (child ~1).
    hold = 'up' (raising arm to hold an adult's hand) / 'down' (adult holding a child's hand).
    returns dict of joint positions (for linking hands)."""
    f = facing
    def P(px, py): return (x + px * s * f, y + py * s)
    J = {}
    if kind == 'child':
        torso, thigh_l, shin_l, upper, lower, head_r = 92, 52, 50, 40, 36, 25
    else:  # adult (Korczak)
        torso, thigh_l, shin_l, upper, lower, head_r = 150, 82, 80, 62, 56, 30
    lean_r = math.radians(lean)
    sh = _add((0, 0), _rot((0, -torso), lean_r * f))
    shoulder = P(*sh)
    # legs (far leg first, slightly behind)
    for k, ph in ((1, phase + 0.5), (0, phase)):
        th, kn = walk_pose(ph, stride)
        hip = P(0, 0)
        knee_rel = _rot((0, thigh_l), -th)
        knee = P(*knee_rel)
        ank_rel = _add(knee_rel, _rot((0, shin_l), -th + kn))
        ankle = P(*ank_rel)
        cv.limb(hip, knee, 10 * s if kind == 'child' else 15 * s, 8 * s if kind == 'child' else 12 * s)
        cv.limb(knee, ankle, 8 * s if kind == 'child' else 12 * s, 6 * s if kind == 'child' else 10 * s)
        toe = P(ank_rel[0] + (16 if kind == 'child' else 26), ank_rel[1] + 2)
        heel = P(ank_rel[0] - 5, ank_rel[1] + 5)
        cv.poly([P(ank_rel[0] - 6, ank_rel[1] - 5), toe, P(ank_rel[0] + (16 if kind == 'child' else 26), ank_rel[1] + 6), heel])
        J['ankle%d' % k] = ankle
    # torso / clothes
    if kind == 'child':
        if hair in ('bob', 'braid'):   # dress
            cv.poly([P(-13, 6), P(13, 6), P(10, -torso + 4), P(-10, -torso + 4)])
            cv.poly([P(-12, -38), P(12, -38), P(30, 22), P(-26, 22)])
        else:                           # coat / shorts
            cv.poly([P(-14, 10), P(14, 10), P(12, -torso + 4), P(-12, -torso + 4)])
    else:  # long coat
        cv.poly([P(-22, 40), P(26, 46), P(20, -torso + 8), P(-20, -torso + 8)])
        cv.poly([P(-24, -torso * 0.45), P(24, -torso * 0.45), P(30, 52), P(-30, 50)])
    # head
    neck = P(*_rot((2, -torso - 8), lean_r * f))
    hc = P(*_rot((5, -torso - 8 - head_r), lean_r * f))
    cv.limb(shoulder, neck, 8 * s, 7 * s)
    cv.circle(hc, head_r * s)
    nose = (hc[0] + f * head_r * s * 0.95, hc[1] + head_r * s * 0.05)
    cv.poly([(hc[0] + f * head_r * 0.6 * s, hc[1] - head_r * 0.25 * s), (nose[0] + f * 5 * s, nose[1] + 3 * s),
             (hc[0] + f * head_r * 0.6 * s, hc[1] + head_r * 0.25 * s)])
    if kind == 'child':
        if hair == 'bob':
            cv.poly([(hc[0] - f * head_r * 1.15 * s, hc[1] + head_r * 0.9 * s), (hc[0] - f * head_r * 1.1 * s, hc[1] - head_r * 0.8 * s),
                     (hc[0] + f * head_r * 0.3 * s, hc[1] - head_r * 1.2 * s), (hc[0] + f * head_r * 0.7 * s, hc[1] - head_r * 0.4 * s),
                     (hc[0] - f * head_r * 0.2 * s, hc[1] + head_r * 0.7 * s)])
        elif hair == 'braid':
            b0 = (hc[0] - f * head_r * 0.9 * s, hc[1] + head_r * 0.2 * s)
            cv.limb(b0, (b0[0] - f * 8 * s, b0[1] + 34 * s), 5 * s, 3 * s)
        if hat == 'cap':
            cv.poly([(hc[0] - f * head_r * 1.05 * s, hc[1] - head_r * 0.2 * s), (hc[0] - f * head_r * 0.7 * s, hc[1] - head_r * 1.15 * s),
                     (hc[0] + f * head_r * 0.7 * s, hc[1] - head_r * 1.1 * s), (hc[0] + f * head_r * 1.55 * s, hc[1] - head_r * 0.45 * s),
                     (hc[0] + f * head_r * 0.6 * s, hc[1] - head_r * 0.45 * s)])
    else:
        # beard + round glasses cut out (light through the lenses)
        cv.poly([(hc[0] + f * head_r * 0.1 * s, hc[1] + head_r * 0.35 * s), (hc[0] + f * head_r * 0.95 * s, hc[1] + head_r * 0.45 * s),
                 (hc[0] + f * head_r * 0.55 * s, hc[1] + head_r * 1.35 * s), (hc[0] - f * head_r * 0.25 * s, hc[1] + head_r * 0.9 * s)])
        gl = (hc[0] + f * head_r * 0.55 * s, hc[1] - head_r * 0.12 * s)
        cv.circle(gl, head_r * 0.27 * s, 0)
        cv.circle(gl, head_r * 0.27 * s, 255, th=2.2 * s)
        cv.limb((gl[0] + f * head_r * 0.25 * s, gl[1]), (gl[0] + f * head_r * 0.5 * s, gl[1] + head_r * 0.05 * s), 1.2 * s, 1.2 * s)
    # arms (far arm then near arm), swinging opposite to legs
    for k, ph in ((1, phase), (0, phase + 0.5)):
        sw = math.radians(20) * stride * math.sin(ph * 2 * math.pi)
        if hold == 'up' and k == 0:
            a1, a2 = math.radians(-115 if f > 0 else -115), math.radians(25)
            a1 = math.radians(-60)  # forward and up towards the adult's hand
        elif hold == 'down' and k == 0:
            a1, a2 = math.radians(-28), math.radians(-6)
        elif arm_pose is not None and k == 0:
            a1, a2 = arm_pose
        else:
            a1, a2 = sw, math.radians(-18) * stride * (0.6 + 0.4 * math.sin(ph * 2 * math.pi))
        el_rel = _add(sh, _rot((0, upper), -a1))
        wr_rel = _add(el_rel, _rot((0, lower), -a1 - a2))
        el, wr = P(*el_rel), P(*wr_rel)
        cv.limb(shoulder, el, 7 * s if kind == 'child' else 11 * s, 6 * s if kind == 'child' else 9 * s)
        cv.limb(el, wr, 6 * s if kind == 'child' else 9 * s, 5 * s if kind == 'child' else 7 * s)
        cv.circle(wr, 6.5 * s if kind == 'child' else 9 * s)
        J['hand%d' % k] = wr
        if bundle and k == 1:
            cv.ellipse((wr[0], wr[1] + 16 * s), (15 * s, 13 * s), 10)
            cv.poly([(wr[0] - 4 * s, wr[1] + 2 * s), (wr[0] + 4 * s, wr[1] + 2 * s), (wr[0], wr[1] + 8 * s)])
    if rods:
        cv.rod(J['hand1'], (J['hand1'][0] - f * 40 * s, H + 40), 1.3 * s)
        cv.rod(shoulder, (shoulder[0] + f * 30 * s, H + 40), 1.6 * s)
    J['shoulder'] = shoulder; J['head'] = hc
    return J
