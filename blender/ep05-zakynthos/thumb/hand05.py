"""Handwriting in code for the ep05 thumbnail: the two names in a 1940s Greek cursive, written with a fountain pen.

Glyphs are pen paths: points that a centripetal Catmull-Rom spline runs through (N = a sharp turn, the pen stays down).
Each lowercase glyph says how it is entered and left, so the hand joins letters the way a Greek hand does
(an upstroke from the baseline into ι υ ν κ μ η τ ω χ, a level stroke at the x-height out of ο σ υ ν ω) and lifts
the pen where a real hand lifts it (after ρ η χ θ ς and capitals, before α ο σ ε ς from the baseline).

The hand is not a machine (the user, 27/11: «ανθρώπινο, όχι τέλειο γραφικά»): every instance of a letter is drawn a
little differently, words lean and sit differently, letters vary in size, the line wanders, the pen trembles faintly.
The pen: a medium, slightly flexible nib held at ~40 degrees, swept along the path as an ellipse; pressure opens the
tines on downstrokes. The flow is never even: each stroke its own ink load, swelling and thinning along the way, some
starts and ends leave a little pool, some fast strokes run dry (ink only on the fibre tops), and a few sprays of
droplets and one drop land on the paper. The ink: aged iron-gall, brown-black and translucent (darker where strokes
cross), ragged a little at the edges, creeping into the fibres, drying darker at the rims. Seeded, so it repeats.
Units: x-height = 1, baseline y = 0, y up. Capitals ~1.7, ascender 1.75, descender -0.72."""
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

N = None
TL, TR, LOW, HIGH = 'TL', 'TR', 'low', 'high'


def g(adv, strokes, entry=None, exit=None):
    return dict(adv=adv, strokes=strokes, entry=entry, exit=exit)


G = {
    # ---- lowercase
    'ο': g(0.80, [[(0.56, 0.93), (0.32, 1.0), (0.09, 0.78), (0.04, 0.36), (0.22, 0.02), (0.48, 0.02), (0.64, 0.32),
                   (0.63, 0.76), (0.46, 0.98), (0.58, 0.96), (0.80, 0.93)]], TR, HIGH),
    'α': g(0.92, [[(0.62, 0.86), (0.42, 1.0), (0.14, 0.84), (0.03, 0.42), (0.16, 0.04), (0.40, 0.02), (0.57, 0.32),
                   (0.66, 0.98), N, (0.66, 0.98), (0.64, 0.40), (0.68, 0.08), (0.80, 0.0), (0.94, 0.10)]], TR, LOW),
    'σ': g(0.96, [[(0.56, 0.96), (0.30, 0.98), (0.08, 0.76), (0.04, 0.36), (0.22, 0.03), (0.46, 0.0), (0.65, 0.24),
                   (0.67, 0.62), (0.52, 0.94), N, (0.52, 0.94), (0.72, 1.0), (0.96, 1.0)]], TR, HIGH),
    'ς': g(0.74, [[(0.66, 0.90), (0.44, 1.0), (0.16, 0.84), (0.06, 0.48), (0.24, 0.18), (0.48, 0.06), (0.56, -0.20),
                   (0.44, -0.50), (0.32, -0.46)]], TR, None),
    'ε': g(0.74, [[(0.62, 0.84), (0.44, 0.99), (0.18, 0.90), (0.16, 0.66), (0.38, 0.53), (0.20, 0.48), (0.06, 0.28),
                   (0.18, 0.03), (0.44, 0.0), (0.66, 0.14)]], TR, LOW),
    'ι': g(0.46, [[(0.10, 1.0), (0.10, 0.40), (0.16, 0.06), (0.30, 0.0), (0.46, 0.14)]], TL, LOW),
    'υ': g(0.88, [[(0.02, 0.86), (0.12, 1.0), (0.13, 0.60), (0.20, 0.18), (0.40, 0.0), (0.64, 0.10), (0.78, 0.48),
                   (0.80, 0.98)]], TL, HIGH),
    'ν': g(0.88, [[(0.02, 0.86), (0.10, 1.0), (0.18, 0.60), (0.28, 0.04), N, (0.28, 0.04), (0.46, 0.42), (0.64, 0.86),
                   (0.78, 1.0), (0.90, 0.98)]], TL, HIGH),
    'κ': g(0.90, [[(0.10, 1.0), (0.09, 0.50), (0.08, 0.0)],
                  [(0.62, 0.98), (0.40, 0.72), (0.12, 0.50), N, (0.12, 0.50), (0.34, 0.40), (0.52, 0.14), (0.66, 0.0),
                   (0.82, 0.08)]], TL, LOW),
    'τ': g(0.80, [[(0.0, 0.93), (0.38, 0.97), (0.78, 1.0)],
                  [(0.40, 0.98), (0.38, 0.40), (0.44, 0.06), (0.58, 0.0), (0.74, 0.10)]], TL, LOW),
    'μ': g(0.98, [[(0.10, 1.0), (0.08, 0.20), (0.05, -0.72)],
                  [(0.10, 0.34), (0.22, 0.04), (0.40, 0.0), (0.58, 0.16), (0.66, 0.55), (0.68, 1.0), N, (0.68, 1.0),
                   (0.66, 0.40), (0.70, 0.06), (0.84, 0.0), (0.96, 0.10)]], TL, LOW),
    'η': g(0.82, [[(0.02, 0.86), (0.10, 1.0), (0.12, 0.5), (0.10, 0.0), N, (0.10, 0.0), (0.15, 0.55), (0.32, 0.94),
                   (0.54, 0.96), (0.64, 0.58), (0.64, 0.0), (0.62, -0.72)]], TL, None),
    'ω': g(1.06, [[(0.12, 0.94), (0.03, 0.50), (0.10, 0.10), (0.28, 0.0), (0.44, 0.20), (0.49, 0.60), N, (0.49, 0.60),
                   (0.53, 0.20), (0.68, 0.0), (0.88, 0.12), (0.96, 0.55), (0.88, 0.96)]], TL, HIGH),
    'θ': g(0.88, [[(0.50, 1.76), (0.22, 1.52), (0.10, 0.90), (0.18, 0.16), (0.44, 0.0), (0.68, 0.18), (0.78, 0.86),
                   (0.68, 1.54), (0.48, 1.78), (0.38, 1.70)],
                  [(0.10, 0.88), (0.46, 0.91), (0.84, 0.95)]], None, None),
    'ρ': g(0.80, [[(0.12, -0.72), (0.10, -0.10), (0.12, 0.50), (0.26, 0.90), (0.46, 1.0), (0.64, 0.82), (0.68, 0.38),
                   (0.50, 0.04), (0.28, 0.04), (0.12, 0.30)]], None, None),
    'χ': g(0.88, [[(0.0, 0.92), (0.14, 0.96), (0.48, 0.14), (0.70, -0.64), (0.86, -0.72)],
                  [(0.78, 0.98), (0.44, 0.16), (0.06, -0.72)]], None, None),
    # ---- capitals (~1.7): written, not drawn: curved strokes, entry hooks, overshoots
    'Ο': g(1.14, [[(0.74, 1.56), (0.46, 1.72), (0.15, 1.48), (0.01, 0.88), (0.12, 0.22), (0.44, 0.0), (0.84, 0.18),
                   (0.99, 0.84), (0.90, 1.44), (0.62, 1.70), (0.40, 1.66), (0.30, 1.58)]]),
    'Ζ': g(1.10, [[(0.02, 1.46), (0.14, 1.62), (0.46, 1.63), (0.86, 1.67), N, (0.86, 1.67), (0.54, 1.04), (0.22, 0.38),
                   (0.06, 0.04), N, (0.06, 0.04), (0.30, 0.10), (0.62, 0.0), (0.90, 0.04), (1.04, 0.20)]]),
    'Χ': g(1.10, [[(0.0, 1.60), (0.14, 1.70), (0.36, 1.30), (0.64, 0.56), (0.90, 0.04), (1.04, 0.06)],
                  [(1.00, 1.70), (0.66, 1.10), (0.30, 0.46), (0.04, 0.0)]]),
    'Δ': g(1.10, [[(0.62, 1.70), (0.42, 1.24), (0.20, 0.56), (0.04, 0.06), N, (0.04, 0.06), (0.40, 0.03), (0.98, 0.07), N,
                   (0.98, 0.07), (0.82, 0.70), (0.68, 1.46), (0.62, 1.74), (0.74, 1.82)]]),
    'Λ': g(1.10, [[(0.0, 0.04), (0.22, 0.60), (0.42, 1.36), (0.56, 1.72), N, (0.56, 1.72), (0.70, 1.10), (0.86, 0.40),
                   (1.0, 0.02), (1.10, 0.08)]]),
    'Κ': g(1.06, [[(0.02, 1.56), (0.12, 1.72), (0.12, 0.90), (0.08, 0.0)],
                  [(0.98, 1.66), (0.80, 1.50), (0.40, 1.06), (0.16, 0.86), N, (0.16, 0.86), (0.44, 0.72), (0.70, 0.20),
                   (0.86, 0.02), (1.02, 0.06)]]),
}
ACCENTED = {'ύ': 'υ', 'ό': 'ο', 'ή': 'η', 'ά': 'α', 'ί': 'ι', 'έ': 'ε', 'ώ': 'ω'}
HOOKED = set('υνη')
ACC_X = {'υ': 0.50, 'ο': 0.40, 'η': 0.30, 'α': 0.36, 'ι': 0.14, 'ε': 0.40, 'ω': 0.50}   # where the tonos sits


def joins(prev, nxt):
    if prev is None or nxt['entry'] is None or prev['exit'] is None: return False
    if prev['exit'] == HIGH: return True
    return nxt['entry'] == TL                      # from the baseline only into letters that start top-left


# ------------------------------------------------------------------ geometry
def cr_centripetal(pts, step):
    """centripetal Catmull-Rom through pts, sampled every ~step (units)."""
    P = [np.array(p, float) for p in pts]
    if len(P) == 2:
        n = max(2, int(np.linalg.norm(P[1] - P[0]) / step) + 1)
        return [P[0] + (P[1] - P[0]) * t for t in np.linspace(0, 1, n)]
    P = [2 * P[0] - P[1]] + P + [2 * P[-1] - P[-2]]
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        t0 = 0.0
        t1 = t0 + max(np.linalg.norm(p1 - p0), 1e-6) ** 0.5
        t2 = t1 + max(np.linalg.norm(p2 - p1), 1e-6) ** 0.5
        t3 = t2 + max(np.linalg.norm(p3 - p2), 1e-6) ** 0.5
        n = max(2, int(np.linalg.norm(p2 - p1) / step) + 1)
        for t in np.linspace(t1, t2, n, endpoint=False):
            a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
            a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
            a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
            b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
            b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
            out.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
    out.append(P[-2])
    return out


def dense(stroke, step):
    """a pen-down list with N corners -> dense (n, 2) path."""
    segs, cur = [], []
    for q in stroke:
        if q is None:
            segs.append(cur); cur = []
        else:
            cur.append(q)
    segs.append(cur)
    path = []
    for s in segs:
        s = [q for i, q in enumerate(s) if i == 0 or np.hypot(q[0] - s[i - 1][0], q[1] - s[i - 1][1]) > 1e-4]
        if len(s) >= 2: path += cr_centripetal(s, step)
    return np.array(path)


def smooth_noise(n, corr, rng):
    """1D smooth noise, unit variance-ish, correlation length ~corr samples."""
    k = max(1, int(corr))
    m = n // k + 4
    base = rng.normal(0, 1, m)
    x = np.arange(n) / k
    i = np.floor(x).astype(int); f = x - i; f = f * f * (3 - 2 * f)
    return base[i + 1] * (1 - f) + base[i + 2] * f


def jitter_glyph(strokes, rng, amp):
    """every instance of a letter is drawn a little differently: smooth, correlated nudges of its points."""
    out = []
    for st in strokes:
        pts = [q for q in st]
        idx = [i for i, q in enumerate(pts) if q is not None]
        d = rng.normal(0, amp, (len(idx), 2))
        if len(idx) > 2:                                         # neighbours move together (no zig-zags)
            d2 = d.copy()
            d2[1:-1] = 0.25 * d[:-2] + 0.5 * d[1:-1] + 0.25 * d[2:]
            d = d2
        q2 = list(pts)
        for k, i in enumerate(idx):
            q2[i] = (pts[i][0] + d[k, 0], pts[i][1] + d[k, 1])
        # corners: the copy after N must stay on its twin
        for i in range(1, len(q2) - 1):
            if q2[i] is None: q2[i + 1] = q2[i - 1]
        out.append(q2)
    return out


def layout(lines, seed, line_gap=2.6, slant_deg=12.0, indents=None):
    """-> list of (path (n, 2) in units, y up) pen-downs. A human hand: every letter instance differs, words lean and
    sit a little differently, letters vary in size, the line wanders, the pen trembles very slightly."""
    rng = np.random.default_rng(seed)
    downs = []; meta = []                     # meta: (baseline y, slant) of each pen-down
    for li, line in enumerate(lines):
        if not line: continue
        y0 = -li * line_gap
        x = (indents[li] if indents else 0.0) + rng.normal(0, 0.06)
        drift = rng.normal(0, 0.04); climb = rng.normal(0, 0.018); bow = rng.normal(0, 0.006)
        for word in line.split(' '):
            cur = None; prev = None; marks = []
            w_slant = math.tan(math.radians(slant_deg + rng.normal(0, 2.2)))
            w_size = 1.0 + rng.normal(0, 0.035)
            w_dy = rng.normal(0, 0.035)

            def push(d):
                downs.append(d); meta.append((y0, w_slant))
            for ci, ch in enumerate(word):
                base = ACCENTED.get(ch, ch); gl = G[base]
                sx = w_size * (1.0 + rng.normal(0, 0.045)); sy = w_size * (1.0 + rng.normal(0, 0.055))
                if ci == len(word) - 1: sx *= 1.03                 # the hand loosens at the end of a word
                dy = y0 + drift + climb * x + bow * x * x + w_dy + rng.normal(0, 0.03)
                lean = rng.normal(0, 0.04)
                if prev is not None and joins(prev, gl): x -= 0.04 + rng.normal(0, 0.02)
                x0 = x
                strokes = jitter_glyph(gl['strokes'], rng, 0.022 if base.islower() else 0.03)

                def tf(q, sx=sx, sy=sy, dy=dy, x0=x0, lean=lean):
                    yy = q[1] * sy
                    return (x0 + q[0] * sx + yy * lean, yy + dy)
                for si, st in enumerate(strokes):
                    pts = [None if q is None else tf(q) for q in st]
                    if si == 0 and cur is not None and joins(prev, gl):
                        if base in HOOKED: pts = pts[1:]           # the join itself is the lead-in
                        cur = cur + pts
                    else:
                        if cur is not None: push(cur)
                        cur = pts
                if ch in ACCENTED:                                 # the tonos, put on after the word, never twice alike
                    ax = x0 + (ACC_X[base] + rng.normal(0, 0.05)) * sx
                    L = 0.34 + abs(rng.normal(0, 0.07)); ang = math.radians(62 + rng.normal(0, 9))
                    top = dy + 1.20 + rng.normal(0, 0.06)
                    bend = rng.normal(0, 0.03)
                    p0 = (ax + L * math.cos(ang) * 0.5, top + L * math.sin(ang))
                    p2 = (ax - L * math.cos(ang) * 0.5, top)
                    marks.append([p0, ((p0[0] + p2[0]) / 2 + bend, (p0[1] + p2[1]) / 2), p2])
                x += gl['adv'] * sx + 0.03 + rng.normal(0, 0.03)
                prev = gl
                if gl['exit'] is None and cur is not None:
                    push(cur); cur = None
            if cur is not None: push(cur)
            for m in marks: push(m)
            x += 0.50 + rng.normal(0, 0.07)
    out = []
    for d, (yb, sl) in zip(downs, meta):
        p = dense(d, 0.01)
        if len(p) < 2: continue
        p[:, 0] += (p[:, 1] - yb) * sl
        # a faint tremor across the path (a real hand is never a spline)
        t = np.gradient(p, axis=0); t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
        nrm = np.column_stack([-t[:, 1], t[:, 0]])
        trem = 0.0045 * smooth_noise(len(p), 22, rng) + 0.0025 * smooth_noise(len(p), 7, rng)
        p += nrm * trem[:, None]
        out.append(p)
    return out


# ------------------------------------------------------------------ the pen
def nib_ellipse(cx, cy, a, b, phi, n=18):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = a * np.cos(t); y = b * np.sin(t)
    c, s = math.cos(phi), math.sin(phi)
    return list(zip(cx + x * c - y * s, cy + x * s + y * c))


def blob(cx, cy, r, rng, n=40, rough=0.12):
    """an irregular round drop outline."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rr = r * (1 + rough * smooth_noise(n + 8, 5, rng)[:n] * 0.6)
    return list(zip(cx + rr * np.cos(t), cy + rr * np.sin(t)))


def write(lines, xh_px, shape, origin, seed=1943, ss=4, **kw):
    """-> D (ink optical density; ~1 = one ordinary stroke), PR (nib pressure, for the dent), DRY (where the pen ran
    dry: ink only on the fibre tops). The flow is never even: each stroke carries its own load, it swells and thins
    along the way, some starts leave a blob, some fast strokes skip, and a few drops fly off the nib."""
    H, W = shape
    rng = np.random.default_rng(seed + 7)
    downs = layout(lines, seed, **kw)
    D = np.zeros((H, W), np.float32); PR = np.zeros((H, W), np.float32); DRY = np.zeros((H, W), np.float32)
    A_ = 0.040 * xh_px; B_ = 0.021 * xh_px                     # a medium nib, slightly stub: half-sizes (px)
    phi0 = math.radians(-38)                                    # nib edge angle in image coords (y down)
    flex = 0.55                                                 # how far the tines open on a full downstroke
    flick_pts = []
    for d in downs:
        P = np.column_stack([origin[0] + d[:, 0] * xh_px, origin[1] - d[:, 1] * xh_px])
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1); s = np.concatenate([[0], np.cumsum(seg)])
        if s[-1] < 1: continue
        u = np.arange(0, s[-1], 0.35)
        P = np.column_stack([np.interp(u, s, P[:, 0]), np.interp(u, s, P[:, 1])])
        v = np.gradient(P, axis=0); v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
        k = np.linalg.norm(np.gradient(v, axis=0), axis=1) / 0.35
        k = np.convolve(k, np.ones(9) / 9, 'same')
        L = len(P); idx = np.arange(L); per_unit = xh_px / 0.35      # samples per x-height
        down = np.clip(v[:, 1], 0, 1) ** 1.4
        ramp_in = np.clip(idx / (0.10 * per_unit), 0, 1)
        ramp_out = np.clip((L - 1 - idx) / (0.14 * per_unit), 0, 1)
        ramp = np.minimum(ramp_in, ramp_out)
        hand = 1 + 0.18 * smooth_noise(L, 0.5 * per_unit, rng)       # the hand presses unevenly
        press = np.clip(down * (0.35 + 0.65 * ramp) * hand, 0, 1.3)
        press = np.convolve(np.pad(press, 12, mode='edge'), np.ones(25) / 25, 'valid')
        phi = phi0 + math.radians(rng.normal(0, 4))                   # the pen turns a little in the fingers
        a = A_ * (1 + flex * 0.30 * press) * (0.92 + 0.08 * ramp) * (1 + 0.03 * smooth_noise(L, 30, rng))
        b = B_ * (1 + flex * 1.5 * press) * (0.80 + 0.20 * ramp) * (1 + 0.06 * smooth_noise(L, 30, rng))
        slow = np.clip(k * xh_px * 0.06, 0, 1)
        load = float(np.exp(rng.normal(0, 0.16)))                     # each stroke its own ink load
        flow = load * (1 + 0.20 * smooth_noise(L, 0.6 * per_unit, rng))
        dens = (0.74 + 0.24 * slow + 0.20 * (1 - ramp_in) + 0.12 * (1 - ramp_out)
                - 0.14 * (1 - down) * (1 - slow)) * flow
        dry = np.zeros(L)
        if s[-1] > 1.4 * xh_px and rng.random() < 0.30:              # the feed can't keep up: a dry, broken patch
            c0 = rng.uniform(0.25, 0.85) * L; half = rng.uniform(0.08, 0.25) * per_unit
            w_ = np.clip(1 - np.abs(idx - c0) / half, 0, 1) ** 0.7
            w_ *= (1 - down * 0.6)                                    # mostly on the fast, light parts
            dry = np.maximum(dry, w_ * rng.uniform(0.55, 0.9))
        dens = dens * (1 - 0.55 * dry)
        dens = np.clip(dens, 0.35, 1.8)
        x0, y0 = int(P[:, 0].min() - A_ * 4), int(P[:, 1].min() - A_ * 4)
        x1, y1 = int(P[:, 0].max() + A_ * 4) + 1, int(P[:, 1].max() + A_ * 4) + 1
        x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1, W), min(y1, H)
        w, h = x1 - x0, y1 - y0
        im_a = Image.new('L', (w * ss, h * ss), 0); da = ImageDraw.Draw(im_a)
        im_k = Image.new('L', (w * ss, h * ss), 0); dk = ImageDraw.Draw(im_k)
        im_p = Image.new('L', (w * ss, h * ss), 0); dp = ImageDraw.Draw(im_p)
        im_d = Image.new('L', (w * ss, h * ss), 0); dd = ImageDraw.Draw(im_d)
        KMAX = 2.4
        for (px, py), aa, bb, kk, pp, dr in zip(P, a, b, dens, press, dry):
            poly = nib_ellipse((px - x0) * ss, (py - y0) * ss, aa * ss, bb * ss * (1 - 0.35 * dr), phi)
            da.polygon(poly, fill=255)
            dk.polygon(poly, fill=int(np.clip(kk / KMAX, 0, 1) * 255))
            dp.polygon(poly, fill=int(np.clip(pp, 0, 1) * 255))
            if dr > 0.02: dd.polygon(poly, fill=int(dr * 255))
        # where the nib sat: some starts leave a little pool, some ends too
        if rng.random() < 0.28:
            r = B_ * rng.uniform(1.05, 1.5) * load
            pt = P[min(2, L - 1)]
            poly = blob((pt[0] - x0) * ss, (pt[1] - y0) * ss, r * ss, rng)
            da.polygon(poly, fill=255); dk.polygon(poly, fill=int(min(1.0, 1.30 * load / KMAX) * 255))
        if rng.random() < 0.15:
            r = B_ * rng.uniform(1.0, 1.4) * load
            pt = P[-1]
            poly = blob((pt[0] - x0) * ss, (pt[1] - y0) * ss, r * ss, rng)
            da.polygon(poly, fill=255); dk.polygon(poly, fill=int(min(1.0, 1.25 * load / KMAX) * 255))
        al = np.asarray(im_a.resize((w, h), Image.LANCZOS), np.float32) / 255
        kd = np.asarray(im_k.resize((w, h), Image.LANCZOS), np.float32) / 255 * KMAX
        pd = np.asarray(im_p.resize((w, h), Image.LANCZOS), np.float32) / 255
        dm = np.asarray(im_d.resize((w, h), Image.LANCZOS), np.float32) / 255
        kd = np.where(al > 0.03, kd / np.maximum(al, 0.03), 0)
        D[y0:y1, x0:x1] += np.clip(al, 0, 1) * kd
        PR[y0:y1, x0:x1] = np.maximum(PR[y0:y1, x0:x1], pd)
        DRY[y0:y1, x0:x1] = np.maximum(DRY[y0:y1, x0:x1], dm)
        # stroke ends: where the sprays start from
        flick_pts.append((P[-1].copy(), v[-1].copy(), s[-1]))
    # spatter: a few small sprays of droplets, flung on from the flicks, plus a single small blot
    sp = Image.new('L', (W * 2, H * 2), 0); ds = ImageDraw.Draw(sp)
    # where they land: off the last flick of a line, out into the empty paper to its right (Χρυσόστομος, Καρρέρ),
    # and one drop that fell from the pen in the gap between the two names
    gap = kw.get('line_gap', 2.6) * xh_px
    sprays = []
    for li in (1, 4):
        band = [f for f in flick_pts if abs(f[0][1] - (origin[1] + li * gap)) < 1.2 * xh_px and f[2] > 0.9 * xh_px]
        if band:
            xe = max(f[0][0] for f in band)
            sprays.append(((xe, origin[1] + li * gap), None, None))
    for (px, py), dv, _ in sprays:
        # a fan of fine droplets thrown down-right into the empty paper: many tiny, a few larger, smaller further out
        n = int(rng.integers(14, 24))
        # well clear of the word and off its baseline, so the drops can't be read as dots after it
        px = px + rng.uniform(1.1, 1.6) * xh_px; py = py - rng.uniform(0.3, 0.7) * xh_px
        fan = rng.uniform(0.85, 1.15)                                    # steeply down-right, into the gap below
        for _ in range(n):
            t = rng.random() ** 0.8
            dist = (0.3 + 2.0 * t) * xh_px
            ang = fan + rng.normal(0, 0.28)
            cx, cy = px + math.cos(ang) * dist, py + math.sin(ang) * dist
            r = float(np.clip(np.exp(rng.normal(0.55, 0.55)) * (1.15 - 0.6 * t), 0.7, 6.0))   # px at 300 dpi
            if r > 3.2:                                       # bigger drops stretch a little along their flight
                e = 1.5
                pts = [(cx + math.cos(q) * r * e * math.cos(ang) - math.sin(q) * r * math.sin(ang),
                        cy + math.cos(q) * r * e * math.sin(ang) + math.sin(q) * r * math.cos(ang)) for q in np.linspace(0, 2 * np.pi, 24)]
                ds.polygon([(x * 2, y * 2) for x, y in pts], fill=int(255 * rng.uniform(0.75, 1)))
            else:
                ds.polygon([(x * 2, y * 2) for x, y in blob(cx, cy, r, rng, 16, 0.2)], fill=int(255 * rng.uniform(0.55, 1)))
    bx = origin[0] + rng.uniform(7.5, 9.5) * xh_px; by = origin[1] + 2 * gap - rng.uniform(0.2, 0.6) * xh_px
    rb = 0.10 * xh_px * rng.uniform(0.9, 1.15)
    ds.polygon([(x * 2, y * 2) for x, y in blob(bx, by, rb, rng, 60, 0.22)], fill=235)
    for _ in range(int(rng.integers(3, 7))):                   # its satellites
        ang = rng.uniform(0, 2 * np.pi); dist = rb * rng.uniform(1.3, 2.4); r = rb * rng.uniform(0.06, 0.18)
        ds.polygon([(x * 2, y * 2) for x, y in blob(bx + math.cos(ang) * dist, by + math.sin(ang) * dist, r, rng, 16, 0.2)], fill=210)
    spa = np.asarray(sp.resize((W, H), Image.LANCZOS), np.float32) / 255
    D += spa * 1.25
    return D, PR, DRY


def ink_on_paper(D, DRY, fibres, grain, rim=0.11, wick=0.30, seed=5):
    """dye on paper: ragged edges where it ran into the fibres, a little creep along them, rims where it dried,
    fibre tops only where the pen was dry, and a fine uneven take-up. fibres, grain: 0..1 maps of the paper."""
    from scipy import ndimage as ndi
    H, W = D.shape
    rng = np.random.default_rng(seed)
    # ragged edges: warp the ink by a small displacement (about 0.3 px at 300 dpi)
    nx = ndi.gaussian_filter(rng.normal(0, 1, (H, W)).astype(np.float32), 1.6)
    ny = ndi.gaussian_filter(rng.normal(0, 1, (H, W)).astype(np.float32), 1.6)
    sc = 0.32 / max(nx.std(), 1e-6)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    Dw = ndi.map_coordinates(D, [yy + ny * sc, xx + nx * sc], order=1, mode='nearest')
    Dw = ndi.gaussian_filter(Dw, 0.35)
    creep = ndi.gaussian_filter(Dw, 1.4) * (0.15 + wick * fibres)
    D2 = np.maximum(Dw, creep)
    cov = np.clip(Dw, 0, 1)
    D2 = D2 + rim * np.clip(cov - ndi.gaussian_filter(cov, 2.0), 0, 1) * cov
    tops = np.clip((grain - 0.42) * 3.0, 0, 1)                  # the pen ran dry: ink stays on the fibre tops
    D2 = D2 * (1 - DRY * (1 - tops) * 0.85)
    return D2 * (0.86 + 0.22 * grain)


INK_BROWN_BLACK = np.array([1.38, 1.72, 2.06])                 # aged iron-gall: brown where thin, near black in pools


if __name__ == '__main__':
    import sys
    from scipy import ndimage as ndi
    out = sys.argv[1]
    lines = ['Ο Ζακύνθου', 'Χρυσόστομος', '', 'Δήμαρχος Ζακυνθίων', 'Λουκάς Καρρέρ']
    H, W = 1500, 2100
    D, PR, DRY = write(lines, 100, (H, W), (120, 260), line_gap=2.6)
    r = np.random.default_rng(3)
    grain = ndi.gaussian_filter(r.random((H, W)).astype(np.float32), 1.0); grain = (grain - grain.min()) / (np.ptp(grain) + 1e-6)
    fib = ndi.gaussian_filter(r.random((H, W)).astype(np.float32), 3.0); fib = (fib - fib.min()) / (np.ptp(fib) + 1e-6)
    D = ink_on_paper(D, DRY, fib, grain)
    paper = np.ones((H, W, 3), np.float32) * np.array([0.925, 0.893, 0.823])
    col = paper * np.exp(-D[..., None] * INK_BROWN_BLACK)
    Image.fromarray((np.clip(col, 0, 1) * 255).astype(np.uint8)).save(out)
