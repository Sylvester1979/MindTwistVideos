"""Cut-paper puppets made from silhouette images, animated like real cut-out stop motion.

A figure is split at the hips. The upper card stays rigid (it is paper); the legs are
re-posed into a 4-pose walk cycle: contact A -> passing -> contact B (legs mirrored) -> passing.
Poses are held for a few frames, as in hand-made cut-out animation."""
import math, functools, cv2, numpy as np
from shadow import cutout

def extract_figures(path, min_area=4000, ground=None, pad=6):
    """silhouette image -> list of alpha crops (float 0..1), left to right.
    ground: fraction of height below which everything is removed (a drawn floor line)."""
    a = cutout(path)
    if ground is not None:
        a[int(a.shape[0] * ground):] = 0
    m = (a > 0.5).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, ar = st[i]
        if ar < min_area: continue
        sub = a[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad].copy()
        keep = (lab[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad] == i)
        keep = cv2.dilate(keep.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        sub[~keep] = 0
        out.append((x, sub))
    out.sort(key=lambda z: z[0])
    return [s for _, s in out]

def split_vertical(alpha, x_frac):
    """cut a group card into two along a vertical line (e.g. adult + child holding hands)."""
    x = int(alpha.shape[1] * x_frac)
    l, r = alpha.copy(), alpha.copy()
    l[:, x:] = 0; r[:, :x] = 0
    return l, r

def hip_line(alpha):
    """lowest row above which the figure is one piece, i.e. where the legs part."""
    m = alpha > 0.5
    h = m.shape[0]
    rows = np.where(m.any(1))[0]; top, bot = rows[0], rows[-1]
    def runs(r):
        v = m[r].astype(np.int8); return int(((np.diff(v) == 1).sum()) + (1 if v[0] else 0))
    y = bot - 3
    two = None
    while y > top + (bot - top) * 0.35:
        if runs(y) >= 2: two = y
        elif two is not None and two - y > 4: break
        y -= 1
    if two is None:            # legs never part (long coat): use 78% of height
        return int(top + 0.78 * (bot - top))
    return int(two)

def make_poses(alpha, squeeze=0.30):
    """returns [A, P, B, P] alpha cards of identical size + (hip_y, center_x)."""
    hy = hip_line(alpha)
    cols = np.where((alpha[hy:] > 0.5).any(0))[0]
    cx = (cols[0] + cols[-1]) / 2 if len(cols) else alpha.shape[1] / 2
    H_, W_ = alpha.shape
    def warp(sx):
        M = np.float32([[sx, 0, cx - sx * cx], [0, 1, 0]])
        return cv2.warpAffine(legs, M, (W_, legs.shape[0]), flags=cv2.INTER_LINEAR)
    poses = []
    y0 = max(0, hy - 8)
    legs = alpha[y0:].copy()
    for sx in (1.0, squeeze, -1.0, squeeze):
        p = alpha.copy(); p[hy:] = 0
        p[y0:] = np.maximum(p[y0:], warp(sx))
        poses.append(np.clip(p, 0, 1))
    return poses, hy, cx

class Walker:
    """a puppet crossing the screen. x(t) continuous; legs held in poses (stop-motion)."""
    def __init__(self, alpha, height_px, x0, y_feet, speed, cycle=0.9, phase=0.0, hold_fps=10, sway=1.2, flip=False):
        if flip: alpha = alpha[:, ::-1].copy()
        self.poses, self.hy, self.cx = make_poses(alpha)
        self.parts = None
        self.h, self.w = alpha.shape
        rows = np.where((alpha > 0.5).any(1))[0]
        self.foot = rows[-1]; self.top = rows[0]
        self.scale = height_px / max(1, self.foot - self.top)
        self.x0, self.yf, self.speed, self.cycle, self.phase = x0, y_feet, speed, cycle, phase
        self.hold_fps, self.sway = hold_fps, sway
    def split(self, alpha, x_frac, cycle2, phase2):
        """one card holding two figures (e.g. adult and child holding hands): rig each
        figure's legs separately, keep the joined upper card intact."""
        l, r = split_vertical(alpha, x_frac)
        self.parts = [(make_poses(l)[0], 1.0, 0.0), (make_poses(r)[0], cycle2, phase2)]
        rows = np.where((r > 0.5).any(1))[0]
        cols = np.where((alpha[int(rows[-1] * 0.9):] > 0.5).any(0))[0]
        self.cx = (cols[0] + cols[-1]) / 2
        return self
    def pose_at(self, t):
        tq = math.floor(t * self.hold_fps) / self.hold_fps          # held frames
        if self.parts:
            card = None
            for poses, cm, ph in self.parts:
                k = int(((tq / (self.cycle * cm) + self.phase + ph) % 1.0) * 4) % 4
                card = poses[k] if card is None else np.maximum(card, poses[k])
            return card, tq
        k = int(((tq / self.cycle + self.phase) % 1.0) * 4) % 4
        return self.poses[k], tq
    def render_into(self, canvas, t, W, H, extra_rot=0.0):
        card, tq = self.pose_at(t)
        x = self.x0 + self.speed * t
        rot = math.radians(self.sway * math.sin(2 * math.pi * (tq / self.cycle + self.phase) * 2) + extra_rot)
        s = self.scale
        # anchor: feet centre on (x, y_feet)
        ax, ay = self.cx, self.foot
        c, si = math.cos(rot) * s, math.sin(rot) * s
        M = np.float32([[c, -si, 0], [si, c, 0]])
        M[:, 2] = np.float32([x, self.yf]) - M[:, :2] @ np.float32([ax, ay])
        a = cv2.warpAffine(card, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
        np.maximum(canvas, a, out=canvas)
        return x
