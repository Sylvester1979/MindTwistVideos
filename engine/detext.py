"""Replace AI pseudo-text on documents with honest illegible scribble lines (no letters)."""
import cv2, numpy as np, sys, os
ROOT = os.environ.get('MT_ROOT', '/home/user/ww1')
BOXES = {'P02_signing': ([0.34, 0.585, 0.83, 0.715], 230, 50, 90),
         'P14_letter': ([0.36, 0.39, 0.93, 0.715], 430, 75, 110)}
ROWS = {'P14_letter': (np.deg2rad(13), 76)}

def run(k):
    img = cv2.imread(f'{ROOT}/plates/{k}.png')
    box, maxw, maxh, thr = BOXES[k]
    Hh, Ww = img.shape[:2]
    x0, y0, x1, y1 = int(box[0] * Ww), int(box[1] * Hh), int(box[2] * Ww), int(box[3] * Hh)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    roi = g[y0:y1, x0:x1]
    dark = (roi < thr).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(dark, 8)
    m = np.zeros_like(dark)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if w < maxw and h < maxh and a > 6:
            m[lab == i] = 1
    m = cv2.dilate(m, np.ones((5, 5), np.uint8))
    full = np.zeros((Hh, Ww), np.uint8); full[y0:y1, x0:x1] = m * 255
    clean = cv2.inpaint(img, full, 9, cv2.INPAINT_TELEA)
    # line blobs -> scribbles
    lines = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 9)))
    if k in ROWS:
        ang, sp = ROWS[k]
        hull = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (71, 71)))
        hull = cv2.erode(hull, np.ones((9, 9), np.uint8))
        d0 = np.array([np.cos(ang), np.sin(ang)]); nr = np.array([-d0[1], d0[0]])
        hh, ww = hull.shape; c = np.array([ww / 2, hh / 2])
        lines = np.zeros_like(hull); R = int(np.hypot(ww, hh))
        for kk in range(-R // sp, R // sp + 1):
            p0 = c + nr * kk * sp - d0 * R; p1 = c + nr * kk * sp + d0 * R
            tmp = np.zeros_like(hull); cv2.line(tmp, tuple(np.int32(p0)), tuple(np.int32(p1)), 1, 9)
            lines |= (tmp & hull)
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(lines, 8)
    rng = np.random.default_rng(5)
    ink = np.zeros((Hh, Ww), np.float32)
    for i in range(1, n2):
        if st2[i][4] < 400: continue
        ys, xs = np.where(lab2 == i)
        pts = np.stack([xs, ys], 1).astype(np.float32)
        mu = pts.mean(0); u, s_, vt = np.linalg.svd(pts - mu, full_matrices=False)
        d = vt[0]; d = d if d[0] > 0 else -d
        proj = (pts - mu) @ d
        a, b = proj.min() + 6, proj.max() - 6
        L = b - a
        if L < 30: continue
        tt = np.linspace(0, 1, int(L / 2))
        nrm = np.array([-d[1], d[0]])
        amp = 7 + rng.random() * 4
        ph = rng.random(6) * 6.28
        wav = amp * (0.6 * np.sin(tt * L / 9 + ph[0]) + 0.3 * np.sin(tt * L / 4.1 + ph[1]) + 0.25 * np.sin(tt * L / 2.3 + ph[2]))
        # gaps between 'words'
        gate = (np.sin(tt * L / 37 + ph[3]) + 0.35 * np.sin(tt * L / 13 + ph[4])) > -0.55
        P = mu + d[None] * (a + tt[:, None] * L) + nrm[None] * wav[:, None]
        P = P + np.array([x0, y0])
        seg = []
        for j in range(len(P)):
            if gate[j]: seg.append(P[j])
            elif len(seg) > 2:
                cv2.polylines(ink, [np.int32(np.array(seg) * 4)], False, 1.0, 3, cv2.LINE_AA, shift=2); seg = []
            else: seg = []
        if len(seg) > 2:
            cv2.polylines(ink, [np.int32(np.array(seg) * 4)], False, 1.0, 3, cv2.LINE_AA, shift=2)
    ink = cv2.GaussianBlur(ink, (0, 0), 0.7)[..., None] * 0.85
    out = clean * (1 - ink) + np.array([30, 28, 26], np.float32) * ink
    cv2.imwrite(f'{ROOT}/plates/{k}.png', np.clip(out, 0, 255).astype(np.uint8))
    print('detext', k, int(m.sum()), n2)

if __name__ == '__main__':
    for k in sys.argv[1:]: run(k)
