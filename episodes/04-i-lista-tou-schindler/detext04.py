"""Ep04: erase AI pseudo-letters on signs, labels and plates (blank surfaces, no fake text).
Each box is the INSIDE of a flat surface (number plate, sign board, label): it is refilled with
that surface's own tone plus a fine grain, edges feathered. Outlines around the box stay.
Run once after prep04.py (keeps a backup of each plate as <key>_pre_detext.png)."""
import cv2, numpy as np, os, sys, shutil

ROOT = os.environ.get('MT_ROOT', '/home/user/ep04')
BOXES = {   # (l, t, r, b) as plate fractions
    'S06_shop':    [(0.837, 0.329, 0.878, 0.3374),     # label on the box, top shelf
                    (0.915, 0.6135, 0.958, 0.6215)],   # label on the packet in the window
    'S21_train':   [(0.3367, 0.5264, 0.3797, 0.5329)], # station name board
    'S24_beverly': [(0.128, 0.634, 0.168, 0.6468)],     # car number plate
}


def run(k):
    src = f'{ROOT}/plates/{k}.png'; bak = f'{ROOT}/plates/{k}_pre_detext.png'
    if not os.path.exists(bak): shutil.copy(src, bak)
    img = cv2.imread(bak).astype(np.float32)
    Hh, Ww = img.shape[:2]
    out = img.copy(); rng = np.random.default_rng(7)
    for l, t, r, b in BOXES[k]:
        x0, y0, x1, y1 = int(l * Ww), int(t * Hh), int(r * Ww), int(b * Hh)
        sub = img[y0:y1, x0:x1]
        lum = sub.mean(2)
        bgc = np.median(sub[lum >= np.percentile(lum, 50)], 0)          # the surface, not the marks
        nz = cv2.GaussianBlur(rng.normal(0, 1, lum.shape).astype(np.float32), (0, 0), 0.8)
        fill = bgc[None, None, :] + (nz / (nz.std() + 1e-6) * 7.0)[..., None]
        m = np.zeros((Hh, Ww), np.float32); m[y0:y1, x0:x1] = 1
        m = cv2.GaussianBlur(m, (0, 0), 1.5)[y0 - 6:y1 + 6, x0 - 6:x1 + 6][..., None]
        F = out[y0 - 6:y1 + 6, x0 - 6:x1 + 6].copy()
        F[6:-6, 6:-6] = fill
        out[y0 - 6:y1 + 6, x0 - 6:x1 + 6] = out[y0 - 6:y1 + 6, x0 - 6:x1 + 6] * (1 - m) + F * m
    cv2.imwrite(src, np.clip(out, 0, 255).astype(np.uint8))
    print('detext', k, len(BOXES[k]))


if __name__ == '__main__':
    for k in (sys.argv[1:] or BOXES):
        run(k)
