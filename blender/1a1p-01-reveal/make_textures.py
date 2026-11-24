"""Textures for the Blender scene, drawn by our own 2D engine (no AI images).
card_A.png / card_B.png: the finished cards (sRGB). stamp_mask_A.png: the ΑΛΗΘΕΙΑ imprint in card-A UV space.
x_wipe_B.png: R = moment of reveal (0..1, stroke 1 then stroke 2), G = ink alpha. 16-bit, non-color.
Also prints the stamp box and the two strokes in card px (for the 3D stamp and pen).
usage: python3 make_textures.py [out_dir]   (default: textures/ next to this file)"""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'mia-alitheia-ena-psema', '01-dyo-vomves-i-tria-navagia'))
import numpy as np, cv2
import cardanim05 as ca, cards05 as T
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'textures'); UP = 2
os.makedirs(OUT, exist_ok=True)
def up(a): return cv2.resize(a, None, fx=UP, fy=UP, interpolation=cv2.INTER_LANCZOS4)
A = ca.card_A(9.0); B = ca.card_B.copy()
cv2.imwrite(f'{OUT}/card_A.png', np.clip(up(A), 0, 255).astype(np.uint8))
cv2.imwrite(f'{OUT}/card_B.png', np.clip(up(B), 0, 255).astype(np.uint8))
# stamp, exactly where the 2D version slams it
m = np.zeros((T.CH, T.CW), np.float32); hh, ww = ca.STAMP.shape
m[ca.SY:ca.SY + hh, ca.SX:ca.SX + ww] = ca.STAMP[:T.CH - ca.SY, :T.CW - ca.SX]
cv2.imwrite(f'{OUT}/stamp_mask_A.png', (np.clip(up(m), 0, 1) * 65535).astype(np.uint16))
# X: reveal time along each stroke
Y, X = np.mgrid[0:T.CH, 0:T.CW].astype(np.float32)
def proj(p0, p1):
    d = np.array(p1, np.float32) - p0; L = np.hypot(*d); u = d / L
    return np.clip(((X - p0[0]) * u[0] + (Y - p0[1]) * u[1]) / L, 0, 1)
S1 = ((70, 120), (430, 330), (840, 620)); S2 = ((830, 90), (480, 360), (90, 640))
a1, a2 = np.clip(ca.X1, 0, 1), np.clip(ca.X2, 0, 1)
r = np.where(a1 > 0.02, 0.5 * proj(np.float32(S1[0]), S1[2]), 0.5 + 0.5 * proj(np.float32(S2[0]), S2[2]))
g = np.maximum(a1, a2)
img = np.zeros((T.CH, T.CW, 3), np.float32); img[..., 2] = r; img[..., 1] = g     # BGR -> R=r, G=g
cv2.imwrite(f'{OUT}/x_wipe_B.png', (np.clip(up(img), 0, 1) * 65535).astype(np.uint16))
info = dict(CW=T.CW, CH=T.CH, stamp=[int(ca.SX), int(ca.SY), int(ww), int(hh)], S1=S1, S2=S2)
print(json.dumps(info))
