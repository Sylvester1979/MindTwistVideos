"""Ep04 thumbnails 1080x1920 (two options) -> $MT_ROOT/thumb_A.jpg, thumb_B.jpg
A: the girl in the red coat (our only colour) + «ΕΝΑΣ ΝΑΖΙ / έσωσε 1.200 Εβραίους»
B: the slab under the stones (the hook) + «ΠΕΤΡΕΣ ΣΤΟΝ ΤΑΦΟ / ΕΝΟΣ ΝΑΖΙ»"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'engine'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2, numpy as np
from eng import *
import render04 as R

WHITE = (236, 241, 245)
REDT = tuple(float(v) for v in np.array([40, 40, 205], np.float32))     # BGR, close to the coat


def frame(k, cx, cy, z):
    P = R.plate(k)
    x, y, zz = R.keep_inside(P, cx * P.pw, cy * P.ph, z, 0, 0, 0)
    return P.render(0, 0, (x, y, zz), boil=0.0)


def finish(img, name):
    img = Film().apply(img, 3, 0.1, strength=0.5, weave=False)
    cv2.imwrite(f'{ROOT}/{name}', np.clip(img, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print('thumb ->', f'{ROOT}/{name}')


# A: the girl sits in the lower third, both lines on a dark band at the top
img = frame('S32_redcoat', 0.50, 0.80, 1.22)
img = band(img, 60, 560, 0.80, 120)
img = paste_alpha(img, text_layer(['ΕΝΑΣ ΝΑΖΙ'], 'serif', 150, 700, track=6), W / 2, 215, WHITE, 1, shadow=1)
img = paste_alpha(img, text_layer(['έσωσε 1.200', 'Εβραίους'], 'serif', 112, 700, 1.05), W / 2, 430, REDT, 1, shadow=1)
finish(img, 'thumb_A.jpg')

# B: the slab, close; text above and below the inscription
img = frame('S35_slab', 0.50, 0.47, 1.25)
img = band(img, 40, 430, 0.82, 110)
img = paste_alpha(img, text_layer(['ΠΕΤΡΕΣ', 'ΣΤΟΝ ΤΑΦΟ'], 'serif', 132, 700, 1.05, track=4), W / 2, 235, WHITE, 1, shadow=1)
img = band(img, 1480, 1840, 0.85, 110)
img = paste_alpha(img, text_layer(['ενός ναζί'], 'serif', 140, 700), W / 2, 1660, REDT, 1, shadow=1)
finish(img, 'thumb_B.jpg')
