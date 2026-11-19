"""Thumbnail 1080x1920: the doctor watering the red geraniums (our motif) + one contrast line.
Writes $MT_ROOT/thumb.jpg"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'engine'))
import cv2, numpy as np
from eng import *

P = Plate('A02_window')
img = P.render(0, 0, (P.pw * 0.50, P.ph * 0.535, 1.30), boil=0.0)
img = band(img, 40, 470, 0.72, 120)
img = paste_alpha(img, text_layer(['ΜΠΟΡΟΥΣΕ', 'ΝΑ ΣΩΘΕΙ'], 'serif', 128, 700, 1.05, track=4), W / 2, 250, (236, 241, 245), 1, shadow=1)
img = band(img, 1500, 1840, 0.82, 110)
img = paste_alpha(img, text_layer(['Δεν άφησε', 'τα παιδιά'], 'serif', 118, 700, 1.05), W / 2, 1670,
                  tuple(float(v) for v in RED * 1.6), 1, shadow=1)
img = Film().apply(img, 3, 0.1, strength=0.6, weave=False)
cv2.imwrite(f'{ROOT}/thumb.jpg', np.clip(img, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 92])
print('thumb ->', f'{ROOT}/thumb.jpg')
