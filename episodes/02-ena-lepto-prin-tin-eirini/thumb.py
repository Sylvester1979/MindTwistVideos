"""Thumbnail 1080x1920: our pocket watch stopped at 10:59. Writes $MT_ROOT/thumb.jpg"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'engine'))
import cv2, numpy as np
from gfx import *

img = watch_scene(0, 0, 10 * 3600 + 59 * 60, cam=(0.5, 0.56, 1.10, -0.05), stopped=True)
img = band(img, 60, 520, 0.7, 120)
img = paste_alpha(img, text_layer(['ΣΚΟΤΩΘΗΚΕ'], 'serif', 120, 700, track=6), W / 2, 150, (236, 241, 245), 1, shadow=1)
img = paste_alpha(img, text_layer(['10:59'], 'type', 230, None), W / 2, 395, tuple(float(v) for v in RED * 1.55), 1, shadow=1)
img = band(img, 1500, 1800, 0.7, 120)
img = paste_alpha(img, text_layer(['Ο πόλεμος τελείωσε', 'στις 11:00'], 'serif', 92, 650, 1.1), W / 2, 1650, (236, 241, 245), 1, shadow=1)
img = Film().apply(img, 3, 0.1, strength=0.6, weave=False)
cv2.imwrite(f'{ROOT}/thumb.jpg', np.clip(img, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 92])
print('thumb ->', f'{ROOT}/thumb.jpg')
