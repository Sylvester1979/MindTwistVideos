"""Style frames for episode 03 (shadow theatre). Writes $MT_ROOT/test/style_*.jpg"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'engine'))
import cv2, numpy as np
from eng import W, H, ROOT, Film, vignette
from shadow import *

IMG = f'{ROOT}/img'
os.makedirs(f'{ROOT}/test', exist_ok=True)
film = Film()

def finish(img, fi=5):
    img = film.apply(img, fi, 0.2, strength=0.55, weave=False, flicker=0.3)
    return np.clip(img, 0, 255).astype(np.uint8)

def procession(t, cv, x0, ground, n_pairs=5, gap=118, s=1.0):
    """Korczak in front holding a child's hand, then pairs of children."""
    ph = t * 0.85
    J = puppet(cv, x0, ground - 160 * s, 1.0 * s, ph, kind='adult', facing=1, hold='down', stride=0.8, lean=4)
    # child holding his hand (just behind, to his left on screen)
    hx, hy = J['hand0']
    puppet(cv, hx - 28 * s, ground - 102 * 0.72 * s, 0.72 * s, ph * 1.25 + 0.3, hold='up', hair='bob', stride=0.9)
    hairs = ['cap', 'braid', 'bob', 'cap', 'bob', 'braid', 'cap', 'bob', 'braid', 'cap']
    for i in range(n_pairs * 2):
        k = i // 2; far = i % 2
        sc = (0.62 + 0.18 * ((i * 7) % 5) / 4) * s * (0.9 if far else 1.0)
        x = x0 - 150 * s - k * gap * s - far * 26 * s
        y = ground - 102 * sc - far * 10 * s
        hr = hairs[i % len(hairs)]
        puppet(cv, x, y, sc, ph * 1.2 + 0.37 * i, hair=hr, hat='cap' if hr == 'cap' else None,
               bundle=(i % 3 != 1), stride=0.9)

# --- frame A: the march along the ghetto wall (warm)
for name, warmth, tt in (('march_warm', 1.0, 0.3), ('march_drained', 0.12, 0.9)):
    img = screen(tt, warmth=warmth, lamp=(0.55, 0.38))
    wall = cutout(f'{IMG}/S04_street.png', size=(W, H))
    wall[int(H * 0.74):] = 0                          # keep the street floor as light
    img = cast(img, wall, blur=2.2, density=0.80)     # scenery further from the screen
    cv = Canvas(); procession(tt, cv, W * 0.80, H * 0.80, n_pairs=4)
    a = cv.alpha()
    img = cast(img, a, blur=0.6, density=0.97)        # puppets pressed on the screen
    cv2.imwrite(f'{ROOT}/test/style_{name}.jpg', finish(img), [cv2.IMWRITE_JPEG_QUALITY, 90])

# --- frame B: the window scene relit on our own screen (AI cut-out)
img = screen(0.2, warmth=1.0, lamp=(0.35, 0.22))
img = cast(img, cutout(f'{IMG}/S01_window.png', size=(W, H)), blur=0.8, density=0.95)
cv2.imwrite(f'{ROOT}/test/style_window.jpg', finish(img, 7), [cv2.IMWRITE_JPEG_QUALITY, 90])

# --- walk-cycle strip (6 phases) to judge the articulation
strip = []
for k in range(6):
    img = screen(0, lamp=(0.5, 0.5), flicker=0)
    cv = Canvas(); puppet(cv, W * 0.45, H * 0.55, 2.6, k / 6, hair='braid', bundle=True, rods=False)
    img = cast(img, cv.alpha(), blur=0.5)
    strip.append(cv2.resize(finish(img)[int(H * 0.2):int(H * 0.85)], (240, 416)))
cv2.imwrite(f'{ROOT}/test/style_walkcycle.jpg', np.hstack(strip), [cv2.IMWRITE_JPEG_QUALITY, 85])
print('style frames ok')
