"""Episode 03 look: period paper-craft colour (muted, warm, lifted blacks).

prep:  python3 look.py U2_window [...]   -> plates/{k}.png (graded, 1080-wide-ready source)
"""
import sys, os, cv2, numpy as np
ROOT = os.environ.get('MT_ROOT', '/home/user/ep03')
CREAM = np.array([200, 222, 236], np.float32)   # BGR warm paper highlight
UMBER = np.array([38, 46, 58], np.float32)      # BGR warm shadow


def grade(img, sat=0.62, warm=0.22, lift=0.06):
    """img float32 BGR 0..255 -> muted 1940s print look."""
    f = img.astype(np.float32)
    g = f.mean(2, keepdims=True)
    f = g + (f - g) * sat                                   # desaturate
    lum = np.clip(g / 255, 0, 1)
    tone = UMBER + (CREAM - UMBER) * lum                    # duotone of the same luminance
    f = f * (1 - warm) + tone * warm                        # warm split-tone
    f = f * (1 - lift) + UMBER * lift                       # lifted, warm blacks
    return np.clip(f, 0, 255)


if __name__ == '__main__':
    os.makedirs(f'{ROOT}/plates', exist_ok=True)
    for k in sys.argv[1:]:
        im = cv2.imread(f'{ROOT}/img/{k}.png').astype(np.float32)
        cv2.imwrite(f'{ROOT}/plates/{k}.png', grade(im).astype(np.uint8))
        print('plate', k, flush=True)
