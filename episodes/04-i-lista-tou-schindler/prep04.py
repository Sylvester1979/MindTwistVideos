"""Episode 4 plates: warm charcoal tone, pure black and white everywhere.
The only colour in the whole episode is the girl's red coat in S32 (kept inside RED_BOX only).
usage: MT_ROOT=/home/user/ep04 python3 prep04.py [keys]"""
import sys, os, glob, cv2, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'engine'))
from prep import tone, remove_marks, paper_mask

ROOT = os.environ.get('MT_ROOT', '/home/user/ep04')
MARKS = {}  # (l, t, r, b) fractions: signatures / stray marks
RED_BOX = {'S32_redcoat': (0.40, 0.72, 0.60, 0.93)}  # where colour is allowed

keys = sys.argv[1:] or sorted(os.path.basename(p)[:-4] for p in glob.glob(f'{ROOT}/img/S*.png'))
os.makedirs(f'{ROOT}/plates', exist_ok=True)
for k in keys:
    img = cv2.imread(f'{ROOT}/img/{k}.png')
    if k in MARKS:
        img = remove_marks(img, MARKS[k])
    toned, sat = tone(img)
    warm = f'{ROOT}/plates/{k}_warm.png'
    if k in RED_BOX:
        H, W = img.shape[:2]
        l, t, r, b = RED_BOX[k]
        keep = np.zeros((H, W), np.float32)
        keep[int(t * H):int(b * H), int(l * W):int(r * W)] = 1
        keep = cv2.GaussianBlur(keep, (0, 0), 6)
        s = sat.astype(np.float32) / 255 * keep
        g = cv2.cvtColor(toned, cv2.COLOR_BGR2GRAY)
        bw, _ = tone(cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR))
        toned = (bw * (1 - s[..., None]) + img.astype(np.float32) * s[..., None]).astype(np.uint8)
        cv2.imwrite(warm, (s * 255).astype(np.uint8))
    else:
        # no colour at all: tone a grey copy so nothing saturated survives
        toned, _ = tone(cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR))
        if os.path.exists(warm): os.remove(warm)
    cv2.imwrite(f'{ROOT}/plates/{k}.png', toned)
    cv2.imwrite(f'{ROOT}/plates/{k}_paper.png', (paper_mask(img) * 255).astype(np.uint8))
    print('plate', k, flush=True)
