"""Episode 5 plates: warm charcoal tone, pure black and white everywhere.
The only colour in the whole episode is the fire of the two 3D clips (B, D); every plate is toned from a grey copy.
The plates came from Replicate (.png) and Leonardo (.jpg): each is first written as img/<k>.png (depth.py reads .png).
usage: MT_ROOT=/home/user/ep05 python3 prep05.py [keys]"""
import sys, os, glob, cv2, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'engine'))
from prep import tone, remove_marks, paper_mask

ROOT = os.environ.get('MT_ROOT', '/home/user/ep05')
MARKS = {}  # (l, t, r, b) fractions: signatures / stray marks


def src(k):
    for ext in ('png', 'jpg'):
        p = f'{ROOT}/img/{k}.{ext}'
        if os.path.exists(p): return p
    raise FileNotFoundError(k)


keys = sys.argv[1:] or sorted({os.path.basename(p)[:3] for p in glob.glob(f'{ROOT}/img/S??.*')})
os.makedirs(f'{ROOT}/plates', exist_ok=True)
for k in keys:
    img = cv2.imread(src(k))
    if not os.path.exists(f'{ROOT}/img/{k}.png'):
        cv2.imwrite(f'{ROOT}/img/{k}.png', img)
    if k in MARKS:
        img = remove_marks(img, MARKS[k])
    toned, _ = tone(cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR))
    cv2.imwrite(f'{ROOT}/plates/{k}.png', toned)
    cv2.imwrite(f'{ROOT}/plates/{k}_paper.png', (paper_mask(img) * 255).astype(np.uint8))
    print('plate', k, img.shape[1], 'x', img.shape[0], flush=True)
