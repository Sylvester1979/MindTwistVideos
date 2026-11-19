"""Episode 3 plates, same recipe as episode 2: warm charcoal tone (keeps only strong colour = red geraniums),
remove artist marks / dates in corners, paper mask.  usage: MT_ROOT=/home/user/ep03 python3 prep03.py [keys]"""
import sys, os, glob, cv2, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'engine'))
from prep import tone, remove_marks, paper_mask

ROOT = os.environ.get('MT_ROOT', '/home/user/ep03')
# (l, t, r, b) fractions: signatures, dates, pseudo-text
MARKS = {
    'B01_portrait': [[0.90, 0.950, 1.00, 0.995], [0.55, 0.085, 0.78, 0.135]],
    'B05_stefa': [[0.82, 0.945, 0.95, 0.985]],
    'C02_sickboy': [[0.87, 0.945, 0.97, 0.995]],
    'D01_gate': [[0.88, 0.935, 1.00, 0.975]],
}

keys = sys.argv[1:] or sorted(os.path.basename(p)[:-4] for p in glob.glob(f'{ROOT}/img/*.png'))
os.makedirs(f'{ROOT}/plates', exist_ok=True)
for k in keys:
    img = cv2.imread(f'{ROOT}/img/{k}.png')
    if k in MARKS:
        img = remove_marks(img, MARKS[k])
    toned, sat = tone(img)
    cv2.imwrite(f'{ROOT}/plates/{k}.png', toned)
    for f in (f'{ROOT}/plates/{k}_warm.png',):
        if os.path.exists(f): os.remove(f)
    cv2.imwrite(f'{ROOT}/plates/{k}_paper.png', (paper_mask(img) * 255).astype(np.uint8))
    print('plate', k, flush=True)
