"""Style test 2: AI-cut puppets walking (stop-motion legs) on our own backlit screen.
Writes stills + a short motion test $MT_ROOT/test/march_test.mp4"""
import sys, os, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'engine'))
import cv2, numpy as np
from eng import W, H, ROOT, Film
from shadow import screen, cast, cutout, place_alpha
from cutpuppet import extract_figures, split_vertical, Walker

IMG = f'{ROOT}/img'
film = Film()
kids = extract_figures(f'{IMG}/C01_kids.png', ground=0.885) + extract_figures(f'{IMG}/C02_kids.png')
grp = extract_figures(f'{IMG}/C03_korczak.png', ground=0.93)
grp = max(grp, key=lambda a: a.size)
print('kids', len(kids), [k.shape for k in kids])

GROUND = H * 0.80
lead = Walker(grp, H * 0.46, W * 0.30, GROUND, speed=70, cycle=1.15, hold_fps=8, sway=0.6)
row = []
for i, k in enumerate(kids):
    hpx = H * (0.20 + 0.035 * ((i * 5) % 4))
    row.append(Walker(k, hpx, W * 0.30 - 260 - i * 175, GROUND + (i % 2) * 6, speed=70,
                      cycle=0.8 + 0.05 * (i % 3), phase=0.27 * i, hold_fps=8, sway=1.0))
back = [Walker(k, H * 0.13, W * 0.10 - i * 120, GROUND - 120, speed=52, cycle=0.75, phase=0.4 * i, hold_fps=8)
        for i, k in enumerate(kids[:6])]
wall = cutout(f'{IMG}/S04_street.png', lo=30, hi=200, size=(W, H)); wall[int(H * 0.74):] = 0
house = cutout(f'{IMG}/C04_house.png')

def frame(t, fi, warmth=1.0):
    img = screen(t, warmth=warmth, lamp=(0.62, 0.40))
    img = cast(img, wall, blur=3.0, density=0.55)                       # far scenery: soft, pale
    img = cast(img, place_alpha(house, W * 0.86, H * 0.42, 1.05), blur=1.8, density=0.8)
    cb = np.zeros((H, W), np.float32)
    for w in back: w.render_into(cb, t, W, H)
    img = cast(img, cb, blur=2.4, density=0.6)                          # children further back
    cm = np.zeros((H, W), np.float32)
    lead.render_into(cm, t, W, H)
    for w in row: w.render_into(cm, t, W, H)
    img = cast(img, cm, blur=0.7, density=0.97)                         # pressed on the screen
    img = film.apply(img, fi, t, strength=0.5, weave=True, flicker=0.4)
    return np.clip(img, 0, 255).astype(np.uint8)

os.makedirs(f'{ROOT}/test', exist_ok=True)
for t in (0.0, 2.0, 4.0):
    cv2.imwrite(f'{ROOT}/test/m2_{t}.jpg', frame(t, int(t * 30)), [cv2.IMWRITE_JPEG_QUALITY, 90])
cv2.imwrite(f'{ROOT}/test/m2_drained.jpg', frame(3.0, 90, warmth=0.12), [cv2.IMWRITE_JPEG_QUALITY, 90])
if len(sys.argv) > 1:
    p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', '30', '-i', '-',
                          '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-pix_fmt', 'yuv420p', f'{ROOT}/test/march_test.mp4'], stdin=subprocess.PIPE)
    for fi in range(int(float(sys.argv[1]) * 30)):
        p.stdin.write(frame(fi / 30, fi).tobytes())
    p.stdin.close(); p.wait()
print('ok')
