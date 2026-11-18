"""Style test 3: tracking shot of the procession on our own backlit screen.
python3 style_test3.py [seconds]  -> stills (+ motion test if seconds given)"""
import sys, os, subprocess, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'engine'))
import cv2, numpy as np
from eng import W, H, ROOT, Film, fbm
from shadow import screen, cast, cutout, place_alpha
from cutpuppet import extract_figures, Walker, make_poses

IMG = f'{ROOT}/img'
film = Film()

def one(path, ground=None):
    f = extract_figures(path, ground=ground)
    return max(f, key=lambda a: (a > 0.5).sum())

korczak = one(f'{IMG}/K01_korczak.png', ground=0.90)
girl = one(f'{IMG}/K02_girl_hand.png')
kids = extract_figures(f'{IMG}/K03_kids4a.png') + extract_figures(f'{IMG}/K04_kids4b.png')
print('figures', len(kids))

# ---- pose sheet: check the rig on every figure
sheet = []
for a in [korczak, girl] + kids:
    poses, hy, cx = make_poses(a)
    row = [cv2.resize((255 - p * 255).astype(np.uint8), (150, int(150 * a.shape[0] / a.shape[1]))) for p in poses[:3]]
    hmax = 260
    row = [cv2.copyMakeBorder(cv2.resize(r, (int(r.shape[1] * hmax / r.shape[0]), hmax)), 0, 0, 0, 6, cv2.BORDER_CONSTANT, value=255) for r in row]
    sheet.append(np.hstack(row))
wmax = max(s.shape[1] for s in sheet)
sheet = [cv2.copyMakeBorder(s, 4, 4, 0, wmax - s.shape[1], cv2.BORDER_CONSTANT, value=255) for s in sheet]
cols = [np.vstack(sheet[i::3]) for i in range(3)]
hh = max(c.shape[0] for c in cols)
cols = [cv2.copyMakeBorder(c, 0, hh - c.shape[0], 0, 10, cv2.BORDER_CONSTANT, value=255) for c in cols]
os.makedirs(f'{ROOT}/test', exist_ok=True)
cv2.imwrite(f'{ROOT}/test/pose_sheet.jpg', cv2.resize(np.hstack(cols), None, fx=0.5, fy=0.5), [cv2.IMWRITE_JPEG_QUALITY, 80])

# ---- scenery strips (cut card), scrolled for a tracking shot
sky = cutout(f'{IMG}/S04_street.png', lo=30, hi=200, size=(W, H))[: int(H * 0.36)]
house = cutout(f'{IMG}/C04_house.png')
rng = np.random.default_rng(4)
def ground_strip(width):
    n = fbm(1, width, 18, 77, 3)[0]
    top = (H * 0.835 + 14 * (n - 0.5)).astype(int)
    g = np.zeros((H, width), np.float32)
    for x in range(width): g[top[x]:, x] = 1
    # cobble cut-outs: small light holes in the card
    for _ in range(width // 9):
        x = rng.integers(0, width); y = rng.integers(int(H * 0.87), H)
        cv2.ellipse(g, (int(x), int(y)), (int(rng.integers(10, 22)), int(rng.integers(4, 8))), 0, 0, 360, 0.0, -1, cv2.LINE_AA)
    return g
GROUND = ground_strip(W * 4)

GY = H * 0.84
SPEED = 95.0                                    # px/s the procession walks (camera follows)
lead = Walker(korczak, H * 0.47, W * 0.62, GY + 2, speed=0, cycle=1.2, hold_fps=8, sway=0.5)
hold = Walker(girl, H * 0.235, W * 0.44, GY, speed=0, cycle=0.82, phase=0.3, hold_fps=8, sway=0.9)
row = []
for i, k in enumerate(kids):
    hpx = H * (0.20 + 0.045 * ((i * 3) % 4) / 3)
    row.append(Walker(k, hpx, W * 0.20 - i * 150, GY + (i % 2) * 5, speed=0, cycle=0.78 + 0.05 * (i % 3),
                      phase=0.29 * i, hold_fps=8, sway=1.0))
far = [Walker(k, H * 0.12, W * 0.9 - i * 110, GY - 150, speed=0, cycle=0.7, phase=0.41 * i, hold_fps=8)
       for i, k in enumerate(kids[:7])]

def scroll(strip, offset):
    w = strip.shape[1]; o = int(offset) % w
    s = np.roll(strip, -o, axis=1)
    return s[:, :W] if w >= W else np.tile(s, (1, W // w + 1))[:, :W]

def houses(off, scale, gap, y):
    out = np.zeros((H, W), np.float32)
    hw = house.shape[1] * scale + gap
    k0 = int(math.floor(off / hw)) - 1
    for k in range(k0, k0 + int(W / hw) + 3):
        x = k * hw - off
        out = np.maximum(out, place_alpha(house, x + hw / 2, y, scale))
    return out

def frame(t, fi, warmth=1.0):
    cam = SPEED * t
    img = screen(t, warmth=warmth, lamp=(0.58, 0.40))
    sk = np.zeros((H, W), np.float32); sk[:sky.shape[0]] = scroll(sky, cam * 0.15)
    img = cast(img, sk, blur=3.2, density=0.45)
    img = cast(img, houses(cam * 0.45, 0.62, 260, H * 0.50), blur=2.2, density=0.62)
    cf = np.zeros((H, W), np.float32)
    for w in far: w.render_into(cf, t, W, H)
    img = cast(img, cf, blur=2.6, density=0.55)
    cm = np.zeros((H, W), np.float32)
    for w in row: w.render_into(cm, t, W, H)
    hold.render_into(cm, t, W, H); lead.render_into(cm, t, W, H)
    np.maximum(cm, scroll(GROUND, cam), out=cm)
    img = cast(img, cm, blur=0.6, density=0.97)
    img = film.apply(img, fi, t, strength=0.5, weave=True, flicker=0.4)
    return np.clip(img, 0, 255).astype(np.uint8)

for t in (0.0, 1.3):
    cv2.imwrite(f'{ROOT}/test/m3_{t}.jpg', frame(t, int(t * 30)), [cv2.IMWRITE_JPEG_QUALITY, 90])
cv2.imwrite(f'{ROOT}/test/m3_drained.jpg', frame(2.0, 60, warmth=0.12), [cv2.IMWRITE_JPEG_QUALITY, 90])
if len(sys.argv) > 1:
    p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', '30', '-i', '-',
                          '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-pix_fmt', 'yuv420p', f'{ROOT}/test/march_test.mp4'], stdin=subprocess.PIPE)
    for fi in range(int(float(sys.argv[1]) * 30)):
        p.stdin.write(frame(fi / 30, fi).tobytes())
    p.stdin.close(); p.wait()
print('ok')
