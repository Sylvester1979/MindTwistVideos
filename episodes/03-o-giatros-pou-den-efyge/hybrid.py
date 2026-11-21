"""Hybrid shot: sharp hi-res still + AI-animated region (static-camera i2v clip) + our camera/parallax.

The i2v clip was generated from the same still with a locked camera, so after a fine alignment (ECC)
only a feathered box (`box`, plate fractions l,t,r,b) is taken from the clip; everything else stays the
original sharp plate. The clip is time-stretched with frame blending to the shot length (calm slow motion).

usage: MT_ROOT=/home/user/ep03 python3 hybrid.py U2_window clip.mp4 out.mp4 secs l t r b
"""
import sys, os, subprocess, numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'engine')); sys.path.insert(0, HERE)
from eng import Plate, Motes, Film, W, H, FPS, ease, lerp
from look import grade

ROOT = os.environ.get('MT_ROOT', '/home/user/ep03')
key, clip, out, secs = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
l, t_, r, b = map(float, sys.argv[5:9])

P = Plate(key)                      # graded plate + depth
raw = cv2.imread(f'{ROOT}/img/{key}.png').astype(np.float32)
ph, pw = raw.shape[:2]

# ---- read clip frames, resize to plate
cap = cv2.VideoCapture(clip); frames = []
while True:
    ok, f = cap.read()
    if not ok: break
    frames.append(cv2.resize(f, (pw, ph), interpolation=cv2.INTER_CUBIC).astype(np.float32))
print('clip frames', len(frames), flush=True)

# ---- align clip to still (affine ECC on downscaled grey)
def grey(a, s=0.25):
    return cv2.resize(cv2.cvtColor(np.clip(a, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY), None, fx=s, fy=s).astype(np.float32)
Wm = np.eye(2, 3, dtype=np.float32)
try:
    _, Wm = cv2.findTransformECC(grey(raw), grey(frames[0]), Wm, cv2.MOTION_AFFINE,
                                 (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6), None, 5)
    Wm[:, 2] *= 4
except cv2.error as e:
    print('ECC failed, identity', e)
print('align', np.round(Wm, 4).tolist(), flush=True)
frames = [cv2.warpAffine(f, Wm, (pw, ph), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)
          for f in frames]

# ---- feathered mask
m = np.zeros((ph, pw), np.float32)
m[int(t_ * ph):int(b * ph), int(l * pw):int(r * pw)] = 1
m = cv2.GaussianBlur(m, (0, 0), 0.025 * pw)[..., None]

# ---- colour match clip to still inside the box (mean/std per channel, from frame 0)
sel = m[..., 0] > 0.5
mu_s, sd_s = raw[sel].mean(0), raw[sel].std(0) + 1e-3
mu_c, sd_c = frames[0][sel].mean(0), frames[0][sel].std(0) + 1e-3
frames = [(f - mu_c) / sd_c * sd_s + mu_s for f in frames]
still = P.img.copy()
gframes = [grade(f) for f in frames]

c0 = (pw / 2, ph / 2, 1.0)
c1 = (pw * 0.52, ph * 0.40, 1.30)
MO, FI = Motes(n=55, seed=4), Film(seed=8)
tmp = out[:-4] + '.part.mp4'
ff = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}',
                       '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-crf', '18', '-preset', 'veryfast',
                       '-pix_fmt', 'yuv420p', tmp], stdin=subprocess.PIPE)
n = int(secs * FPS); nc = len(gframes)
for fi in range(n):
    t = fi / FPS
    x = fi / (n - 1) * (nc - 1); i0 = int(x); i1 = min(nc - 1, i0 + 1); a = x - i0
    cf = gframes[i0] * (1 - a) + gframes[i1] * a
    P.img = still * (1 - m) + cf * m
    u = ease(fi / (n - 1))
    cam = (lerp(c0[0], c1[0], u), lerp(c0[1], c1[1], u), lerp(c0[2], c1[2], u))
    img = P.render(t, fi, cam, par=(lerp(0, 10, u), 0, lerp(0, 70, u)), focus=0.55, boil=0.0)
    img = MO.apply(img, t, 0.55)
    img = FI.apply(img, fi, t, strength=0.45, weave=False, flicker=0.4, grain=0.8)
    ff.stdin.write(np.clip(img, 0, 255).astype(np.uint8).tobytes())
ff.stdin.close()
if ff.wait() == 0:
    os.replace(tmp, out); print('ok', out)
